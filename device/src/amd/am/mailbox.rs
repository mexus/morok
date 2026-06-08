//! SR-IOV guest↔GIM mailbox (gfx9 `mxgpu_ai` protocol).
//!
//! Ten dwords in NBIO segment 2: TRN_DW0..3 (us → host), RCV_DW0..3 (host →
//! us), CONTROL (byte 0 = TX valid/ack, byte 1 = RX valid/ack), INT_CNTL.
//! These are absent from the bare-metal register tables (tinygrad never runs
//! virtualized), so offsets are vendored here from kernel `nbio_7_9_0_offset.h`.

use crate::error::{Error, RuntimeSnafu};

use super::pci::Bar;

type Result<T> = std::result::Result<T, Error>;

/// NBIO 7.9.0 segment-2 base dword index (SOC15 `nbio_7_9_0` BASE_IDX 2).
/// Hardcoded because the mailbox must run *before* discovery can be read: on a
/// VF, framebuffer access (hence the discovery table) is gated until the GIM
/// grants it via this very mailbox. Validated against discovery post-handshake.
/// (M0 confirmed: `0xd20 + 0xc3 = 0xde3 = RCC_CONFIG_MEMSIZE`.)
pub const NBIO_7_9_SEG2_BASE: u64 = 0xd20;

// Dword offsets within NBIO segment 2 (mxgpu_ai over NBIO 7.9).
const MAILBOX_MSGBUF_TRN_DW0: usize = 0x136;
const MAILBOX_MSGBUF_RCV_DW0: usize = 0x13a;
const MAILBOX_CONTROL: usize = 0x13e;
/// `RCC_DEV0_EPF0_RCC_IOV_FUNC_IDENTIFIER`: bit 0 = this function is a VF.
const RCC_IOV_FUNC_IDENTIFIER: usize = 0xc5;

const TRN_MSG_VALID: u8 = 1 << 0;
const TRN_MSG_ACK: u8 = 1 << 1;
const RCV_MSG_ACK: u8 = 1 << 1;

const POLL_ACK_MS: u64 = 500;
const POLL_MSG_MS: u64 = 6000;

/// Guest → host request ids (`enum idh_request`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Req {
    ReqGpuInitAccess = 1,
    RelGpuInitAccess = 2,
    ReqGpuFiniAccess = 3,
    RelGpuFiniAccess = 4,
    ReqGpuResetAccess = 5,
    ReqGpuInitData = 6,
}

/// Host → guest event ids (`enum idh_event`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Event {
    ClrMsgBuf = 0,
    ReadyToAccessGpu = 1,
    FlrNotification = 2,
    FlrNotificationCmpl = 3,
    Success = 4,
    Fail = 5,
    QueryAlive = 6,
    ReqGpuInitDataReady = 7,
}

/// The mailbox, bound to the NBIO instance-0 segment-2 base from discovery.
pub struct Mailbox {
    base: usize,
}

impl Mailbox {
    pub fn new(nbio_seg2_base: u64) -> Self {
        Self { base: nbio_seg2_base as usize }
    }

    /// Is this PCI function a VF (host GIM on the other end)?
    pub fn is_vf(&self, mmio: &Bar) -> bool {
        mmio.read_u32(self.base + RCC_IOV_FUNC_IDENTIFIER) & 1 != 0
    }

    fn ctl_trn(&self, mmio: &Bar) -> u8 {
        mmio.read_u8((self.base + MAILBOX_CONTROL) * 4)
    }

    fn set_valid(&self, mmio: &Bar, valid: bool) {
        mmio.write_u8((self.base + MAILBOX_CONTROL) * 4, if valid { TRN_MSG_VALID } else { 0 });
    }

    /// Ack the event the host just delivered.
    fn ack_rcv(&self, mmio: &Bar) {
        mmio.write_u8((self.base + MAILBOX_CONTROL) * 4 + 1, RCV_MSG_ACK);
    }

    /// Current host→guest event word, if any.
    pub fn peek_event(&self, mmio: &Bar) -> u32 {
        mmio.read_u32(self.base + MAILBOX_MSGBUF_RCV_DW0)
    }

    /// `xgpu_ai_mailbox_trans_msg`: drain stale ack, write DW0..3, raise
    /// valid, wait host ack, drop valid.
    pub fn send(&self, mmio: &Bar, req: Req, data: [u32; 3]) -> Result<()> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(POLL_ACK_MS);
        loop {
            self.set_valid(mmio, false);
            if self.ctl_trn(mmio) & TRN_MSG_ACK == 0 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
            if std::time::Instant::now() > deadline {
                return Err(Error::Runtime { message: "mailbox: stale ack never cleared".into() });
            }
        }
        for (i, d) in [req as u32, data[0], data[1], data[2]].into_iter().enumerate() {
            mmio.write_u32(self.base + MAILBOX_MSGBUF_TRN_DW0 + i, d);
        }
        self.set_valid(mmio, true);
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(POLL_ACK_MS);
        while self.ctl_trn(mmio) & TRN_MSG_ACK == 0 {
            snafu::ensure!(
                std::time::Instant::now() <= deadline,
                RuntimeSnafu { message: format!("mailbox: no ack for {req:?}") }
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        self.set_valid(mmio, false);
        Ok(())
    }

    /// Poll for `event`; acks it on arrival. Returns RCV_DW2 (checksum).
    pub fn wait_event(&self, mmio: &Bar, event: Event) -> Result<u32> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(POLL_MSG_MS);
        loop {
            if self.peek_event(mmio) == event as u32 {
                let dw2 = mmio.read_u32(self.base + MAILBOX_MSGBUF_RCV_DW0 + 2);
                self.ack_rcv(mmio);
                return Ok(dw2);
            }
            snafu::ensure!(
                std::time::Instant::now() <= deadline,
                RuntimeSnafu {
                    message: format!("mailbox: timeout waiting {event:?} (rcv={:#x})", self.peek_event(mmio))
                }
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    /// Request exclusive init access; host replies READY_TO_ACCESS_GPU.
    pub fn request_init_access(&self, mmio: &Bar) -> Result<u32> {
        self.send(mmio, Req::ReqGpuInitAccess, [0; 3])?;
        self.wait_event(mmio, Event::ReadyToAccessGpu)
    }

    pub fn release_init_access(&self, mmio: &Bar) -> Result<()> {
        self.send(mmio, Req::RelGpuInitAccess, [0; 3])
    }
}
