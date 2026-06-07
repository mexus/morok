#!/usr/bin/env bash
# Release the MI300X VF from amdgpu so the AM userspace driver can own it.
# Recovery: device/tools/am_rebind.sh (or reboot the VM).
set -euo pipefail
BDF="${1:-$(for d in /sys/bus/pci/devices/*; do [ "$(cat "$d/vendor" 2>/dev/null)" = 0x1002 ] && basename "$d" && break; done)}"
echo "unbinding amdgpu from $BDF"
# Block auto-rebind: pin driver_override to none before unbinding.
echo "none" | sudo tee "/sys/bus/pci/devices/$BDF/driver_override" >/dev/null
if [ -e "/sys/bus/pci/devices/$BDF/driver" ]; then
  # amdgpu's SR-IOV teardown (REL-access mailbox + queue/IH/VRAM teardown) can
  # take tens of seconds; bound by `timeout` so a stall surfaces instead of
  # parking forever in the kernel.
  echo "unbind may take up to ~120s (amdgpu VF teardown)..."
  if ! echo "$BDF" | sudo timeout 120 tee "/sys/bus/pci/devices/$BDF/driver/unbind" >/dev/null; then
    echo "ERROR: unbind did not complete within 120s — VF teardown stalled" >&2
    echo "driver still: $(basename "$(readlink "/sys/bus/pci/devices/$BDF/driver" 2>/dev/null)" 2>/dev/null || echo none)" >&2
    exit 1
  fi
fi
echo "driver now: $(basename "$(readlink "/sys/bus/pci/devices/$BDF/driver" 2>/dev/null)" 2>/dev/null || echo none)"
