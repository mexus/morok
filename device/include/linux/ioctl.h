/* Vendored minimal <linux/ioctl.h> shim (asm-generic _IOC encoding).
 *
 * kfd_ioctl.h includes <linux/ioctl.h> for the _IO/_IOR/_IOW/_IOWR macros it
 * uses to define the AMDKFD_IOC_* request codes. bindgen cannot const-fold
 * those macro expansions, so it never emits the AMDKFD_IOC_* constants — the
 * ioctl numbers are computed Rust-side by `nix::ioctl_readwrite!` (see
 * sys/ioctl.rs). These macros therefore only need to let the header *parse*;
 * the asm-generic layout below (shared by x86-64 and aarch64, the in-scope
 * targets) is the faithful definition regardless. Vendored so the build is
 * hermetic — no system kernel headers required on any platform. */
#ifndef _SVOD_VENDORED_LINUX_IOCTL_H
#define _SVOD_VENDORED_LINUX_IOCTL_H

#include <linux/types.h>

#define _IOC_NRBITS   8
#define _IOC_TYPEBITS 8
#define _IOC_SIZEBITS 14
#define _IOC_DIRBITS  2

#define _IOC_NRSHIFT   0
#define _IOC_TYPESHIFT (_IOC_NRSHIFT + _IOC_NRBITS)
#define _IOC_SIZESHIFT (_IOC_TYPESHIFT + _IOC_TYPEBITS)
#define _IOC_DIRSHIFT  (_IOC_SIZESHIFT + _IOC_SIZEBITS)

#define _IOC_NONE  0U
#define _IOC_WRITE 1U
#define _IOC_READ  2U

#define _IOC(dir, type, nr, size)         \
	(((dir) << _IOC_DIRSHIFT) |       \
	 ((type) << _IOC_TYPESHIFT) |     \
	 ((nr) << _IOC_NRSHIFT) |         \
	 ((size) << _IOC_SIZESHIFT))

#define _IO(type, nr)         _IOC(_IOC_NONE, (type), (nr), 0)
#define _IOR(type, nr, size)  _IOC(_IOC_READ, (type), (nr), sizeof(size))
#define _IOW(type, nr, size)  _IOC(_IOC_WRITE, (type), (nr), sizeof(size))
#define _IOWR(type, nr, size) _IOC(_IOC_READ | _IOC_WRITE, (type), (nr), sizeof(size))

#endif /* _SVOD_VENDORED_LINUX_IOCTL_H */
