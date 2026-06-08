/* Vendored minimal <linux/types.h> shim.
 *
 * The vendored KFD UAPI header (kfd_ioctl.h) uses the kernel fixed-width
 * aliases __u8..__u64 / __s8..__s64. On Linux these come from the system
 * <linux/types.h>; vendoring them makes the bindgen build hermetic on every
 * platform (Linux + macOS) with no kernel headers installed. */
#ifndef _SVOD_VENDORED_LINUX_TYPES_H
#define _SVOD_VENDORED_LINUX_TYPES_H

#include <stdint.h>

typedef uint8_t  __u8;
typedef int8_t   __s8;
typedef uint16_t __u16;
typedef int16_t  __s16;
typedef uint32_t __u32;
typedef int32_t  __s32;
typedef uint64_t __u64;
typedef int64_t  __s64;

#endif /* _SVOD_VENDORED_LINUX_TYPES_H */
