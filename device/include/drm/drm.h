/* Vendored empty <drm/drm.h> shim.
 *
 * kfd_ioctl.h includes <drm/drm.h> but uses no drm_* type from it — the only
 * DRM-adjacent fields are plain `__u32 drm_fd`. The include is vestigial; this
 * shim resolves it so the header parses without the system libdrm UAPI headers,
 * keeping the bindgen build hermetic on every platform. */
#ifndef _SVOD_VENDORED_DRM_DRM_H
#define _SVOD_VENDORED_DRM_DRM_H

#include <linux/types.h>

#endif /* _SVOD_VENDORED_DRM_DRM_H */
