/* Bindgen wrapper for the HSA runtime + AMD-vendor queue/kernel-code ABI used
 * by the KFD-direct AQL path. The headers under `hsa/` are vendored verbatim
 * from ROCm 7.2.0 (/opt/rocm/include/hsa, AMD copyright 2014-2020) so the build
 * does not depend on a ROCm install. Only the structs/enums allowlisted in
 * build.rs are emitted; everything else in hsa.h is dropped. */
#include "hsa/hsa.h"
#include "hsa/amd_hsa_queue.h"
#include "hsa/amd_hsa_kernel_code.h"
#include "hsa/amd_hsa_signal.h"
