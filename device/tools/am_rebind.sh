#!/usr/bin/env bash
# Return the MI300X VF to amdgpu (restores the KFD backend + rocm-smi).
set -euo pipefail
BDF="${1:-$(for d in /sys/bus/pci/devices/*; do [ "$(cat "$d/vendor" 2>/dev/null)" = 0x1002 ] && basename "$d" && break; done)}"
echo "rebinding amdgpu to $BDF"
# Clear the override pin set by am_unbind.sh, then let the bus re-probe.
echo "" | sudo tee "/sys/bus/pci/devices/$BDF/driver_override" >/dev/null 2>&1 || true
echo "$BDF" | sudo tee /sys/bus/pci/drivers/amdgpu/bind >/dev/null 2>&1 \
  || echo "$BDF" | sudo tee /sys/bus/pci/drivers_probe >/dev/null
sleep 3
echo "driver now: $(basename "$(readlink "/sys/bus/pci/devices/$BDF/driver" 2>/dev/null)" 2>/dev/null || echo none)"
