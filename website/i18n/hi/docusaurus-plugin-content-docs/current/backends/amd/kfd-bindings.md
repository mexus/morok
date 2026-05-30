---
sidebar_label: KFD Bindings
---

# KFD Bindings

बैकएंड kernel से `/dev/kfd` पर `ioctl` calls के एक छोटे, निश्चित set के माध्यम से बात करता
है। यह पेज कवर करता है कि वे calls Rust में कैसे bind होती हैं, बैकएंड असल में किनका उपयोग
करता है, GPU nodes कैसे खोजे जाते हैं, और वह allocation flow जो एक `ioctl` को एक mapped GPU
buffer में बदल देता है। बैकएंड HIP-based के बजाय KFD-direct *क्यों* है, इसके लिए
[अवलोकन](./overview.md) देखें।

---

## Bindings कैसे जनरेट होती हैं

KFD का ABI एक C header है, `kfd_ioctl.h`, जिसे kernel से verbatim
`device/include/kfd_ioctl.h` में vendor किया गया है (upstream AMD फ़ाइल, अपने ABI version
history के साथ पूरी)। उससे build time पर `bindgen` द्वारा Rust bindings जनरेट होती हैं:

- `device/build.rs` `bindgen` को **केवल Linux पर** चलाता है, ठीक-ठीक उन KFD types और
  constants को allow-list करते हुए जिनकी बैकएंड को ज़रूरत है:

  ```text
  allowlist_type:  kfd_ioctl_.*_args, kfd_event_data,
                   kfd_hsa_memory_exception_data, kfd_hsa_hw_exception_data,
                   kfd_memory_exception_failure, __u\d+, __s\d+, …
  allowlist_var:   KFD_IOC_.*, AMDKFD_IOC_.*, KFD_MAX_QUEUE_PERCENTAGE, …
  ```

  `.derive_default(true).layout_tests(false).generate_comments(false)` के साथ। output
  `$OUT_DIR/kfd_sys.rs` में लिखा जाता है।

- **non-Linux** hosts पर `build.rs` इसके बजाय एक empty stub लिखता है, ताकि module हमेशा
  compile हो (तब AMD path runtime पर `Err(NoAmdGpu)` return करता है)।

- `device/src/amd/sys/kfd.rs` एक one-liner है जो जनरेट की गई फ़ाइल को `include!` करता है।

:::note hand-written ioctl macros क्यों
`bindgen` argument *structs* emit करता है लेकिन `_IOWR` ioctl-number macros नहीं। वे
`device/src/amd/sys/ioctl.rs` में `nix::ioctl_readwrite!` का उपयोग करते हुए हाथ से declare
किए जाते हैं, type code `KFD_IOCTL_BASE = b'K'` के साथ। हर ioctl `readwrite` declare होता है,
यहाँ तक कि जहाँ header `_IOR`/`_IOW` कहता है — KFD argument struct को in/out मानता है, और
kernel दोनों दिशाओं को सहन कर लेता है।
:::

---

## बैकएंड जो ioctls उपयोग करता है

`(group, opcode, args)` triples सीधे `kfd_ioctl.h` से आते हैं। ये वही हैं जिनके live call
sites हैं:

| Wrapper | Op | किसके लिए उपयोग होता है |
|---|---|---|
| `kfd_get_version` | `0x01` | KFD ABI version पढ़ें (`RUNTIME_ENABLE` को gate करता है) |
| `kfd_create_queue` | `0x02` | `setup_ring` — एक compute/SDMA queue बनाएँ |
| `kfd_destroy_queue` | `0x03` | `teardown_ring` |
| `kfd_create_event` | `0x08` | queue-signal, memory-fault, और hw-exception events; event page bind करना |
| `kfd_wait_events` | `0x0C` | `wait_events` — completion / fault events पर block करें |
| `kfd_acquire_vm` | `0x15` | GPU के लिए DRM render fd को इस process के VM के रूप में register करें |
| `kfd_alloc_memory_of_gpu` | `0x16` | `alloc_raw` — VRAM/GTT allocate करें |
| `kfd_free_memory_of_gpu` | `0x17` | `free_raw` |
| `kfd_map_memory_to_gpu` | `0x18` | एक allocation को GPU page table में bind करें |
| `kfd_unmap_memory_from_gpu` | `0x19` | `free_raw` |
| `kfd_runtime_enable` | `0x25` | runtime enable करें (केवल KFD ABI ≥ 1.14) |

कुछ और (`set_memory_policy`, `get_clock_counters`, `get_process_apertures`,
`update_queue`, `destroy_event`, `set_event`, `reset_event`) completeness के लिए declare
किए गए हैं लेकिन फ़िलहाल call नहीं होते।

### Device bring-up sequence

`KfdIface::open` (`device/src/amd/iface.rs`) इन्हें क्रम में जारी करता है, tinygrad के
`ops_amd.py` को mirror करते हुए:

```text
open /dev/kfd  (process-shared, one fd)
open /dev/dri/renderD<minor>  (per node — the DRM render fd)
   │
   ├─ GET_VERSION            → capture ABI version
   ├─ ACQUIRE_VM(drm_fd)     → register this fd as the process VM for the GPU
   ├─ RUNTIME_ENABLE         → only if ABI ≥ 1.14
   ├─ (event page: alloc + bind once per process, map per device)
   └─ CREATE_EVENT × 3       → queue-signal, memory-fault, hw-exception
```

DRM render fd दिलचस्प है: यहाँ **कोई DRM ioctls नहीं** हैं। `drm_fd` को केवल दो तरीक़ों से
उपयोग किया जाता है — `ACQUIRE_VM` में *by number* pass किया जाता है, और host-visible
mappings के लिए `mmap` fd के रूप में। doorbell, इसके विपरीत, KFD fd से `mmap` किया जाता है।

---

## Topology: GPU ढूँढना

GPU nodes को sysfs से enumerate किया जाता है, किसी ioctl के माध्यम से नहीं।
`device/src/amd/topology.rs`
`/sys/devices/virtual/kfd/kfd/topology/nodes/<N>/properties` पढ़ता है — प्रति line एक
`key value` pair — और एक `Vec<AmdNode>` return करता है, CPU nodes (`gpu_id == 0`) को
छोड़ते हुए। यह कभी panic नहीं करता: बिना `/dev/kfd` वाला host एक empty vector देता है, जिसे
device factory एक साफ़ `Err(NoAmdGpu)` में बदल देता है।

हर `AmdNode` वे fields रखता है जिनकी बाक़ी बैकएंड को ज़रूरत होती है:
`gpu_id`, `drm_render_minor`, `gfx_target_version` (जैसे `110000` → gfx1100),
`simd_count`, `simd_per_cu`, `max_waves_per_simd`, `num_xcc`, `lds_size_in_kb`,
`max_slots_scratch_cu`, और उनके साथी — ये scratch sizing और PM4-बनाम-AQL decision को
feed करते हैं।

:::tip बिना hardware के testing
sysfs root को **`SVOD_KFD_TOPOLOGY`** से override किया जा सकता है, इसलिए parser को बिना
किसी GPU की मौजूदगी के एक fabricated nodes directory के विरुद्ध unit-test किया जाता है।
:::

---

## Allocation flow

हर buffer वही चार-चरणीय path follow करता है, जो `KfdIface::alloc_raw` में एक बार
implement है:

```text
1. reserve_va(size)                     mmap(PROT_NONE, …) — reserve host VA
2. ALLOC_MEMORY_OF_GPU(va, size, flags) → returns handle + mmap_offset
3. if host-visible:                     mmap(va, …, MAP_FIXED, drm_fd, offset)
4. MAP_MEMORY_TO_GPU(handle)            bind into the GPU page table
```

host VA पहले एक anonymous `PROT_NONE` mapping से reserve किया जाता है ताकि step 3 का
host-visible `mmap` ठीक उसी address पर land कर सके (`MAP_FIXED`)। Free करना इसे उलट देता है:
`UNMAP_MEMORY_FROM_GPU` → `munmap` → `FREE_MEMORY_OF_GPU`।

### Allocation flavors

`alloc_raw` एक `AllocKind` लेता है जो KFD flag set चुनता है — वह एकमात्र जगह जहाँ वे flags
compose होते हैं:

| `AllocKind` | Flags | किसके लिए उपयोग होता है |
|---|---|---|
| `DeviceVram { executable }` | `VRAM \| WRITABLE \| NO_SUBSTITUTE` (+ code के लिए `EXECUTABLE`, + host-visible होने पर `PUBLIC`) | Tensor data, code objects, scratch |
| `UncachedGtt` | `GTT \| WRITABLE \| EXECUTABLE \| NO_SUBSTITUTE \| PUBLIC \| COHERENT \| UNCACHED` | Command rings, GART pages, signal slots, event page |

`UNCACHED | COHERENT` GTT flavor मायने रखता है: command ring और signal slots CPU और GPU के
बीच तुरंत visible होने चाहिए, वरना host एक ऐसे completion value पर हमेशा के लिए spin करता
रहता है जो GPU L2 में अटका रह जाता है। KFD एक plain-VRAM ring पर `CREATE_QUEUE` को `EINVAL`
के साथ reject कर देता है।

### हर जगह host-visible

चूँकि कोई SDMA queue नहीं है, allocator (`device/src/amd/allocator.rs`) हर buffer पर
`cpu_access = true` force करता है: `has_sdma_queue()` हमेशा `false` होता है, इसलिए `_alloc`
इसे OR कर देता है। इसलिए copies (`_copyin`/`_copyout`/`_transfer`) एक `synchronize()` के बाद
सादे host `memmove` होती हैं। generic `LruAllocator` (`device/src/allocator.rs`) freed
buffers को `(size, BufferSpec)` के अनुसार pool करता है; `nolru` spec code objects, scratch,
और queue infrastructure के लिए pool को bypass करता है।

:::note Process-shared state
`/dev/kfd` प्रति process एक बार खोला जाता है और सभी devices द्वारा साझा होता है (events को
उस fd के विरुद्ध id से address किया जाता है)। 0x8000-byte KFD **event page** भी इसी तरह
प्रति process एक बार allocate और bind होता है; बाद के devices उसे केवल अपने अपने `gpu_id`
में `MAP_MEMORY_TO_GPU` करते हैं। दोनों tinygrad के per-process model को mirror करते हैं।
:::

---

## यह क्यों ज़रूरी है

पूरी kernel-facing surface है **एक vendored header, ग्यारह ioctls, और एक sysfs parser**।
यही पूरी वजह है कि बैकएंड ROCm userspace stack से बच सकता है: kernel ABI छोटा और stable है,
इसलिए उसे सीधे bind करना HIP को integrate करने से कम code है — और यह
[backend seam](./overview.md) को इस तरह खुला छोड़ देता है कि उसके ऊपर किसी चीज़ को छुए बिना
KFD को userspace [AM driver](./am-driver.md) से swap किया जा सके।
