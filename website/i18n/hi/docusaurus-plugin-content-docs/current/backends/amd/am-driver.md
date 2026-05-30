---
sidebar_label: AM Driver
---

# AM Driver (Userspace)

**AM** driver एक दूसरा [`AmdIface`](./overview.md) बैकएंड है जो GPU के PCI BARs को सीधे drive
करता है, kernel `amdgpu`/KFD driver को पूरी तरह bypass करते हुए। यह tinygrad के userspace AM
driver का एक port है। प्रेरणा ठोस है: lock-free [multi-queue dispatch](./queues-and-dispatch.md)
path भारी concurrent load के तहत kernel के MES/runlist scheduler को overload कर देता है और
kernel को crash कर सकता है। यदि हम GPU के मालिक हैं — page tables, firmware, scheduling — तो
kernel कभी dispatch path में नहीं होता और overload नहीं हो सकता।

:::caution Work in progress — अभी selectable नहीं
यह पेज आज जो मौजूद है और बाक़ी के लिए roadmap, दोनों का दस्तावेज़ीकरण करता है।
**`SVOD_AMD_BACKEND=am` फ़िलहाल एक error देता है** (`device.rs` केवल `kfd` स्वीकार करता है)।
जो implement है वह है unprivileged, GPU-free *logic*; privileged hardware bring-up स्थगित है।
नीचे के sections हर टुकड़े की status को explicit रूप से चिह्नित करते हैं।
:::

कोड `device/src/amd/am/` के अंतर्गत रहता है। यह Linux पर unconditionally compile होता है
(pure logic, कोई extra dependencies नहीं), इसलिए यह हमेशा type-checked, linted, और
unit-tested होता है — बैकएंड को *runtime* पर चुना जाता है, कभी किसी ऐसे cargo feature के पीछे
नहीं जो rot हो सकता है।

---

## kernel ने हमारे लिए क्या किया

KFD ने बैकएंड को तीन चीज़ें दीं। AM को हर एक ख़ुद उपलब्ध करानी है:

| KFD ने प्रदान किया | AM को करना है |
|---|---|
| VRAM allocation + GPU page-table mapping | एक GMMU: VA allocator + 4-level page-table walker + PTE encoding |
| Queue creation (MES/HQD setup) | MQD लिखें, MEC enable करें, doorbell map करें |
| Memory + firmware bring-up | PCI BAR mapping, IP discovery, PSP firmware load |

महत्वपूर्ण रूप से, seam के *ऊपर* सब कुछ — PM4/AQL packet builders, ring, signals, kernarg
arena, timeline, और back-pressure — **अपरिवर्तित** carry over होता है। AM केवल पाँच
`AmdIface` methods को बदलता है।

---

## आज क्या मौजूद है (built और tested)

pure-logic आधा बिना GPU के implement और unit-tested है, page tables को एक injectable
`PhysMem` trait से back करते हुए (tests में एक plain buffer, असली driver में BAR-mapped
VRAM)।

| Module | यह क्या implement करता है | Status |
|---|---|---|
| `am/mm/tlsf.rs` | TLSF (Two-Level Segregated Fit) allocator — tinygrad के `TLSFAllocator` का port | **Done** + unit tests + एक proptest |
| `am/mm/pagetable.rs` | GMMU geometry + PTE/PDE bit encoding | **gfx11 के लिए Done** + tests |
| `am/mm/manager.rs` | `MemoryManager`: VA alloc, 4-level page-table walk, huge-page selection, table reclaim, `valloc`/`vfree` | **Done** + एक fake VRAM के विरुद्ध tests |
| `am/regs.rs` | `RegDef`/`RegField` types + `select(prefix, ip_ver)` resolver | **Done** + tests |
| `am/regs_gen.rs` | Vendored register tables (`GC_11_5_0`, `MMHUB_3_3_0`, `MP_14_0_2`, …) | **Generated और committed** |

### GMMU

page-table geometry **4-level / 48-bit** है (`va_shifts = [12, 21, 30, 39]`), एक आकार जो
**gfx9/11/12 में साझा है** — इसलिए geometry ख़ुद arch पर branch नहीं करती। केवल leaf PTE
encoding (विशेष रूप से MTYPE memory-type field) arch-specific है। `MemoryManager` तीन TLSF
sub-allocators (VA space, physical VRAM, page-table pool) चलाता है और table को `Inspect` /
`Create` / `Free` modes में walk करता है, जहाँ alignment अनुमति देती है वहाँ huge pages चुनते
हुए और unmap पर empty tables को reclaim करते हुए।

### Register tables एक-बार generate होते हैं, फिर vendor किए जाते हैं

tinygrad एक कभी-कभी-अनुपस्थित submodule है, इसलिए build को कभी उस पर निर्भर नहीं होना चाहिए।
इसके बजाय `device/tools/gen_am_regs.py` को एक arch जोड़ते या update करते समय **manually**
चलाया जाता है: यह tinygrad के `autogen/am/regs.py` को parse करता है और committed
`am/regs_gen.rs` emit करता है। `regs.rs` बस उसे `include!` करता है। boot पर सही table को
discovered `ip_ver` से चुना जाता है (`select` वह सबसे बड़ा version `≤ ip_ver` चुनता है जो वही
major साझा करता है — tinygrad का `import_module` नियम)। एक arch जोड़ना generator की module
list को widen करना और उसे re-run करना है — कोई build या runtime logic change नहीं।

---

## क्या स्थगित है (अभी tree में नहीं)

privileged bring-up को root/caps चाहिए (`amdgpu` unbind करना, PCI BARs को `mmap` करना,
mode-1 reset) और यह अभी source में मौजूद नहीं है:

- **AMDev orchestrator** (BAR map, boot sequence);
- **PCI/BAR** access और **IP discovery** parsing;
- **PSP firmware load** (सबसे-अधिक-जोखिम वाला subsystem — एक version-specific handshake);
- **IP-block** modules (SOC / GMC / IH / PSP / SMU / GFX / SDMA);
- **`AmIface`** implementor जो इन सबको seam से जोड़ता है।

*implemented* page-table module के भीतर, केवल **gfx11/RDNA3** live है: gfx9 (VG10) और gfx12
PTE-encoding paths जान-बूझकर `unimplemented!` panics हैं, हर एक एक ऐसे test से guarded है जो
assert करता है कि वह panic करता है — constants captured हैं लेकिन अभी तक hardware-validated
नहीं।

---

## Target hardware और arch parametrization

register और page-table target है **gfx1151 — "Strix Halo" APU** (जो GC 11.5.1 report करता है
→ `gc_11_5_0` table)। driver को उसी तरह parametrize किया गया है जैसे tinygrad का: **boot पर
IP discovery से पढ़े गए `ip_ver` tuples द्वारा**, न कि एक हाथ से रखे गए arch enum द्वारा। Arch
differences को shared modules के अंदर छोटे inline `if ip_ver >= (X, Y, Z)` branches प्लस
version-keyed register tables होना चाहिए — इसलिए gfx12 ज़्यादातर एक data addition बन जाता है
और gfx9 accommodate तो किया जाता है पर स्थगित है।

:::note इस मशीन पर bring-up क्यों स्थगित है
असली hardware एक Strix Halo APU है जो **primary display GPU** भी है। AM को `amdgpu` unbind
करना और exclusive ownership लेना पड़ता है, जो display को मार देता है; और tinygrad का AM discrete
RDNA3/4 device IDs को whitelist करता है, इस APU को नहीं। इसलिए इस मशीन पर validate करने के
लिए कोई काम करने वाला AM oracle नहीं है। privileged bring-up (नीचे के phases) एक
external/discrete GPU पर स्थगित है जहाँ tinygrad AM एक सिद्ध reference है। इस बीच,
[single-queue KFD mode](./queues-and-dispatch.md) पहले ही उस kernel crash को fix कर चुका है
जिसने AM को प्रेरित किया, इसलिए अंतरिम में कुछ भी blocked नहीं है।
:::

---

## Roadmap

हर phase स्वतंत्र रूप से testable है, उसी card पर tinygrad AM के साथ per-phase oracle के रूप
में:

| Phase | Milestone |
|---|---|
| **A** | PCI + discovery (read-only): `amdgpu` unbind करें, BARs map करें, IP discovery parse करें, हर value को tinygrad के विरुद्ध diff करें |
| **B** | regs + GMC page tables: `valloc` + एक buffer map करें, PTE वापस पढ़ें, BAR के माध्यम से data round-trip करें |
| **C** | PSP firmware load (risk gate): sOS bootloader handshake, TMR, per-IP firmware load — एक tinygrad transcript के विरुद्ध dword-for-dword diffed |
| **D** | GFX MEC + `setup_ring`: v11 compute MQD लिखें, MEC enable करें, doorbell map करें (`CP_HQD_ACTIVE == 1`) |
| **E** | एक kernel dispatch करें: AM-backed core पर पूरे मौजूदा ऊपरी हिस्से को reuse करें |
| **F** | concurrency + de-stub: असली interrupt handler, max clocks, वह workload चलाएँ जिसने KFD को crash किया — जो अब crash नहीं हो सकता, kernel bypass है |

एक बार Phase F crash-inducing concurrency को साफ़-साफ़ चला ले, AM multi-queue/streaming
workloads के लिए recommended mode बन जाता है, जबकि KFD (single-queue) portable fallback बना
रहता है। gfx12/RDNA4 फिर एक सस्ता follow-on है (register tables widen करें + `gc >= (12,0,0)`
branches जोड़ें); gfx9/CDNA एक बड़ा, बाद का प्रयास है।

---

## यह क्यों ज़रूरी है

AM driver उस kernel-overload समस्या का असली उत्तर है जिसे [single-queue mode](./queues-and-dispatch.md)
केवल sidestep करता है। महँगे, GPU-free हिस्से — GMMU और register tables — पहले ही built और
tested हैं, इसलिए बचा हुआ काम privileged bring-up है, जो design के बजाय hardware पर gated है।
और चूँकि यह उसी पाँच-method [seam](./overview.md) के पीछे slot होता है, dispatch, compile, या
graph machinery में से किसी को इसके उतरने पर बदलना नहीं पड़ता।
