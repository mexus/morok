---
sidebar_label: Phase 2 — Expander
---

# Phase 2: Expander

**गोल**: ऑप्टिमाइज़ेशन primitives (UNROLL/UPCAST) को एक्सप्लिसिट ऑपरेशन में बदलें।

---

## Stage 8: Post-Opt Symbolic

> **स्टेज एक नज़र में**
>
> **गोल**: ऑप्टिमाइज़ेशन के बाद symbolic सिम्प्लीफ़िकेशन
> **मुख्य Patterns**: WHERE मूवमेंट, constant folding
> **प्रभाव**: बेहतर load combining और वेक्टराइज़ेशन सक्षम करता है

**यह क्या करता है**: ऑप्टिमाइज़ेशन के बाद symbolic सिम्प्लीफ़िकेशन, साथ में WHERE मूवमेंट।

**यह क्यों ज़रूरी है**: WHERE ऑपरेशन `if` स्टेटमेंट जैसे हैं। यह स्टेज `if` चेक को load के बाद से load के पहले ले जाता है। जब कंडीशन false हो, हार्डवेयर loading स्किप कर सकता है — मेमोरी बैंडविड्थ बचती है।

**Pattern**: `sym + pm_move_where_on_load`

```text
// Before: WHERE guards a load
WHERE(valid, LOAD(index), alt)

// After: validity moved to INDEX
LOAD(INDEX(ptr, idx, valid=valid), alt)
```

Validity को INDEX में मूव करने से बेहतर load combining और वेक्टराइज़ेशन मिलता है।

**नोट**: यह pattern तभी मैच होता है जब alternative वैल्यू `0` हो। ट्रांसफ़ॉर्मेशन में कॉम्प्लेक्स clause एनालिसिस होता है: duplicate डिटेक्शन, range डिपेंडेंसी चेक, और data-dependent load वेरिफ़िकेशन।

**नोट**: Morok इम्प्लीमेंटेशन `valid=` के बजाय `gate=` इस्तेमाल करता है (Index struct में `gate` फ़ील्ड है)। कॉन्सेप्ट एक ही है।

**Morok**: `pm_move_where_on_load()` in `symbolic/patterns.rs`

---

## Stage 9: Expander

> **स्टेज एक नज़र में**
>
> **गोल**: UNROLL/UPCAST को एक्सप्लिसिट ऑपरेशन में बदलें
> **मुख्य कॉन्सेप्ट**: UNROLL, CONTRACT, pattern ऑर्डर
> **प्रभाव**: वेक्टराइज़ेशन एक्सप्लिसिट बनाता है और हार्डवेयर के लिए तैयार करता है

**यह क्या करता है**: UNROLL/UPCAST ऑप्टिमाइज़ेशन primitives को एक्सप्लिसिट ऑपरेशन में ट्रांसफ़ॉर्म करता है।

**यह क्यों ज़रूरी है**: UPCAST और UNROLL इंटेंट मार्क करते हैं — हम क्या करना चाहते हैं। यह स्टेज उस इंटेंट को एक्सप्लिसिट बनाता है ताकि हार्डवेयर वाकई कर सके।

**Pattern**: `symbolic_simple() + pm_pre_expander + pm_group_for_reduce + expander`

नोट: Morok इस स्टेज पर `symbolic_simple()` इस्तेमाल करता है (`sym` नहीं) क्योंकि `symbolic()` Stage 4 पर पहले ही चल चुका है। Tinygrad `sym` इस्तेमाल करता है जिसमें अतिरिक्त patterns शामिल हैं।

> **ज़रूरी: Pattern Precedence**

Patterns कम्बाइन होकर fixpoint तक चलते हैं। ऑर्डर तय करता है कि कई मैच होने पर कौन सा pattern पहले ट्राई हो:
1. `sym` पहले (symbolic सिम्प्लीफ़िकेशन)
2. `pm_pre_expander` दूसरा (UPCAST/UNROLL ranges कन्वर्ट करता है)
3. `pm_group_for_reduce` तीसरा (GROUP_REDUCE axis हैंडल करता है)
4. `expander` आखिर में (मुख्य expansion)

गलत precedence से गलत वेक्टराइज़ेशन या reduction scoping हो सकती है।

**UNROLL और CONTRACT**:

UNROLL और CONTRACT साथ मिलकर काम करते हैं:

```text
UNROLL: "Take this one thing and make N copies for different positions"
Example:  x → [x_0, x_1, x_2, x_3]

CONTRACT: "Take these N things and combine them back"
Example:  [a, b, c, d] → one vector containing all four
```

साथ में: UPCAST वेक्टराइज़ करने का इंटेंट मार्क करता है → UNROLL एक्सपैंड करता है → CONTRACT कम्बाइन करता है।

**UPCAST range → VECTORIZE**:
```text
// Before: UPCAST marks vectorization intent
RANGE(end=4, UPCAST)
      ↓ [pm_pre_expander]
// Step 1: Convert to UNROLL with constant indices
UNROLL(VCONST([0, 1, 2, 3]))
      ↓ [expander]
// Step 2: Expand operations with UNROLL sources
// Operations now have unrolled sources
      ↓ [CONTRACT or implicit]
// After: explicit VECTORIZE
VECTORIZE(op[0], op[1], op[2], op[3])
```

**UNROLL range → repeated ऑपरेशन**:

जब हम "ऑपरेशन डुप्लीकेट होते हैं" कहते हैं, तो ऐसा नहीं है कि कॉपी-पेस्ट होता है। कम्पाइलर एक सिंगल SIMD इंस्ट्रक्शन बनाता है जो सभी N एलिमेंट एक साथ प्रोसेस करती है। SIMD रजिस्टर को 4 नंबर रखने वाला बॉक्स सोचें; दो बॉक्स जोड़ने से सभी 8 नंबर एक साथ जुड़ते हैं।

```text
// Before: UPCAST marks vectorization intent
RANGE(end=3, UPCAST)
      ↓ [pm_pre_expander]
// Step 1: Convert to UNROLL
UNROLL(VCONST([0, 1, 2]))
      ↓ [expander]
// Step 2: Operations expand to handle all positions
// After: operations processed together (not duplicated)
UNROLL([op_at_0, op_at_1, op_at_2])
```

**UNROLL/END/CONTRACT इंटरैक्शन**:
```text
Before: END(STORE(...), [RANGE(UPCAST)])
             ↓ [pm_pre_expander]
Step 1: END(STORE(...), [UNROLL(VCONST([0,1,2,3]))])
             ↓ [expander]
Step 2: END(CONTRACT(STORE(...×4)), [])
```

**AFTER/END से ब्रॉडकास्ट**:
```text
// Broadcast VECTORIZE (all elements identical)
AFTER(VECTORIZE([x, x, x, x]), deps) → VECTORIZE([AFTER(x, deps), AFTER(x, deps), ...])
```

**GROUP_REDUCE हैंडलिंग** (`pm_group_for_reduce`):

GROUP_REDUCE tensor core reductions के लिए एक स्पेशल axis type है:

```text
// Before: REDUCE with GROUP_REDUCE ranges
REDUCE(src, [range(GROUP_REDUCE)])
           ↓ [pm_group_for_reduce]
// After: Shared memory reduction pattern
1. Track upstream LOCAL ranges
2. BUFFERIZE result with group ranges (AddrSpace.LOCAL)
3. INDEX into buffer with transformed ranges
4. Final REDUCE with axes (range_id+100, AxisType.REDUCE)
```

यह shared memory से एफ़िशिएंट tensor core accumulation सक्षम करता है।

**Morok**: `expand.rs`

---

## Stage 10: Add Local Buffers

> **स्टेज एक नज़र में**
>
> **गोल**: फ़ास्ट मेमोरी (shared / L1) के लिए बफ़र तैयार करें
> **मुख्य Patterns**: Locals के साथ bufferize, hints एक्सट्रैक्ट करें
> **प्रभाव**: बार-बार एक्सेस होने वाला डेटा फ़ास्ट मेमोरी में रहता है

**यह क्या करता है**: लोकल मेमोरी यूज़ के लिए बफ़र तैयार करता है और codegen-स्पेसिफ़िक क्लीनअप अप्लाई करता है।

**यह क्यों ज़रूरी है**: **लोकल बफ़र** = कम्प्यूट यूनिट के पास फ़ास्ट मेमोरी:
- GPU: Shared memory (LDS) — global memory से 100x तेज़
- CPU: L1 cache — main memory से 10x तेज़

कम्पाइलर बार-बार एक्सेस होने वाले डेटा को लोकल बफ़र में ले जाता है — ठीक वैसे जैसे ज़रूरी फ़ाइलें नेटवर्क ड्राइव के बजाय डेस्कटॉप पर रखना।

**Pattern**: `pm_add_buffers_local + rangeify_codegen`

| ट्रांसफ़ॉर्म | उद्देश्य |
|-------------|----------|
| `bufferize_to_store` | `allow_locals=true` वाले BUFFERIZE कन्वर्ट करें |
| CONTIGUOUS wrapper हटाएँ | Codegen से पहले ऑप्टिमाइज़ेशन hints हटाएँ |
| NOOP हटाना | No-op ऑपरेशन साफ़ करें |

**Morok**: `rangeify/patterns.rs`, `rangeify/transforms.rs`, `optimizer/mod.rs`
