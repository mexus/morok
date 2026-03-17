---
sidebar_label: Phase 4 — Linearizer
---

# Phase 4: Linearizer

**गोल**: DAG को लीनियर इंस्ट्रक्शन सीक्वेंस में बदलें।

---

## Stage 16: Post-Index Symbolic

> **स्टेज एक नज़र में**
>
> **गोल**: Index lowering के बाद पूर्ण symbolic सिम्प्लीफ़िकेशन
> **मुख्य Patterns**: सभी symbolic नियम (140+)
> **प्रभाव**: सीरियलाइज़ेशन से पहले फ़ाइनल क्लीनअप

**यह क्या करता है**: Index lowering के बाद पूर्ण symbolic सिम्प्लीफ़िकेशन।

**यह क्यों ज़रूरी है**: अब indices कॉन्क्रीट integers (i32/i64) हैं, अरिथमेटिक पूरी तरह सिम्प्लीफ़ाई हो सकता है। लीनियराइज़ेशन से पहले एक्सप्रेशन साफ़ करने का यह आखिरी मौका है।

**Pattern**: `symbolic`

इसमें GEP pushing patterns शामिल हैं — अरिथमेटिक से address कैलकुलेशन गुज़ारना:
```text
Before:  GEP(ADD(arr_a, arr_b), idx)
              ↓ [Push GEP through ADD]
After:   ADD(GEP(arr_a, idx), GEP(arr_b, idx))
```
*क्यों?* GEPs का पैरेलल कम्प्यूटेशन सक्षम करता है और downstream वेक्टराइज़ेशन सक्षम कर सकता है। (नोट: Pattern तभी अप्लाई होता है जब GEP का dtype और ALU का dtype पॉइंटर न हों।)

---

## Stage 17: Pre-Matcher (ऑप्शनल)

> **स्टेज एक नज़र में**
>
> **गोल**: Decomposition से पहले बैकएंड-स्पेसिफ़िक patterns
> **मुख्य Patterns**: Renderer-स्पेसिफ़िक
> **प्रभाव**: हार्डवेयर-स्पेसिफ़िक ऑप्टिमाइज़ेशन

**यह क्या करता है**: Decomposition से पहले renderer-स्पेसिफ़िक patterns अप्लाई करता है।

**यह क्यों ज़रूरी है**: हर बैकएंड अपने patterns जोड़ सकता है। उदाहरण के लिए, DSP बैकएंड इसे जेनेरिक patterns को DSP-स्पेसिफ़िक SIMD intrinsics से बदलने के लिए इस्तेमाल करता है। इससे जेनेरिक पाइपलाइन बदले बिना हार्डवेयर-स्पेसिफ़िक ऑप्टिमाइज़ेशन मिलते हैं।

**Pattern**: `renderer.pre_matcher`

ज़्यादातर बैकएंड (CPU, GPU) को इसकी ज़रूरत नहीं। सिर्फ़ स्पेशलाइज़्ड हार्डवेयर इस्तेमाल करता है।

**नोट**: Morok अभी इस स्टेज को इम्प्लीमेंट नहीं करता। `Renderer` trait में `render()`, `backend_name()`, और `decompositor()` मेथड हैं, लेकिन `pre_matcher` सपोर्ट अभी नहीं है। यह DSP और दूसरे स्पेशलाइज़्ड बैकएंड के लिए फ़्यूचर एनहैंसमेंट है।

---

## Stage 18: Decompositions

> **स्टेज एक नज़र में**
>
> **गोल**: जो ऑपरेशन टारगेट सपोर्ट नहीं करता उन्हें रीराइट करें
> **मुख्य Patterns**: Power-of-2, transcendental approximations
> **प्रभाव**: हाई-लेवल ops को हार्डवेयर इंस्ट्रक्शन से मैप करता है

**यह क्या करता है**: जो ऑपरेशन टारगेट सपोर्ट नहीं करता, उनके लिए late rewrites।

**यह क्यों ज़रूरी है**: हार्डवेयर में हर ऑपरेशन नहीं होता। उदाहरण के लिए, ज़्यादातर CPUs में डायरेक्ट `sin` इंस्ट्रक्शन नहीं है। हम इसे उन ऑपरेशनों से approximate करते हैं जो मौजूद हैं (addition, multiplication, आदि)।

**Pattern**: `symbolic_simple() + get_late_rewrite_patterns()`

नोट: `pm_render()` और `pm_split_ends()` इस combined पास का हिस्सा नहीं हैं — वे Stage 19 में अलग से चलते हैं।

| Pattern | उदाहरण | कब इस्तेमाल |
|---------|--------|-------------|
| `MOD → AND` | `x % 8 → x & 7` | Power-of-2 divisor |
| `MUL → SHL` | `x * 16 → x << 4` | Power-of-2 multiplier |
| `DIV → SHR` | `x // 8 → x >> 3` | Power-of-2 divisor |
| `FDIV → MUL` | `x / 2.0 → x * 0.5` | Float constant divisor |
| `NEG` | `x * -1 → NEG(x)` | जब NEG सपोर्टेड हो |
| `MULACC` | `a * b + c → MULACC(a, b, c)` | जब FMA सपोर्टेड हो |
| Fast integer division | `x // 7 → (x * M) >> S` | Non-power-of-2 divisor |
| De Morgan's laws | `(!x) & (!y) → !(x \| y)` | Boolean सिम्प्लीफ़िकेशन (दोनों दिशाएँ) |
| Comparison negations | `!(x < c) → (c-1) < x` | Integer comparisons |

Transcendental function approximations (SIN, EXP, LOG, आदि) `decompositor()` pathway से इम्प्लीमेंट हैं (`ir/src/decompositions/transcendentals.rs` देखें)।

**Morok**: `optimizer/mod.rs`

---

## Stage 19: Final Rewrite

> **स्टेज एक नज़र में**
>
> **गोल**: लीनियराइज़ेशन की तैयारी
> **मुख्य Patterns**: CONST वेक्टराइज़ेशन, GEP resolution, END splitting
> **प्रभाव**: लीनियराइज़ेशन के लिए साफ़ representation

**यह क्या करता है**: लीनियराइज़ेशन की तैयारी।

**यह क्यों ज़रूरी है**: कुछ patterns decomposition के बाद अप्लाई करना आसान होता है। यह स्टेज लीनियर सीक्वेंस में कन्वर्ट करने से पहले फ़ाइनल क्लीनअप करता है।

**Pattern**: `symbolic_simple() + get_late_rewrite_patterns() + pm_render()`

नोट: `extra_matcher` और `pm_split_ends` अलग से चलते हैं, इस combined पास का हिस्सा नहीं हैं।

**CONST वेक्टराइज़ेशन**:
```text
// Make vector constants explicit
CONST(1.0) used as vec4 → VECTORIZE(1.0, 1.0, 1.0, 1.0)
```

**CAT to VECTORIZE** (`pm_render` से):
```text
CAT(a, b, c, d) → VECTORIZE(a, b, c, d)
```
CAT डायरेक्ट render नहीं हो सकता; codegen के लिए एक्सप्लिसिट VECTORIZE ज़रूरी है।

**GEP resolution**: बचे हुए GEP ऑपरेशन कन्वर्ट करें।

**मल्टी-range ENDs स्प्लिट करें**:
```text
// Before: END closing multiple ranges
END(op, [range_a, range_b])

// After: nested single ENDs
END(END(op, range_a), range_b)
```

**extra_matcher**: हर बैकएंड अपने फ़ाइनल patterns जोड़ सकता है। इससे जेनेरिक पाइपलाइन बदले बिना हार्डवेयर-स्पेसिफ़िक ऑप्टिमाइज़ेशन मिलते हैं।

**Morok**: `devectorize.rs`, `linearize/mod.rs`, `optimizer/mod.rs`

---

## Stage 20: Add Control Flow

> **स्टेज एक नज़र में**
>
> **गोल**: कंट्रोल फ़्लो ग्राफ़ बनाएँ और range डिपेंडेंसी जोड़ें
> **मुख्य कॉन्सेप्ट**: तीन रिलेशनशिप टाइप (nested, dependent, independent)
> **प्रभाव**: सही इंस्ट्रक्शन ऑर्डरिंग

**यह क्या करता है**: कंट्रोल फ़्लो ग्राफ़ बनाता है और range डिपेंडेंसी जोड़ता है।

**यह क्यों ज़रूरी है**: ऑपरेशन सही ऑर्डर में एक्ज़ीक्यूट होने चाहिए। अगर load कोई RANGE की वैल्यू इस्तेमाल करता है, तो RANGE पहले आना चाहिए। यह स्टेज इन डिपेंडेंसीज़ को ट्रैक और एनफ़ोर्स करता है।

**Pattern**: `pm_add_control_flow` (bottom-up)

```text
// Analyze which END operations depend on which
END(computation, [RANGE_A]) and END(other_computation, [RANGE_B]) are siblings
→ Creates edge: RANGE_B.src += END(computation)

// Add explicit dependency
RANGE_B waits for RANGE_A to complete
```

**तीन रिलेशनशिप टाइप**:

| रिलेशनशिप | उदाहरण | मतलब |
|------------|--------|------|
| Nested | RANGE_A RANGE_B के अंदर | A को B शुरू होने से पहले पूरा होना चाहिए |
| Dependent | END_A और END_B siblings हैं | END_B को END_A का इंतज़ार करना चाहिए (sibling डिपेंडेंसी) |
| Independent | RANGE_X और RANGE_Y इंटरैक्ट नहीं करते | पैरेलल चल सकते हैं |

Bottom-up ट्रैवर्सल सुनिश्चित करता है कि डिपेंडेंसी leaves से roots तक सही बहे।

**Morok**: `schedule/src/linearize/mod.rs`

---

## Stage 21: Linearize

> **स्टेज एक नज़र में**
>
> **गोल**: DAG को लीनियर इंस्ट्रक्शन सीक्वेंस में बदलें
> **मुख्य अल्गोरिदम**: Priority-aware topological sort
> **प्रभाव**: वैलिड एक्ज़ीक्यूशन ऑर्डर

**यह क्या करता है**: DAG को priority-aware topological sort से लीनियर इंस्ट्रक्शन सीक्वेंस में बदलता है।

**यह क्यों ज़रूरी है**: ग्राफ़ स्ट्रक्चर एक्ज़ीक्यूशन ऑर्डर specify नहीं करता। हमें डिपेंडेंसी respect करते हुए फ़्लैटन करना होगा। Priorities सेंसिबल ऑर्डरिंग सुनिश्चित करती हैं (definitions uses से पहले, loads कम्प्यूटेशन से पहले, stores बाद में)।

**फ़ंक्शन**: `linearize(sink)`

| ऑपरेशन | प्रायोरिटी | क्यों |
|---------|-----------|------|
| DEFINE_GLOBAL | -20 | आर्ग्युमेंट पहले डिफ़ाइन होने चाहिए |
| DEFINE_VAR | -19 | वेरिएबल पहले डिफ़ाइन होने चाहिए |
| DEFINE_LOCAL | -18 | एलोकेशन पहले |
| DEFINE_REG | -17 | रजिस्टर पहले |
| CONST | -10 | Reuse के लिए constants जल्दी (Morok एक्सटेंशन; Tinygrad डिफ़ॉल्ट 0) |
| LOAD | -1 | इस्तेमाल से पहले Loads |
| END | -5 | Ranges बंद करता है |
| STORE | +1 | कम्प्यूटेशन के बाद Stores |
| RANGE | +5 | इस्तेमाल से पहले Ranges खुलें |

कम प्रायोरिटी = सीक्वेंस में पहले। इससे सुनिश्चित होता है:
- Definitions पहले आएँ
- Loads कम्प्यूटेशन से पहले हों
- Stores आखिर में हों
- Ranges अपने contents से पहले खुलें, बाद में बंद हों

**Run_count ऑर्डरिंग**: ऑपरेशन मुख्य रूप से एक्ज़ीक्यूशन फ़्रीक्वेंसी (run_count) से सॉर्ट होते हैं, फिर प्रायोरिटी से। कम एक्ज़ीक्यूशन फ़्रीक्वेंसी वाले ऑपरेशन (inner loops के बाहर) पहले शेड्यूल होते हैं, जबकि inner loops वाले (ज़्यादा run_count) बाद में। उदाहरण: 100 बार एक्ज़ीक्यूट होने वाला CONST, 1M बार वाले से पहले आता है।

**run_count कैलकुलेशन**:
```text
run_count = prod(int(r.vmax) + 1 for r in u.ranges)
```
यह कैलकुलेट करता है कि enclosing ranges के आधार पर ऑपरेशन कितनी बार एक्ज़ीक्यूट होता है।

**Morok**: `schedule/src/linearize/mod.rs`

---

## Stage 22: Cleanup IF/ENDIF

> **स्टेज एक नज़र में**
>
> **गोल**: लीनियर इंस्ट्रक्शन लिस्ट का फ़ाइनल क्लीनअप
> **मुख्य ट्रांसफ़ॉर्मेशन**: Gated INDEX → IF/STORE/ENDIF
> **प्रभाव**: बिना predicated stores वाले हार्डवेयर को हैंडल करता है

**यह क्या करता है**: लीनियर इंस्ट्रक्शन लिस्ट का फ़ाइनल क्लीनअप।

**यह क्यों ज़रूरी है**: कुछ हार्डवेयर (मॉडर्न GPUs) "predicated stores" सपोर्ट करता है — मेमोरी में तभी लिखो जब condition true हो। पुराना हार्डवेयर नहीं करता। उनके लिए, store को IF स्टेटमेंट में रैप करना पड़ता है। यह स्टेज सिर्फ़ तभी चलता है जब हार्डवेयर में predicated store सपोर्ट न हो।

**Pattern**: `pm_linearize_cleanups` (`line_rewrite` से, `graph_rewrite` नहीं)

```text
// Gated INDEX in STORE becomes conditional store
STORE(INDEX(ptr, idx, valid=cond), value)
→ IF(cond) { STORE(INDEX(ptr, idx), value) } ENDIF
```

**नोट**: यह स्टेज `graph_rewrite` के बजाय `line_rewrite` इस्तेमाल करता है क्योंकि यह DAG के बजाय पहले से लीनियराइज़्ड इंस्ट्रक्शन लिस्ट पर ऑपरेट करता है।

इस पॉइंट पर, इंस्ट्रक्शन लिस्ट कोड जनरेशन के लिए तैयार है।

**Morok**: `schedule/src/linearize/mod.rs` (predicated stores path)
