---
sidebar_label: ऑप्टिमाइज़ेशन सिस्टम
---

# पैटर्न-आधारित ऑप्टिमाइज़ेशन

कोई भी प्रोडक्शन ML कम्पाइलर खोलिए और आपको दर्जनों ऑप्टिमाइज़ेशन पासेज़ मिलेंगे: constant folding, dead code elimination, operator fusion, loop tiling, vectorization, memory layout optimization। हर पास के अपने डेटा स्ट्रक्चर, अपना traversal लॉजिक, अपने बग्स।

Morok एक अलग तरीका अपनाता है: **हर चीज़ के लिए एक मैकेनिज़्म**।

```text
Traditional Compiler:              Morok:
┌─────────────────────────┐       ┌─────────────────────────┐
│  Constant Folding       │       │                         │
│  Dead Code Elimination  │       │   patterns! {           │
│  Loop Unrolling         │       │       Add[x, @zero] ~> x│
│  Operator Fusion        │       │       Mul[x, @zero] ~> 0│
│  Vectorization          │       │       // ...more        │
│  Memory Planning        │       │   }                     │
│  ...20 more passes      │       │                         │
└─────────────────────────┘       │   graph_rewrite(...)    │
     Custom logic each            └─────────────────────────┘
                                       One mechanism
```

Morok में हर ऑप्टिमाइज़ेशन एक **पैटर्न** के रूप में एक्सप्रेस होता है: "जब यह स्ट्रक्चर दिखे, उसे इस स्ट्रक्चर से बदल दो।" वही `graph_rewrite()` फ़ंक्शन constant folding अप्लाई करता है, movement ops को लूप्स में बदलता है, मेमोरी एक्सेस पैटर्न ऑप्टिमाइज़ करता है, और हार्डवेयर प्रिमिटिव्स में लोअर करता है।

यह चैप्टर बताता है कि पैटर्न-आधारित ऑप्टिमाइज़ेशन कैसे काम करता है और यह पावरफ़ुल क्यों है।

---

## `patterns!` DSL

Morok ऑप्टिमाइज़ेशन पैटर्न लिखने के लिए एक domain-specific language प्रदान करता है। यह कुछ ऐसा दिखता है:

```rust
patterns! {
    // Identity folding: x + 0 → x
    Add[x, @zero] ~> |x| x.clone(),

    // Constant folding: 3 + 4 → 7
    Add(a @const(a_val), b @const(b_val))
        => |a, a_val, b_val| eval_add(a_val, b_val).map(|r| UOp::const_(a.dtype(), r)),

    // Self-folding: x / x → 1
    Idiv(x, x) ~> |x| UOp::one(x.dtype()),

    // Dead code elimination: if(true) { t } else { f } → t
    Where(@true, t, _f) ~> |t| t.clone(),
}
```

मैक्रो इन पैटर्न को एफ़िशिएंट Rust कोड में कम्पाइल करता है। चलिए सिंटैक्स तोड़कर देखते हैं:

| सिंटैक्स | मतलब | उदाहरण |
|----------|------|--------|
| `(x, y)` | **Ordered।** एक्ज़ैक्ट ऑर्डर में मैच। | `Sub(x, @zero) ~> x` |
| `[x, y]` | **Commutative।** दोनों orderings ट्राई करे। | `Add[x, @zero] ~> x` |
| `@zero` | **ज़ीरो कॉन्स्टेंट।** 0 या 0.0 मैच करे। | `Mul[_, z @ @zero] ~> z` |
| `@one` | **वन कॉन्स्टेंट।** 1 या 1.0 मैच करे। | `Mul[x, @one] ~> x` |
| `@const(val)` | **कॉन्स्टेंट एक्सट्रैक्ट।** वैल्यू बाइंड करे। | `Add(@const(a), @const(b))` |
| `x, x` | **Same operand।** ptr_eq चेक ऑटो-जनरेट। | `Idiv(x, x) ~> UOp::one(...)` |
| `~>` | **Infallible।** हमेशा सफ़ल, `Arc<UOp>` रिटर्न। | `Add[x, @zero] ~> x` |
| `=>` | **Fallible।** फ़ेल हो सकता है, `Option<Arc<UOp>>` रिटर्न। | `=> eval(...).map(...)` |
| `for op in binary [...]` | **Template।** मल्टीपल ops के लिए पैटर्न जनरेट। | नीचे देखें |
| `@context Type` | **Stateful।** पैटर्न में mutable context एक्सेस। | नीचे देखें |

### Template एक्सपैंशन

हर binary ऑपरेशन के लिए एक ही पैटर्न लिखने की बजाय, for-loop इस्तेमाल करें:

```rust
patterns! {
    for op in binary [Add, Mul, Sub, Idiv, Fdiv, Max] {
        op(a @const(a_val), b @const(b_val))
            => |a, a_val, b_val| eval_binary(op, a_val, b_val)
                .map(|r| UOp::const_(a.dtype(), r))
    }
}
```

यह कम्पाइल टाइम पर छह अलग-अलग पैटर्न में एक्सपैंड होता है — हर ऑपरेशन के लिए एक।

### Stateful पैटर्न

कुछ ऑप्टिमाइज़ेशन को context चाहिए (जैसे, हम किस कर्नेल में हैं, कौन सी ranges एक्टिव हैं)। एक context type डिक्लेयर करें:

```rust
patterns! {
    @context KernelContext;

    ReduceAxis { src } => |reduce, src, ctx| {
        ctx.record_reduction(reduce);
        transform_reduce(reduce, src, ctx)
    }
}
```

Context पैटर्न closures को लास्ट आर्ग्युमेंट के रूप में पास होता है।

---

## पैटर्न मैचिंग कैसे काम करता है

`patterns!` मैक्रो एक `SimplifiedPatternMatcher` जनरेट करता है जो **O(1)** टाइम में पैटर्न डिस्पैच करता है।

### OpKey इंडेक्स

हर UOp का एक ऑपरेशन टाइप होता है (Add, Mul, Load, वगैरह)। `#[derive(PatternEnum)]` मैक्रो एक `OpKey` enum जनरेट करता है जो ऑपरेशन को hashable keys में मैप करता है:

```rust
pub enum OpKey {
    Binary(BinaryOp),    // Add, Mul, Sub, ...
    Unary(UnaryOp),      // Neg, Sqrt, Exp, ...
    Ternary(TernaryOp),  // Where, MulAcc
    Const,
    Load,
    Store,
    // ... one variant per operation category
}
```

### Matcher स्ट्रक्चर

```rust
pub struct SimplifiedPatternMatcher<C = ()> {
    indexed: HashMap<OpKey, Vec<PatternClosure<C>>>,  // O(1) lookup
    wildcards: Vec<PatternClosure<C>>,                 // patterns matching any op
}
```

UOp मैच करते समय:

1. UOp के ऑपरेशन से **OpKey एक्सट्रैक्ट** करें
2. HashMap में **लुकअप** करें — O(1)
3. हर closure **ट्राई** करें जब तक कोई मैच न हो
4. कोई indexed पैटर्न मैच न हो तो **wildcards पर फ़ॉलबैक**

यह सभी पैटर्न को linearly स्कैन करने से 5-10x तेज़ है।

### Commutative हैंडलिंग

`Add[x, @zero]` जैसे पैटर्न के लिए, मैक्रो ऐसा कोड जनरेट करता है जो दोनों orderings ट्राई करे:

```rust
// Try (x, @zero)
if let Some(result) = try_match_ordered(&children[0], &children[1]) {
    return result;
}
// Try (@zero, x)
if let Some(result) = try_match_ordered(&children[1], &children[0]) {
    return result;
}
```

### डुप्लिकेट डिटेक्शन

जब आप `Idiv(x, x)` लिखते हैं, पैटर्न सिर्फ़ तभी मैच होना चाहिए जब दोनों operands *एक ही* UOp हों (pointer equality, structural equality नहीं)। मैक्रो ऑटोमैटिकली यह चेक जनरेट करता है:

```rust
// Generated code for Idiv(x, x)
let x = &children[0];
let x_dup = &children[1];
if !Arc::ptr_eq(x, x_dup) {
    return NoMatch;
}
// ... rest of pattern
```

यह hash consing का फ़ायदा उठाता है — आइडेंटिकल subexpressions एक ही पॉइंटर शेयर करते हैं।

---

## रीराइट इंजन: टू-स्टेज एल्गोरिदम

> **नोट:** यह सिम्प्लिफ़ाइड प्रेज़ेंटेशन है। असल इंजन एफ़िशिएंसी के लिए path compression के साथ 3-स्टेज stack-based एल्गोरिदम इस्तेमाल करता है।

सिर्फ़ पैटर्न मैचिंग काफ़ी नहीं है। यह एक्सप्रेशन देखें:

```text
WHERE(Lt(3, 5), t, f)
```

इसे सिम्प्लिफ़ाई करने के लिए, दो स्टेप चाहिए:
1. `Lt(3, 5)` → `true` (constant folding)
2. `WHERE(true, t, f)` → `t` (dead code elimination)

लेकिन `WHERE` पैटर्न तब तक मैच नहीं करेगा जब तक उसका child सिम्प्लिफ़ाई न हो। रीराइट इंजन इसे **टू-स्टेज एल्गोरिदम** से सॉल्व करता है।

### स्टेज 0: पैटर्न अप्लिकेशन

```rust
fn rewrite_stage0(&mut self, uop: &Arc<UOp>) -> RewriteResult {
    match self.matcher.try_match(uop) {
        Some(replacement) => RewriteResult::Rewritten(replacement),
        None => RewriteResult::Gate(uop.clone()),  // process children
    }
}
```

कोई पैटर्न मैच न हो तो `Gate` रिटर्न करें — children को पहले प्रोसेस करने का सिग्नल।

### स्टेज 1: सोर्स रीकंस्ट्रक्शन

Children रीराइट होने के बाद, नए children के साथ नोड रीबिल्ड करें और पैटर्न फिर ट्राई करें:

```rust
fn rewrite_stage1(&mut self, uop: &Arc<UOp>, new_children: Vec<Arc<UOp>>) {
    // Rebuild with optimized children
    let rebuilt = uop.with_sources(new_children);

    // Try patterns again—might match now!
    match self.matcher.try_match(&rebuilt) {
        Some(replacement) => replacement,
        None => rebuilt,
    }
}
```

### जादू: कैस्केडिंग ऑप्टिमाइज़ेशन

```text
Stage 0: WHERE(Lt(3, 5), t, f)     → Gate (no match, process children)
         └── Lt(3, 5)              → true (constant folding matches!)

Stage 1: WHERE(true, t, f)         → t (dead code elimination matches!)
```

रीकंस्ट्रक्शन स्टेज पैटर्न री-अप्लाई करता है, जो सिंगल traversal में मल्टी-स्टेप ऑप्टिमाइज़ेशन सक्षम करता है।

### सेफ़्टी लिमिट्स

इन्फ़िनिट लूप रोकने के लिए, इंजन में लिमिट्स हैं:
- प्रति नोड मैक्सिमम **1000 iterations**
- कुल मैक्सिमम **500,000 iterations**
- लिमिट पार होने पर डायग्नोस्टिक इन्फ़ो के साथ panic

प्रैक्टिस में, ठीक से बने पैटर्न जल्दी converge करते हैं।

---

## पूरी ऑप्टिमाइज़ेशन पाइपलाइन

पैटर्न मैचिंग एक बड़ी पाइपलाइन का एक हिस्सा है। जब आप `tensor.realize()` कॉल करते हैं, यह होता है:

```text
Tensor.realize()
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  RANGEIFY                                               │
│  Convert movement ops (RESHAPE, PERMUTE, EXPAND)        │
│  into explicit RANGE loops with INDEX operations        │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  KERNEL SPLITTING                                       │
│  Split computation graph at STORE boundaries            │
│  Each STORE becomes a separate kernel                   │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  FOR EACH KERNEL:                                       │
│                                                         │
│  1. Symbolic Simplification (algebraic patterns)        │
│                                                         │
│  2. Scheduler Creation                                  │
│     └── Convert LOOP → GLOBAL for GPU parallelization   │
│                                                         │
│  3. Kernel Optimization (heuristic OR beam search)      │
│     ├── Tensor Cores (WMMA) for matmul                  │
│     ├── Vectorization (UPCAST)                          │
│     ├── Loop Unrolling (UNROLL)                         │
│     ├── GPU Local Memory (LOCAL)                        │
│     ├── Grouped Reductions (GROUP)                      │
│     └── Threading (THREAD) for CPU                      │
│                                                         │
│  4. Post-Optimization Passes                            │
│     ├── Devectorize (memory coalescing)                 │
│     ├── Expand (UNROLL → vector operations)             │
│     ├── FMA Decomposition (a*b+c → MulAcc)              │
│     └── Bool Storage (cast bool↔uint8 for memory)       │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  CODE GENERATION                                        │
│  Render optimized AST to LLVM IR, compile, execute      │
└─────────────────────────────────────────────────────────┘
```

हर बॉक्स पैटर्न-आधारित रीराइटिंग इस्तेमाल करता है। फ़र्क सिर्फ़ इतना है कि कौन से पैटर्न अप्लाई होते हैं:

- **Rangeify**: Movement op → BUFFERIZE + INDEX पैटर्न
- **Symbolic**: एल्जेब्रिक सिम्प्लिफ़िकेशन पैटर्न
- **Post-opt**: मेमोरी एक्सेस ऑप्टिमाइज़ेशन पैटर्न

---

## कर्नेल ऑप्टिमाइज़ेशन: Heuristics बनाम Beam Search

Symbolic सिम्प्लिफ़िकेशन के बाद, हर कर्नेल को *scheduling decisions* चाहिए: लूप कैसे tile करें, कहाँ पैरेललाइज़ करें, tensor cores इस्तेमाल करें या नहीं। Morok दो स्ट्रैटेजी ऑफ़र करता है।

### Heuristics (डिफ़ॉल्ट)

Heuristic ऑप्टिमाइज़र एक फ़िक्स्ड ऑर्डर में ऑप्टिमाइज़ेशन अप्लाई करता है:

```rust
pub fn hand_coded_optimizations(scheduler: &mut Scheduler) {
    // 1. Tensor cores (if matmul pattern detected)
    if let Some(tc) = detect_tensor_core_pattern(scheduler) {
        apply_tensor_core(scheduler, tc);
        return;  // TC handles everything
    }

    // 2. Grouped reductions (two-stage for large reductions)
    apply_grouped_reduction_if_needed(scheduler);

    // 3. Vectorization (UPCAST output dimensions)
    apply_upcast(scheduler, 4);

    // 4. GPU local memory (workgroup dimensions)
    apply_local_dims(scheduler);

    // 5. CPU threading
    apply_threading(scheduler);
}
```

**फ़ायदे**: तेज़ (~50ms प्रति कर्नेल), predictable, हार्डवेयर measurement नहीं चाहिए।

**नुकसान**: ऑप्टिमाइज़ेशन के मौके छूट सकते हैं, फ़िक्स्ड heuristics वर्कलोड के हिसाब से adapt नहीं करते।

### Beam Search (ऑप्शनल)

प्रोडक्शन वर्कलोड के लिए, beam search बेहतर schedules ढूँढता है:

```rust
pub fn beam_search(scheduler: Scheduler, config: BeamConfig) -> Scheduler {
    let mut beam = vec![scheduler];
    let deadline = Instant::now() + config.time_limit;

    while Instant::now() < deadline {
        let mut candidates = vec![];

        for state in &beam {
            // Generate all valid next actions
            for action in generate_actions(state) {
                if let Ok(next) = state.apply(action) {
                    candidates.push(next);
                }
            }
        }

        // Compile and time each candidate
        let timed: Vec<_> = candidates.par_iter()
            .map(|c| (c, measure_kernel_time(c)))
            .collect();

        // Keep top K by execution time
        beam = timed.into_iter()
            .sorted_by_key(|(_, time)| *time)
            .take(config.beam_width)
            .map(|(c, _)| c)
            .collect();
    }

    beam.into_iter().next().unwrap()
}
```

एक्शन स्पेस में ~162 बेस एक्शन शामिल हैं (उपलब्ध parallelism के अनुसार अलग-अलग):
- `UPCAST(axis, amount)` — आउटपुट डायमेंशन vectorize करें
- `UNROLL(axis, amount)` — रिडक्शन लूप अनरोल करें
- `LOCAL(axis, amount)` — GPU shared memory इस्तेमाल करें
- `GROUP(axis, amount)` — टू-स्टेज रिडक्शन
- `GROUPTOP(axis, amount)` — tensor cores के लिए grouped reduction
- `THREAD(axis, amount)` — CPU पैरेललाइज़ेशन
- `SWAP(axis1, axis2)` — global dimensions रीऑर्डर करें

**फ़ायदे**: नियर-ऑप्टिमल schedules ढूँढता है, हार्डवेयर के हिसाब से adapt करता है।

**नुकसान**: प्रति कर्नेल मिनटों में (लेकिन रिज़ल्ट AST hash से कैश होते हैं)।

### कॉन्फ़िगरेशन

```bash
# Disable optimization (debugging)
MOROK_NOOPT=1 cargo run

# Enable beam search with width 8
MOROK_BEAM=8 cargo run
```

या प्रोग्रामैटिकली:

```rust
let config = PrepareConfig::builder()
    .strategy(OptStrategy::Beam { width: 8 })
    .build();

tensor.realize_with(config)?;
```

---

## तुलना: दूसरे कम्पाइलर कैसे ऑप्टिमाइज़ करते हैं

अलग-अलग ML कम्पाइलर ऑप्टिमाइज़ेशन के लिए अलग-अलग तरीके अपनाते हैं:

| पहलू | XLA | TVM/Ansor | Triton | **Morok** |
|-------|-----|-----------|--------|-----------|
| **फ़िलॉसफ़ी** | फ़िक्स्ड heuristics | सर्च-आधारित | प्रोग्रामर-गाइडेड | पैटर्न-आधारित |
| **Fusion** | कंज़र्वेटिव नियम | Tile-and-fuse | Block-level | ग्राफ़ रीराइटिंग |
| **Auto-tuning** | कोई नहीं | Evolutionary + cost model | Grid search | Beam search |
| **ट्यूनिंग कॉस्ट** | 0 | घंटों | मिनटों | मिनटों (कैश्ड) |
| **फ़्लेक्सिबिलिटी** | कम | ज़्यादा | मध्यम | ज़्यादा |
| **ट्रांसपैरेंसी** | कम (C++ पासेज़) | मध्यम (Python) | मध्यम (DSL) | ज़्यादा (patterns!) |

### XLA — प्रोडक्शन कंज़र्वेटिव

XLA fusion decisions के लिए फ़िक्स्ड heuristics इस्तेमाल करता है। सेफ़ और predictable, लेकिन परफ़ॉर्मेंस टेबल पर छूट जाती है। Fusion rules C++ में hard-coded हैं — उन्हें extend करने के लिए डीप कम्पाइलर नॉलेज चाहिए।

### TVM/Ansor — मैक्सिमम Auto-Tuning

TVM *क्या* कम्प्यूट करना है और *कैसे* कम्प्यूट करना है को अलग करता है। Ansor schedule space explore करने के लिए learned cost model के साथ evolutionary search इस्तेमाल करता है। बेस्ट-इन-क्लास परफ़ॉर्मेंस दे सकता है, लेकिन ट्यूनिंग में प्रति मॉडल घंटे लगते हैं।

### Triton — प्रोग्रामर-गाइडेड

Triton एक Python-जैसा DSL एक्सपोज़ करता है जहाँ आप blocked algorithms एक्सप्लिसिटली लिखते हैं। कम्पाइलर register allocation और memory management हैंडल करता है। कंट्रोल और ऑटोमेशन का अच्छा बैलेंस, लेकिन GPU प्रोग्रामिंग एक्सपर्टीज़ चाहिए।

### Morok — पैटर्न कम्पोज़िशन

Morok की इनसाइट: ऑप्टिमाइज़ेशन को composable पैटर्न के रूप में एक्सप्रेस करें। हर पैटर्न लोकल और verifiable है। कॉम्प्लेक्स ऑप्टिमाइज़ेशन composition से निकलते हैं। Beam search ज़रूरत पड़ने पर auto-tuning जोड़ता है, रिज़ल्ट रीयूज़ के लिए कैश होते हैं।

---

## यह क्यों ज़रूरी है: प्रैक्टिकल फ़ायदे

पैटर्न-आधारित ऑप्टिमाइज़ेशन के डेवलपर्स के लिए ठोस फ़ायदे हैं:

**डीबगिंग डायरेक्ट है।** पैटर्न पढ़ने योग्य कोड हैं। किसी भी पैटर्न में `println!` जोड़ें ताकि पता चले कब फ़ायर होता है:

```rust
patterns! {
    Add[x, @zero] ~> |x| {
        println!("Folding add-zero: {:?}", x);
        x.clone()
    }
}
```

**एक्सटेंसिबिलिटी आसान है।** कस्टम ऑप्टिमाइज़ेशन जोड़ना दो लाइन है:

```rust
patterns! {
    // Your domain-specific optimization
    MyOp(x, y) if is_special_case(x, y) ~> transform(x, y)
}
```

कम्पाइलर इंटरनल्स समझने, visitors लिखने, या pass managers मॉडिफ़ाई करने की ज़रूरत नहीं।

**करेक्टनेस लोकल है।** हर पैटर्न एक छोटा theorem है: "अगर यह स्ट्रक्चर दिखे, उसे इससे बदलने से सिमैंटिक्स preserve होते हैं।" हर पैटर्न को इंडिपेंडेंटली वेरिफ़ाई करें। करेक्ट पैटर्न का composition करेक्ट प्रोग्राम देता है।

**परफ़ॉर्मेंस ट्यूनेबल है।** O(1) पैटर्न dispatch डिफ़ॉल्ट से तेज़ है। प्रोडक्शन वर्कलोड के लिए beam search इनेबल करें। AST hash से रिज़ल्ट कैश करें — एक बार ट्यून करें, हमेशा फ़ायदा।

---

## गहरी समझ

पैटर्न मैचिंग generality को composability से ट्रेड करता है।

एक general-purpose ऑप्टिमाइज़ेशन पास कुछ भी कर सकता है — लेकिन यही प्रॉब्लम है। इसे वेरिफ़ाई करना मुश्किल, extend करना मुश्किल, दूसरे passes के साथ compose करना मुश्किल। ऑर्डरिंग मैटर करती है। इंटरैक्शन सटल हैं।

पैटर्न constrained है: यह एक स्पेसिफ़िक स्ट्रक्चर मैच करता है और स्पेसिफ़िक replacement प्रोड्यूस करता है। लेकिन constraints composition सक्षम करते हैं। पैटर्न किसी भी ऑर्डर में चलाएँ — रिज़ल्ट एक ही fixed point पर converge करता है। नए पैटर्न जोड़ें मौजूदा को तोड़े बिना। पैटर्न हटाएँ बिना cascading failures के।

हर पैटर्न semantic equivalence का एक theorem है। रीराइट इंजन एक theorem prover है, जो इनपुट से ऑप्टिमाइज़्ड आउटपुट तक derivations ढूँढता है। करेक्टनेस individual steps की करेक्टनेस से आती है।

यह Unix philosophy है जो कम्पाइलरों पर लागू है: छोटे, focused टूल जो compose करें। पैटर्न-आधारित ऑप्टिमाइज़ेशन हर प्रॉब्लम सॉल्व नहीं करेगा — लेकिन जो प्रॉब्लम सॉल्व करता है, उन्हें एलिगेंटली सॉल्व करता है।
