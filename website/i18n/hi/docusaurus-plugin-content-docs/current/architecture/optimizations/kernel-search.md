---
sidebar_label: कर्नेल सर्च
---

# कर्नेल ऑप्टिमाइज़ेशन सर्च

Algebraic simplification के बाद, हर कर्नेल को *scheduling decisions* चाहिए: loops कैसे tile करें, कहाँ parallelize करें, tensor cores इस्तेमाल करें या नहीं। Morok दो strategies offer करता है: fast heuristics और thorough beam search।

यह [codegen pipeline](../codegen/overview.md) के Stage 7 में चलता है।

Tinygrad source: `tinygrad/codegen/opt/`। Morok source: `schedule/src/optimizer/`।

---

## Action Space

ऑप्टिमाइज़ेशन loop structures को axis types बदलकर transform करता है। हर action एक range modify करता है:

| Action | Effect | Hardware Target |
|--------|--------|-----------------|
| UPCAST(axis, amount) | Dimension vectorize करे (SIMD) | सभी |
| UNROLL(axis, amount) | Loop dimension unroll करे | सभी |
| LOCAL(axis, amount) | GPU shared memory इस्तेमाल करे | GPU (LDS) / CPU (L1) |
| GROUP(axis, amount) | Two-stage reduction | सभी |
| GROUPTOP(axis, amount) | Tensor cores के लिए grouped reduction | GPU |
| THREAD(axis, amount) | CPU thread-based parallelism | CPU |
| SWAP(axis1, axis2) | Global dimensions reorder करे | सभी |
| PADTO(axis, amount) | Alignment के लिए pad करे | सभी |
| NOLOCALS | Local memory disable करे | सभी (constraint) |
| TC | Tensor core usage enable करे | NVIDIA GPUs |

Total action space ~162 base actions है (kernel structure और available parallelism के अनुसार अलग-अलग)।

---

## Heuristics (डिफ़ॉल्ट)

Heuristic optimizer एक fixed order में optimizations apply करता है (simplified pseudocode):

```rust
// Pseudocode — simplified from optimizer/heuristics.rs
fn hand_coded_optimizations(scheduler: &mut Scheduler) {
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

**फ़ायदे**: तेज़ (~50ms प्रति कर्नेल), predictable, hardware measurement नहीं चाहिए।

**नुकसान**: ऑप्टिमाइज़ेशन के मौके छूट सकते हैं, fixed heuristics workload के हिसाब से adapt नहीं करते।

---

## Beam Search (ऑप्शनल)

प्रोडक्शन workloads के लिए, beam search बेहतर schedules ढूँढता है candidates compile और time करके (simplified pseudocode):

```rust
// Pseudocode — simplified from optimizer/beam.rs
// Actual API: beam_search_cached(scheduler, config, compile_and_time) -> Result<BeamResult>
fn beam_search(scheduler: Scheduler, config: BeamConfig) -> Scheduler {
    let mut beam = vec![scheduler];
    let deadline = Instant::now() + config.time_limit;

    while Instant::now() < deadline {
        let mut candidates = vec![];

        for state in &beam {
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

**फ़ायदे**: Near-optimal schedules ढूँढता है, hardware के हिसाब से adapt करता है।

**नुकसान**: प्रति कर्नेल मिनटों में (लेकिन results AST hash से cache होते हैं)।

---

## कॉन्फ़िगरेशन

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

| पहलू | XLA | TVM/Ansor | Triton | **Morok** |
|-------|-----|-----------|--------|-----------|
| **फ़िलॉसफ़ी** | फ़िक्स्ड heuristics | सर्च-आधारित | प्रोग्रामर-गाइडेड | पैटर्न-आधारित |
| **Fusion** | कंज़र्वेटिव नियम | Tile-and-fuse | Block-level | ग्राफ़ रीराइटिंग |
| **Auto-tuning** | कोई नहीं | Evolutionary + cost model | Grid search | Beam search |
| **ट्यूनिंग कॉस्ट** | 0 | घंटों | मिनटों | मिनटों (कैश्ड) |
| **फ़्लेक्सिबिलिटी** | कम | ज़्यादा | मध्यम | ज़्यादा |
| **ट्रांसपैरेंसी** | कम (C++ पासेज़) | मध्यम (Python) | मध्यम (DSL) | ज़्यादा (declarative patterns) |

**XLA** fusion decisions के लिए fixed heuristics इस्तेमाल करता है। Safe और predictable, लेकिन performance table पर छूट जाती है। Fusion rules C++ में hard-coded हैं।

**TVM/Ansor** *क्या* compute करना है और *कैसे* compute करना है को अलग करता है। Ansor learned cost model के साथ evolutionary search इस्तेमाल करता है। Best-in-class performance, लेकिन tuning में प्रति model घंटे लगते हैं।

**Triton** blocked algorithms के लिए Python-जैसा DSL expose करता है। Control और automation का अच्छा balance, लेकिन GPU programming expertise चाहिए।

**Morok** optimizations को composable patterns में express करता है। Beam search ज़रूरत पड़ने पर auto-tuning जोड़ता है, results reuse के लिए AST hash से cache होते हैं।
