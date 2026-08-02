//! DTW word-level alignment via cross-attention weights.
//!
//! Pure CPU implementation matching `whisper/timing.py`. Pipeline:
//! 1. Extract cross-attention QK weights from alignment heads
//! 2. Softmax over time → standardize → median filter (width=7)
//! 3. Average across heads → `matrix[n_tokens, n_audio_frames]`
//! 4. DTW on `-matrix` → backtrace → `(text_idx, time_idx)` path
//! 5. Map jump points to word boundaries

/// One word's timing from DTW alignment.
#[derive(Clone, Debug)]
pub struct WordTiming {
    pub word: String,
    pub tokens: Vec<u32>,
    pub start: f32,
    pub end: f32,
    pub probability: f32,
}

/// Classic DTW + backtrace on a cost matrix. Returns `(text_indices, time_indices)`.
///
/// Matches `whisper.timing.dtw_cpu`. Input `x` is `[N, M]` row-major (N=text,
/// M=time). Returns two parallel vectors that together form the warp path.
pub fn dtw(x: &[f32], n_rows: usize, n_cols: usize) -> (Vec<usize>, Vec<usize>) {
    let n = n_rows;
    let m = n_cols;
    let cost_size = (n + 1) * (m + 1);

    // Cost matrix (row-major, (N+1)×(M+1)), initialized to +inf
    let mut cost = vec![f32::INFINITY; cost_size];
    let mut trace = vec![-1i32; cost_size];

    // cost[0,0] = 0
    cost[0] = 0.0;

    // Fill cost matrix
    for j in 1..=m {
        for i in 1..=n {
            let c0 = cost[(i - 1) * (m + 1) + (j - 1)]; // diagonal
            let c1 = cost[(i - 1) * (m + 1) + j]; // up
            let c2 = cost[i * (m + 1) + (j - 1)]; // left

            let (c, t) = if c0 < c1 && c0 < c2 {
                (c0, 0i32)
            } else if c1 < c0 && c1 < c2 {
                (c1, 1i32)
            } else {
                (c2, 2i32)
            };

            cost[i * (m + 1) + j] = x[(i - 1) * m + (j - 1)] + c;
            trace[i * (m + 1) + j] = t;
        }
    }

    // Backtrace setup: trace[0, :] = 2 (first row), trace[:, 0] = 1 (first column)
    // The original Python sets both, but the second overwrites on trace[0,0].
    for slot in &mut trace[..=m] {
        *slot = 2;
    }
    for i in 0..=n {
        trace[i * (m + 1)] = 1; // trace[i, 0] = 1  (first column, row i)
    }

    let mut text_indices = Vec::new();
    let mut time_indices = Vec::new();
    let mut i = n;
    let mut j = m;

    while i > 0 || j > 0 {
        text_indices.push(i - 1);
        time_indices.push(j - 1);

        match trace[i * (m + 1) + j] {
            0 => {
                i -= 1;
                j -= 1;
            }
            1 => {
                i -= 1;
            }
            2 => {
                j -= 1;
            }
            _ => break,
        }
    }

    text_indices.reverse();
    time_indices.reverse();
    (text_indices, time_indices)
}

/// Median filter along the last axis with reflect padding.
/// Input shape `[.., width]` flattened. `data` is `[n_rows * n_cols]` with
/// `filter_width` applied to the last axis (n_cols).
pub fn median_filter(data: &[f32], n_rows: usize, n_cols: usize, filter_width: usize) -> Vec<f32> {
    assert!(filter_width > 0 && filter_width % 2 == 1, "filter_width must be odd");
    let pad = filter_width / 2;
    let mut out = vec![0.0f32; n_rows * n_cols];

    for i in 0..n_rows {
        for j in 0..n_cols {
            let mut window = Vec::with_capacity(filter_width);
            for off in -(pad as isize)..=(pad as isize) {
                let idx = j as isize + off;
                // Reflect padding (single-bounce)
                let idx = if idx < 0 {
                    -idx
                } else if idx >= n_cols as isize {
                    2 * (n_cols as isize - 1) - idx
                } else {
                    idx
                };
                let idx = idx.clamp(0, n_cols as isize - 1) as usize;
                window.push(data[i * n_cols + idx]);
            }
            window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            out[i * n_cols + j] = window[filter_width / 2];
        }
    }
    out
}

/// Extract alignment matrix from cross-attention weights and run DTW.
///
/// `qk_weights[layer]` is `[B, H, S_text, S_audio]` softmaxed attention weights.
/// `alignment_heads` is a list of `(layer, head)` pairs.
/// `num_frames` is the actual number of audio frames to use (≤ S_audio).
/// `medfilt_width` is the median filter width (default 7).
///
/// Returns `(text_indices, time_indices)` from DTW on the alignment matrix.
#[allow(clippy::too_many_arguments)]
pub fn find_alignment_path(
    qk_weights: &[Vec<f32>], // per-layer, flattened [B*H*S_text*S_audio]
    _batch: usize,
    _n_heads: usize,
    s_text: usize,
    s_audio: usize,
    alignment_heads: &[(usize, usize)],
    num_frames: usize,
    medfilt_width: usize,
    sot_len: usize, // prefix tokens to strip from text axis (SOT sequence only, not no_timestamps)
) -> (Vec<usize>, Vec<usize>) {
    let n_heads_sel = alignment_heads.len();
    let audio_frames = (num_frames / 2).min(s_audio);

    // Collect weights for selected heads: [n_sel, s_text, audio_frames]
    let mut weights = vec![0.0f32; n_heads_sel * s_text * audio_frames];
    for (sel_i, &(layer, head)) in alignment_heads.iter().enumerate() {
        let layer_data = &qk_weights[layer];
        // layer_data is [B, H, S_text, S_audio]. We use batch=0.
        for t in 0..s_text {
            for f in 0..audio_frames {
                let src = (head * s_text + t) * s_audio + f;
                let dst = (sel_i * s_text + t) * audio_frames + f;
                weights[dst] = layer_data[src];
            }
        }
    }

    // Softmax over time axis (last axis)
    for sel_i in 0..n_heads_sel {
        for t in 0..s_text {
            let row = &mut weights[(sel_i * s_text + t) * audio_frames..(sel_i * s_text + t + 1) * audio_frames];
            let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            if max_val == f32::NEG_INFINITY {
                continue;
            }
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - max_val).exp();
                sum += *v;
            }
            for v in row.iter_mut() {
                *v /= sum.max(1e-10);
            }
        }
    }

    // Standardize over token axis (axis=1): (x - mean) / std
    for sel_i in 0..n_heads_sel {
        for f in 0..audio_frames {
            // Collect column values
            let col: Vec<f32> = (0..s_text).map(|t| weights[(sel_i * s_text + t) * audio_frames + f]).collect();
            let mean = col.iter().sum::<f32>() / s_text as f32;
            let variance = col.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / s_text as f32;
            let std = variance.sqrt().max(1e-10);
            for t in 0..s_text {
                let idx = (sel_i * s_text + t) * audio_frames + f;
                weights[idx] = (weights[idx] - mean) / std;
            }
        }
    }

    // Median filter over time axis
    let filtered = median_filter(&weights, n_heads_sel * s_text, audio_frames, medfilt_width);

    // Average over selected heads → [s_text, audio_frames]
    let mut matrix = vec![0.0f32; s_text * audio_frames];
    for t in 0..s_text {
        for f in 0..audio_frames {
            let mut sum = 0.0f32;
            for sel_i in 0..n_heads_sel {
                sum += filtered[(sel_i * s_text + t) * audio_frames + f];
            }
            matrix[t * audio_frames + f] = sum / n_heads_sel as f32;
        }
    }

    // Strip SOT prefix and EOT suffix from text axis
    let text_len = s_text - sot_len - 1; // -1 for EOT
    let stripped: Vec<f32> = (0..text_len)
        .flat_map(|t| {
            let row_start = (t + sot_len) * audio_frames;
            &matrix[row_start..row_start + audio_frames]
        })
        .copied()
        .collect();

    // DTW on negated matrix
    let negated: Vec<f32> = stripped.iter().map(|&x| -x).collect();
    dtw(&negated, text_len, audio_frames)
}

/// Convert a DTW alignment path into word-level timings.
///
/// `text_indices`, `time_indices`: DTW path. `word_boundaries`: cumulative
/// token counts at word starts (including leading 0). `token_probs`: per-token
/// probabilities from the decoder.
pub fn path_to_word_timings(
    text_indices: &[usize],
    time_indices: &[usize],
    word_boundaries: &[usize],
    words: &[String],
    word_token_lists: &[Vec<u32>],
    token_probs: &[f32],
    tokens_per_second: f32,
) -> Vec<WordTiming> {
    if word_boundaries.len() <= 1 {
        return Vec::new();
    }

    // Find jump points: positions where text_indices changes
    let mut jumps = vec![false; text_indices.len()];
    jumps[0] = true;
    for i in 1..text_indices.len() {
        jumps[i] = text_indices[i] != text_indices[i - 1];
    }

    // Map jump times
    let jump_times: Vec<f32> =
        time_indices.iter().zip(&jumps).filter(|&(_, j)| *j).map(|(&ti, _)| ti as f32 / tokens_per_second).collect();

    let n_words = word_boundaries.len() - 1;
    let mut result = Vec::with_capacity(n_words);

    for w in 0..n_words {
        let start = jump_times.get(word_boundaries[w]).copied().unwrap_or(0.0);
        let end = jump_times.get(word_boundaries[w + 1]).copied().unwrap_or(start);

        // Token probabilities for this word
        let tok_start = word_boundaries[w];
        let tok_end = word_boundaries[w + 1];
        let prob = if tok_end > tok_start {
            let slice = &token_probs[tok_start..tok_end.min(token_probs.len())];
            slice.iter().sum::<f32>() / slice.len() as f32
        } else {
            0.0
        };

        result.push(WordTiming {
            word: words.get(w).cloned().unwrap_or_default(),
            tokens: word_token_lists.get(w).cloned().unwrap_or_default(),
            start,
            end,
            probability: prob,
        });
    }

    result
}
