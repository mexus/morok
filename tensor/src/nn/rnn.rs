use super::*;

/// Output of an RNN forward pass.
pub struct RnnOutput {
    /// All hidden states: `[seq_length, num_directions, batch, hidden_size]`
    pub y: Tensor,
    /// Final hidden state: `[num_directions, batch, hidden_size]`
    pub y_h: Tensor,
}

/// Output of an LSTM forward pass.
pub struct LstmOutput {
    /// All hidden states: `[seq_length, num_directions, batch, hidden_size]`
    pub y: Tensor,
    /// Final hidden state: `[num_directions, batch, hidden_size]`
    pub y_h: Tensor,
    /// Final cell state: `[num_directions, batch, hidden_size]`
    pub y_c: Tensor,
}

impl Tensor {
    /// Simple RNN (Elman network).
    ///
    /// `H_t = tanh(X_t @ W^T + H_{t-1} @ R^T + Wb + Rb)`
    ///
    /// - `x`: input sequence `[seq_length, batch_size, input_size]`
    /// - `w`: input weights `[num_directions, hidden_size, input_size]`
    /// - `r`: recurrence weights `[num_directions, hidden_size, hidden_size]`
    /// - `bias`: optional bias `[num_directions, 2 * hidden_size]` (Wb ++ Rb)
    /// - `initial_h`: optional initial hidden state `[num_directions, batch_size, hidden_size]`
    pub fn rnn(
        &self,
        w: &Tensor,
        r: &Tensor,
        bias: Option<&Tensor>,
        initial_h: Option<&Tensor>,
        hidden_size: usize,
    ) -> Result<RnnOutput> {
        let x = self;
        let x_shape = x.shape()?;
        let seq_length = x_shape[0].as_const().expect("static seq_length");
        let batch_size = x_shape[1].as_const().expect("static batch_size");
        let num_directions = w.shape()?[0].as_const().expect("static num_directions");
        let dtype = x.uop().dtype();

        assert_eq!(num_directions, 1, "RNN: only forward direction supported");

        let w0 = w.try_squeeze(Some(0))?; // [hidden, input]
        let r0 = r.try_squeeze(Some(0))?; // [hidden, hidden]
        let wt = w0.try_permute(&[1, 0])?; // [input, hidden]
        let rt = r0.try_permute(&[1, 0])?; // [hidden, hidden]

        let combined_bias = if let Some(b) = bias {
            let b0 = b.try_squeeze(Some(0))?; // [2*hidden]
            let parts = b0.split(&[hidden_size, hidden_size], 0)?;
            Some(parts[0].try_add(&parts[1])?) // [hidden]
        } else {
            None
        };

        let mut h_t = if let Some(h0) = initial_h {
            h0.try_squeeze(Some(0))? // [batch, hidden]
        } else {
            Tensor::full(&[batch_size, hidden_size], 0.0f32, dtype)?
        };

        let input_size = x_shape[2].as_const().expect("static input_size");
        let mut h_list = Vec::with_capacity(seq_length);
        for t in 0..seq_length {
            let x_t =
                x.try_shrink(&[(t as isize, t as isize + 1), (0, batch_size as isize), (0, input_size as isize)])?;
            let x_t = x_t.try_squeeze(Some(0))?; // [batch, input]

            let mut gate = x_t.matmul(&wt)?.try_add(&h_t.matmul(&rt)?)?;
            if let Some(ref b) = combined_bias {
                gate = gate.try_add(b)?;
            }
            h_t = gate.tanh()?;
            h_list.push(h_t.clone());
        }

        let h_refs: Vec<&Tensor> = h_list.iter().collect();
        let y_seq = Tensor::stack(&h_refs, 0)?; // [seq, batch, hidden]
        let y = y_seq.try_unsqueeze(1)?; // [seq, 1, batch, hidden]
        let y_h = h_t.try_unsqueeze(0)?; // [1, batch, hidden]

        Ok(RnnOutput { y, y_h })
    }

    /// LSTM (Long Short-Term Memory).
    ///
    /// Gate order: `[i, o, f, c]` (input, output, forget, cell).
    ///
    /// - `x`: input `[seq_length, batch_size, input_size]` (layout=0) or
    ///         `[batch_size, seq_length, input_size]` (layout=1)
    /// - `w`: input weights `[num_directions, 4*hidden_size, input_size]`
    /// - `r`: recurrence weights `[num_directions, 4*hidden_size, hidden_size]`
    /// - `bias`: optional `[num_directions, 8*hidden_size]` (Wb ++ Rb)
    /// - `initial_h`: optional `[num_directions, batch_size, hidden_size]`
    /// - `initial_c`: optional `[num_directions, batch_size, hidden_size]`
    /// - `peepholes`: optional `[num_directions, 3*hidden_size]` (p_i, p_o, p_f)
    /// - `layout`: 0 = seq-first (default), 1 = batch-first
    pub fn lstm(
        &self,
        w: &Tensor,
        r: &Tensor,
        bias: Option<&Tensor>,
        initial_h: Option<&Tensor>,
        initial_c: Option<&Tensor>,
        peepholes: Option<&Tensor>,
        hidden_size: usize,
        layout: usize,
    ) -> Result<LstmOutput> {
        let x = if layout != 0 {
            self.try_permute(&[1, 0, 2])? // batch-first → seq-first
        } else {
            self.clone()
        };
        let x_shape = x.shape()?;
        let seq_length = x_shape[0].as_const().expect("static seq_length");
        let batch_size = x_shape[1].as_const().expect("static batch_size");
        let input_size = x_shape[2].as_const().expect("static input_size");
        let num_directions = w.shape()?[0].as_const().expect("static num_directions");
        let dtype = x.uop().dtype();

        assert_eq!(num_directions, 1, "LSTM: only forward direction supported");

        let w0 = w.try_squeeze(Some(0))?; // [4*hidden, input]
        let r0 = r.try_squeeze(Some(0))?; // [4*hidden, hidden]
        let wt = w0.try_permute(&[1, 0])?; // [input, 4*hidden]
        let rt = r0.try_permute(&[1, 0])?; // [hidden, 4*hidden]

        // Bias: [8*hidden] → split into Wb [4*hidden] and Rb [4*hidden], add together
        let combined_bias = if let Some(b) = bias {
            let b0 = b.try_squeeze(Some(0))?;
            let hs4 = 4 * hidden_size;
            let parts = b0.split(&[hs4, hs4], 0)?;
            Some(parts[0].try_add(&parts[1])?)
        } else {
            None
        };

        // Peepholes: [3*hidden] → [p_i, p_o, p_f]
        let (p_i, p_o, p_f) = if let Some(p) = peepholes {
            let p0 = p.try_squeeze(Some(0))?;
            let parts = p0.split(&[hidden_size, hidden_size, hidden_size], 0)?;
            (Some(parts[0].clone()), Some(parts[1].clone()), Some(parts[2].clone()))
        } else {
            (None, None, None)
        };

        let mut h_t = if let Some(h0) = initial_h {
            h0.try_squeeze(Some(0))?
        } else {
            Tensor::full(&[batch_size, hidden_size], 0.0f32, dtype.clone())?
        };
        let mut c_t = if let Some(c0) = initial_c {
            c0.try_squeeze(Some(0))?
        } else {
            Tensor::full(&[batch_size, hidden_size], 0.0f32, dtype)?
        };

        let mut h_list = Vec::with_capacity(seq_length);
        for t in 0..seq_length {
            let x_t = x.try_shrink(&[
                (t as isize, t as isize + 1),
                (0, batch_size as isize),
                (0, input_size as isize),
            ])?;
            let x_t = x_t.try_squeeze(Some(0))?; // [batch, input]

            // gates = X_t @ W^T + H_{t-1} @ R^T + bias
            let mut gates = x_t.matmul(&wt)?.try_add(&h_t.matmul(&rt)?)?;
            if let Some(ref b) = combined_bias {
                gates = gates.try_add(b)?;
            }

            // Split into [i, o, f, c] — each [batch, hidden]
            let gate_parts = gates.split(&[hidden_size; 4], -1)?;
            let (mut gi, mut go, mut gf, gc) =
                (gate_parts[0].clone(), gate_parts[1].clone(), gate_parts[2].clone(), gate_parts[3].clone());

            // Peephole connections: i and f use previous cell state
            if let Some(ref pi) = p_i {
                gi = gi.try_add(&c_t.try_mul(pi)?)?;
            }
            if let Some(ref pf) = p_f {
                gf = gf.try_add(&c_t.try_mul(pf)?)?;
            }

            let i = gi.sigmoid()?;
            let f = gf.sigmoid()?;
            let c = gc.tanh()?;

            // C = f * C_prev + i * c
            c_t = f.try_mul(&c_t)?.try_add(&i.try_mul(&c)?)?;

            // Peephole: o uses NEW cell state
            if let Some(ref po) = p_o {
                go = go.try_add(&c_t.try_mul(po)?)?;
            }
            let o = go.sigmoid()?;

            // H = o * tanh(C)
            h_t = o.try_mul(&c_t.tanh()?)?;
            h_list.push(h_t.clone());
        }

        let h_refs: Vec<&Tensor> = h_list.iter().collect();
        let y_seq = Tensor::stack(&h_refs, 0)?; // [seq, batch, hidden]
        let y = y_seq.try_unsqueeze(1)?; // [seq, 1, batch, hidden]

        // Apply layout transform to output
        let y = if layout != 0 {
            y.try_permute(&[2, 0, 1, 3])? // [batch, seq, 1, hidden]
        } else {
            y
        };

        let (y_h, y_c) = if layout != 0 {
            // layout=1: Y_h/Y_c are [batch, num_directions, hidden]
            (h_t.try_unsqueeze(1)?, c_t.try_unsqueeze(1)?)
        } else {
            // layout=0: Y_h/Y_c are [num_directions, batch, hidden]
            (h_t.try_unsqueeze(0)?, c_t.try_unsqueeze(0)?)
        };

        Ok(LstmOutput { y, y_h, y_c })
    }
}
