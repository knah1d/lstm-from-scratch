# LSTM cell backward derivation (BPTT, single cell)

Notation: `dX` means `dL/dX`. `⊙` is elementwise product. Row-vector convention
(matches `core/cell.py`): `x_t` is `(batch, input_size)`, `h_t`/`C_t` are
`(batch, hidden_size)`, `Wx_*` is `(input_size, hidden_size)`, `Wh_*` is
`(hidden_size, hidden_size)`.

## Forward recap

```
f_pre = x_t @ Wx_f + h_prev @ Wh_f + b_f      f_t = sigmoid(f_pre)
i_pre = x_t @ Wx_i + h_prev @ Wh_i + b_i      i_t = sigmoid(i_pre)
o_pre = x_t @ Wx_o + h_prev @ Wh_o + b_o      o_t = sigmoid(o_pre)
g_pre = x_t @ Wx_g + h_prev @ Wh_g + b_g      g_t = tanh(g_pre)

C_t = f_t ⊙ C_prev + i_t ⊙ g_t
h_t = o_t ⊙ tanh(C_t)
```

## Why two accumulators (`dh_next`, `dC_next`)

`h_prev` and `C_prev` are each used twice going forward: once to produce
`h_t`/`C_t` at this step, and `h_t`/`C_t` themselves feed into step `t+1`.
So going backward, each step receives gradient contributions from two
places and must sum them:

- `dh_t = dh_ext_t + dh_next` — `dh_ext_t` is any gradient landing on `h_t`
  directly from an external loss at this timestep (e.g. an output head);
  `dh_next` is what step `t+1` sends back because it used `h_t` as its
  `h_prev`.
- `dC_t = dC_next + dh_t ⊙ o_t ⊙ dtanh(C_t)` — `dC_next` is what step `t+1`
  sends back because it used `C_t` as its `C_prev`; the second term is
  `C_t`'s direct contribution to `h_t` in *this* step (`h_t = o_t ⊙
  tanh(C_t)` → `∂h_t/∂C_t = o_t ⊙ (1 - tanh(C_t)²)`).

This is the standard "two state highways" structure of an LSTM: gradient
flows back through both `h` and `C`, and `C_t`'s path is why LSTMs don't
suffer vanishing gradients the way plain RNNs do — there's no repeated
squashing nonlinearity on the `C` path, just the elementwise `f_t` gate.

## Gate-level gradients

From `C_t = f_t ⊙ C_prev + i_t ⊙ g_t`:

```
df_t = dC_t ⊙ C_prev      (∂C_t/∂f_t = C_prev)
di_t = dC_t ⊙ g_t         (∂C_t/∂i_t = g_t)
dg_t = dC_t ⊙ i_t         (∂C_t/∂g_t = i_t)
```

From `h_t = o_t ⊙ tanh(C_t)`:

```
do_t = dh_t ⊙ tanh(C_t)   (∂h_t/∂o_t = tanh(C_t))
```

## Pre-activation gradients (through sigmoid/tanh)

Using `σ'(x) = σ(x)(1-σ(x))` and `tanh'(x) = 1 - tanh(x)²`, applied to the
*already-computed* activations (no need to re-evaluate at `*_pre`):

```
df_pre = df_t ⊙ f_t ⊙ (1 - f_t)
di_pre = di_t ⊙ i_t ⊙ (1 - i_t)
do_pre = do_t ⊙ o_t ⊙ (1 - o_t)
dg_pre = dg_t ⊙ (1 - g_t²)
```

## Weight and bias gradients

Weights are shared across all timesteps, so their gradients **accumulate**
(sum, not overwrite) over the backward loop. For each gate `k ∈ {f,i,o,g}`:

```
dWx_k += x_t.T @ dk_pre
dWh_k += h_prev.T @ dk_pre
db_k  += sum(dk_pre, axis=0)     # sum over batch dimension
```

## Propagating to inputs and previous states

`x_t` and `h_prev` each feed all four gates, so their gradients sum across
gates:

```
dx_t = df_pre @ Wx_f.T + di_pre @ Wx_i.T + do_pre @ Wx_o.T + dg_pre @ Wx_g.T

dh_prev = df_pre @ Wh_f.T + di_pre @ Wh_i.T + do_pre @ Wh_o.T + dg_pre @ Wh_g.T
```

`dh_prev` here is exactly the `dh_next` that step `t-1` will receive.

`C_prev` only feeds `C_t` via the forget gate, so:

```
dC_prev = dC_t ⊙ f_t
```

`dC_prev` here is exactly the `dC_next` that step `t-1` will receive.

## Summary — what `cell.backward()` needs to compute

Inputs: `dh_ext_t` (external gradient on `h_t`, may be zero), `dh_next`,
`dC_next` (accumulators from step `t+1`), and `cache` from the matching
`forward()` call.

```
dh_t = dh_ext_t + dh_next
dC_t = dC_next + dh_t ⊙ o_t ⊙ (1 - tanh(C_t)²)

df_t = dC_t ⊙ C_prev;  di_t = dC_t ⊙ g_t;  dg_t = dC_t ⊙ i_t;  do_t = dh_t ⊙ tanh(C_t)

df_pre = df_t ⊙ f_t ⊙ (1-f_t)
di_pre = di_t ⊙ i_t ⊙ (1-i_t)
do_pre = do_t ⊙ o_t ⊙ (1-o_t)
dg_pre = dg_t ⊙ (1-g_t²)

dWx_k += x_t.T @ dk_pre         for k in f,i,o,g
dWh_k += h_prev.T @ dk_pre
db_k  += sum(dk_pre, axis=0)

dx_t    = Σ_k dk_pre @ Wx_k.T
dh_prev = Σ_k dk_pre @ Wh_k.T   # → dh_next for step t-1
dC_prev = dC_t ⊙ f_t            # → dC_next for step t-1
```

Outputs to return: `dx_t`, `dh_prev`, `dC_prev`, and the weight/bias
gradient dict (accumulated, so the caller passes the same dict in on every
step of a sequence).

This will be implemented one gate at a time (Step 4), with numerical
gradient checking (`(L(w+ε)-L(w-ε))/2ε` vs. analytical) as the
non-negotiable pass/fail gate before moving to full-sequence BPTT.
