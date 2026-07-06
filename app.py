"""
Streamlit walkthrough for the lstm-from-scratch project.

Run with:  streamlit run app.py

Each "step" below is a self-contained slide that imports and calls the
actual code in core/ and experiments/ — nothing here re-implements the
LSTM, it only visualizes what the real implementation is doing.
"""
import os
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.ops import sigmoid, dsigmoid, tanh, dtanh, xavier_init, orthogonal_init
from core.cell import LSTMCell
from core.layer import LSTMLayer
from core.losses import mse_loss
from core.optimizer import SGD, Adam

st.set_page_config(page_title="LSTM from Scratch — Walkthrough", layout="wide")

STEPS = [
    "0. Overview",
    "1. Building Blocks (ops.py)",
    "2. The LSTM Cell (cell.py)",
    "3. Unrolling in Time (layer.py)",
    "4. Backward Pass / BPTT",
    "5. Loss & Optimizers",
    "6. End-to-End: Predict the Next Number",
    "7. Test Suite",
]

if "step" not in st.session_state:
    st.session_state.step = 0

with st.sidebar:
    st.title("LSTM from Scratch")
    choice = st.radio("Steps", STEPS, index=st.session_state.step)
    st.session_state.step = STEPS.index(choice)
    st.markdown("---")
    st.caption("Pure NumPy LSTM: forward pass, backward pass (BPTT), gradient "
               "checking, and a next-number-prediction training experiment.")

col_prev, col_mid, col_next = st.columns([1, 6, 1])
with col_prev:
    if st.button("Prev", disabled=st.session_state.step == 0, use_container_width=True):
        st.session_state.step -= 1
        st.rerun()
with col_next:
    if st.button("Next", disabled=st.session_state.step == len(STEPS) - 1, use_container_width=True):
        st.session_state.step += 1
        st.rerun()

step = st.session_state.step
st.header(STEPS[step])


# ---------------------------------------------------------------- Step 0
def step_overview():
    st.markdown("""
This project implements an **LSTM (Long Short-Term Memory) network from
scratch**, using only NumPy — no PyTorch, no TensorFlow. It covers the
forward pass, the backward pass (backpropagation through time), gradient
verification, and a small training experiment.
    """)

    st.subheader("Pipeline")
    pipeline = """
    ┌────────────┐   ┌──────────────────┐   ┌────────┐   ┌────────┐
    │    DATA    │──▶│  LSTM LAYER      │──▶│  HEAD  │──▶│  LOSS  │
    │ number seq │   │  forward (t=0..T)│   │ linear │   │  MSE   │
    └────────────┘   └──────────────────┘   └────────┘   └───┬────┘
                                                              │ gradients
    ┌──────────────┐   ┌──────────────────┐   ┌────────────┐ │
    │  OPTIMIZER   │◀──│  LSTM LAYER      │◀──│  HEAD      │◀┘
    │  SGD / Adam  │   │  backward (BPTT) │   │  backward  │
    └──────────────┘   └──────────────────┘   └────────────┘
    """
    st.code(pipeline, language=None)

    st.subheader("Repository map")
    st.markdown("""
| File | Role |
|---|---|
| `core/ops.py` | Activation functions (`sigmoid`, `tanh`) + derivatives, weight init |
| `core/cell.py` | `LSTMCell` — forward/backward for **one** timestep |
| `core/layer.py` | `LSTMLayer` — unrolls the cell over a whole sequence (BPTT) |
| `core/losses.py` | MSE and cross-entropy loss + gradient |
| `core/optimizer.py` | `SGD` and `Adam` |
| `experiments/toy_task.py` | Repo's saved experiment: trains the LSTM on a sine wave (offline script, see `toy_task_result.png`) |
| `tests/` | Shape tests + numerical gradient checks (13 tests) |
| `notes/derivations.md` | Full chain-rule derivation of the backward pass |
    """)
    st.caption("This walkthrough's own live demo (Step 6) trains on a different, simpler dataset — "
               "next-number-in-a-sequence prediction — so you can type your own input and get an "
               "answer interactively.")
    st.info("Use **Next** in the top bar (or the sidebar) to walk through each piece step by step.")


# ---------------------------------------------------------------- Step 1
def step_ops():
    st.markdown("`core/ops.py` provides the two nonlinearities every gate uses, "
                "their derivatives (used in the backward pass), and two weight "
                "initialization schemes.")

    c1, c2 = st.columns(2)
    x = np.linspace(-6, 6, 400)

    with c1:
        st.subheader("sigmoid — used by gates f, i, o")
        st.latex(r"\sigma(x) = \frac{1}{1+e^{-x}} \qquad \sigma'(x)=\sigma(x)(1-\sigma(x))")
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        ax.plot(x, sigmoid(x), label="sigmoid(x)")
        ax.plot(x, dsigmoid(x), label="dsigmoid(x)", linestyle="--")
        ax.axhline(0, color="gray", lw=0.5)
        ax.legend()
        st.pyplot(fig)
        st.caption("Squashes to (0, 1) → interpreted as a 'how much to let through' gate.")

    with c2:
        st.subheader("tanh — used by candidate g and cell readout")
        st.latex(r"\tanh(x) \qquad \tanh'(x)=1-\tanh(x)^2")
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        ax.plot(x, tanh(x), label="tanh(x)")
        ax.plot(x, dtanh(x), label="dtanh(x)", linestyle="--")
        ax.axhline(0, color="gray", lw=0.5)
        ax.legend()
        st.pyplot(fig)
        st.caption("Squashes to (-1, 1) → used where a signed value/update is needed.")

    st.subheader("Weight initialization")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Xavier init** (`Wx`, input → gate) — keeps activation variance stable.")
        w = xavier_init((64, 64))
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(w.flatten(), bins=40)
        st.pyplot(fig)
    with c4:
        st.markdown("**Orthogonal init** (`Wh`, hidden → gate) — rows/cols are orthonormal, "
                     "which helps gradients neither explode nor vanish through recurrence.")
        w = orthogonal_init((30, 30))
        fig, ax = plt.subplots(figsize=(4, 3))
        im = ax.imshow(w @ w.T, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_title("W · Wᵗ  (should be ≈ identity)")
        fig.colorbar(im, ax=ax, fraction=0.046)
        st.pyplot(fig)


# ---------------------------------------------------------------- Step 2
def step_cell():
    st.markdown("`core/cell.py` — `LSTMCell` computes **one timestep**: given "
                "`x_t`, `h_prev`, `C_prev`, it produces `h_t`, `C_t`.")

    st.latex(r"""
    \begin{aligned}
    f_t &= \sigma(x_t W_x^f + h_{t-1} W_h^f + b^f) &\text{(forget gate)}\\
    i_t &= \sigma(x_t W_x^i + h_{t-1} W_h^i + b^i) &\text{(input gate)}\\
    o_t &= \sigma(x_t W_x^o + h_{t-1} W_h^o + b^o) &\text{(output gate)}\\
    g_t &= \tanh(x_t W_x^g + h_{t-1} W_h^g + b^g) &\text{(candidate)}\\
    C_t &= f_t \odot C_{t-1} + i_t \odot g_t &\text{(new cell state)}\\
    h_t &= o_t \odot \tanh(C_t) &\text{(new hidden state)}
    \end{aligned}
    """)
    st.caption("Forget-gate bias is initialized to 1.0 so the gate starts near-open, "
               "protecting gradient flow early in training (core/cell.py:18).")

    st.subheader("Try it")
    c1, c2, c3 = st.columns(3)
    with c1:
        hidden_size = st.slider("hidden_size", 2, 12, 4)
    with c2:
        input_size = st.slider("input_size", 1, 6, 3)
    with c3:
        seed = st.slider("random seed", 0, 100, 0)

    rng = np.random.RandomState(seed)
    cell = LSTMCell(input_size, hidden_size, rng)
    x_t = rng.randn(1, input_size)
    h_prev = rng.randn(1, hidden_size) * 0.3
    C_prev = rng.randn(1, hidden_size) * 0.3

    h_t, C_t, cache = cell.forward(x_t, h_prev, C_prev)

    st.markdown(f"Random `x_t` (shape `(1, {input_size})`), `h_prev`/`C_prev` (shape `(1, {hidden_size})`) fed through the cell:")

    labels = [f"u{i}" for i in range(hidden_size)]
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.6), sharey=True)
    for ax, gate, title in zip(axes, "fiog", ["forget f_t", "input i_t", "output o_t", "candidate g_t"]):
        vals = cache[f"{gate}_t"].flatten()
        colors = ["#4C72B0" if v >= 0 else "#DD8452" for v in vals]
        ax.bar(labels, vals, color=colors)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0, color="gray", lw=0.5)
        ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    st.pyplot(fig)

    c4, c5 = st.columns(2)
    with c4:
        st.markdown("**New cell state `C_t`**")
        st.dataframe(C_t.round(4), hide_index=True)
    with c5:
        st.markdown("**New hidden state `h_t`**")
        st.dataframe(h_t.round(4), hide_index=True)


# ---------------------------------------------------------------- Step 3
def step_layer():
    st.markdown("`core/layer.py` — `LSTMLayer` loops the cell over a sequence "
                "`t = 0..T-1`, feeding `h_t`/`C_t` forward into the next step.")
    st.code(
        "for t in range(T):\n"
        "    h_t, C_t, cache = self.cell.forward(x_seq[:, t, :], h_prev, C_prev)\n"
        "    h_seq[:, t, :] = h_t\n"
        "    h_prev, C_prev = h_t, C_t",
        language="python",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        hidden_size = st.slider("hidden_size", 4, 24, 12, key="layer_hidden")
    with c2:
        T = st.slider("sequence length T", 5, 60, 20, key="layer_T")
    with c3:
        seed = st.slider("random seed", 0, 100, 0, key="layer_seed")

    rng = np.random.RandomState(seed)
    start, step_ = rng.uniform(-3, 3), rng.uniform(-1, 1)
    x_seq = (start + step_ * np.arange(T))[None, :, None]  # (1, T, 1) — a number sequence

    layer = LSTMLayer(input_size=1, hidden_size=hidden_size, rng=rng)
    h_seq, C_final, caches = layer.forward(x_seq)

    st.subheader("Input sequence (a constant-step number sequence)")
    fig, ax = plt.subplots(figsize=(9, 2))
    ax.plot(x_seq.flatten(), marker="o", markersize=3)
    ax.set_xlabel("t")
    st.pyplot(fig)

    st.subheader("Hidden state h_t across time and hidden units")
    fig, ax = plt.subplots(figsize=(9, 3.5))
    im = ax.imshow(h_seq[0].T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xlabel("timestep t")
    ax.set_ylabel("hidden unit")
    fig.colorbar(im, ax=ax, fraction=0.03, label="h_t value")
    st.pyplot(fig)
    st.caption("Each row is one hidden unit's activation over the whole sequence — "
               "this is the 'unrolled' view of the recurrence.")

    st.subheader("Forget gate f_t across time (unit 0)")
    f_over_time = np.array([c["f_t"][0, 0] for c in caches])
    fig, ax = plt.subplots(figsize=(9, 2))
    ax.plot(f_over_time, color="#4C72B0")
    ax.set_ylim(0, 1)
    ax.set_xlabel("timestep t")
    ax.set_ylabel("f_t[unit 0]")
    st.pyplot(fig)
    st.caption("Values near 1.0 mean the cell is mostly *keeping* its memory at that step "
               "(consistent with the forget-gate bias initialized to +1).")


# ---------------------------------------------------------------- Step 4
def step_backward():
    st.markdown("""
The backward pass propagates a loss gradient **back through time**. Because
`h_prev` and `C_prev` each feed two places (this step's output, and the next
step's input), two accumulators carry gradient backward: `dh_next` and
`dC_next`. See `notes/derivations.md` for the full chain-rule derivation.
    """)
    st.latex(r"""
    \begin{aligned}
    dh_t &= dh_{ext} + dh_{next} \\
    dC_t &= dC_{next} + dh_t \odot o_t \odot (1-\tanh(C_t)^2) \\
    dC_{prev} &= dC_t \odot f_t \quad\text{(→ becomes } dC_{next}\text{ for step } t-1)
    \end{aligned}
    """)
    st.markdown("This is why LSTMs resist vanishing gradients: the cell-state path "
                "back through time is multiplied only by the elementwise forget gate "
                "`f_t`, not repeatedly squashed by a nonlinearity.")

    st.subheader("Live gradient check")
    st.markdown("This numerically verifies the analytic backward pass: for each weight, "
                "compare the hand-derived gradient against the finite-difference estimate "
                "`(L(w+ε) − L(w−ε)) / 2ε`. This is exactly what `tests/test_gradients.py` does.")

    if st.button("Run gradient check now", type="primary"):
        rng = np.random.RandomState(0)
        input_size, hidden_size, batch = 3, 4, 2
        cell = LSTMCell(input_size, hidden_size, rng)
        x_t = rng.randn(batch, input_size)
        h_prev = rng.randn(batch, hidden_size)
        C_prev = rng.randn(batch, hidden_size)
        eps = 1e-5

        def compute_loss():
            h_t, _, _ = cell.forward(x_t, h_prev, C_prev)
            return 0.5 * np.sum(h_t ** 2)

        h_t, C_t, cache = cell.forward(x_t, h_prev, C_prev)
        dh_ext = h_t.copy()
        dh_next = np.zeros_like(h_t)
        dC_next = np.zeros_like(C_t)
        _, _, _, grads = cell.backward(dh_ext, dh_next, dC_next, cache)

        rows = []
        check_rng = np.random.RandomState(123)
        for gate in "fiog":
            for _ in range(2):
                i, j = check_rng.randint(0, input_size), check_rng.randint(0, hidden_size)
                orig = cell.Wx[gate][i, j]
                cell.Wx[gate][i, j] = orig + eps
                Lp = compute_loss()
                cell.Wx[gate][i, j] = orig - eps
                Lm = compute_loss()
                cell.Wx[gate][i, j] = orig
                numeric = (Lp - Lm) / (2 * eps)
                analytic = grads["Wx"][gate][i, j]
                rel_err = abs(analytic - numeric) / max(1e-8, abs(analytic) + abs(numeric))
                rows.append((f"Wx[{gate}][{i},{j}]", analytic, numeric, rel_err))

        table_md = "| parameter | analytic | numeric | rel. error |\n|---|---|---|---|\n"
        max_err = 0.0
        for name, a, n, e in rows:
            max_err = max(max_err, e)
            table_md += f"| `{name}` | {a:.6f} | {n:.6f} | {e:.2e} |\n"
        st.markdown(table_md)

        if max_err < 1e-5:
            st.success(f"All checked gradients match within tolerance (max rel. error = {max_err:.2e}). ✅")
        else:
            st.error(f"Max rel. error = {max_err:.2e} — exceeds tolerance.")


# ---------------------------------------------------------------- Step 5
def step_loss_optim():
    st.markdown("`core/losses.py` and `core/optimizer.py`.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("MSE loss")
        st.latex(r"L = \frac{1}{N}\sum (pred - target)^2 \qquad \frac{\partial L}{\partial pred} = \frac{2}{N}(pred-target)")
    with c2:
        st.subheader("Cross-entropy loss")
        st.latex(r"L = -\frac{1}{N}\sum \log(\text{softmax}(logits)_{label})")

    st.subheader("SGD vs Adam — same random loss surface")
    st.markdown("A toy 2D bowl-shaped loss, optimized from the same starting point with "
                "both optimizers, to show the difference in trajectory (Adam adapts its "
                "per-parameter step size; SGD does not).")

    def toy_loss_grad(p):
        A = np.array([[3.0, 0.0], [0.0, 0.3]])
        return p @ A, 2 * (p @ A)

    start = np.array([2.5, 2.5])
    sgd = SGD(lr=0.15)
    adam = Adam(lr=0.15)
    p_sgd, p_adam = start.copy(), start.copy()
    path_sgd, path_adam = [p_sgd.copy()], [p_adam.copy()]
    for _ in range(40):
        _, g_sgd = toy_loss_grad(p_sgd)
        sgd.step(p_sgd, g_sgd)
        path_sgd.append(p_sgd.copy())

        _, g_adam = toy_loss_grad(p_adam)
        adam.step(p_adam, g_adam)
        path_adam.append(p_adam.copy())

    path_sgd, path_adam = np.array(path_sgd), np.array(path_adam)
    xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    zz = 1.5 * xx ** 2 + 0.15 * yy ** 2

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contour(xx, yy, zz, levels=15, cmap="Greys", alpha=0.6)
    ax.plot(path_sgd[:, 0], path_sgd[:, 1], marker=".", label="SGD", color="#DD8452")
    ax.plot(path_adam[:, 0], path_adam[:, 1], marker=".", label="Adam", color="#4C72B0")
    ax.scatter([0], [0], marker="*", s=150, color="green", label="minimum")
    ax.legend()
    st.pyplot(fig)


# ---------------------------------------------------------------- Step 6
MIN_LEN, MAX_LEN = 3, 12
RATIO_RANGE = (0.6, 1.6)


def make_pattern_batch(rng, L, samples, pattern):
    """Generate `samples` sequences of length L+1 (last value is the target),
    normalized by their own scale so arithmetic and geometric patterns — which
    live on very different magnitudes — can be trained together without one
    dominating the loss or saturating the gates."""
    if pattern == "arithmetic":
        starts = rng.uniform(-8, 8, samples)
        steps_ = rng.uniform(-3, 3, samples)
        seqs = np.array([starts + steps_ * i for i in range(L + 1)]).T
    else:  # geometric
        starts = rng.uniform(1, 5, samples) * rng.choice([-1.0, 1.0], samples)
        ratios = rng.uniform(*RATIO_RANGE, samples)
        seqs = np.array([starts * (ratios ** i) for i in range(L + 1)]).T

    scale = np.maximum(np.max(np.abs(seqs[:, :-1]), axis=1, keepdims=True), 1e-3)
    seqs = seqs / scale
    return seqs[:, :L, None], seqs[:, L:L + 1]


def step_experiment():
    st.markdown(f"""
This trains the real `LSTMLayer` + a linear output head to **predict the
next number in a sequence**. Training mixes two pattern families — **additive**
(`x, x+d, x+2d, ...`, e.g. `2, 4, 6, 8`) and **multiplicative**
(`x, x·r, x·r², ...`, e.g. `2, 4, 8, 16`) — with lengths ranging from
{MIN_LEN} to {MAX_LEN}. Each sequence is normalized by its own scale before
training (dividing every value by the max magnitude in that sequence) so the
huge range of a multiplicative sequence doesn't swamp the loss or saturate
the gates — the pattern itself (additive step vs. multiplicative ratio) is
unchanged by that scaling. This is easier than `experiments/toy_task.py`'s
sine wave, and it's easy to test yourself afterward: type **any number of
values** and the model predicts what comes next.
    """)

    st.code(
        "for L in range(MIN_LEN, MAX_LEN + 1):\n"
        "    arithmetic: seq[i] = start + step * i          # step   ~ U(-3, 3)\n"
        "    geometric:  seq[i] = start * ratio ** i         # ratio  ~ U(0.6, 1.6)\n"
        "    seq /= max(abs(seq[:-1]))                       # normalize by input scale\n"
        "    X, y = seq[:L], seq[L]",
        language="python",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        hidden_size = st.slider("hidden_size", 4, 32, 20, key="exp_hidden")
    with c2:
        epochs = st.slider("epochs", 10, 300, 150, key="exp_epochs")
    with c3:
        optimizer_name = st.selectbox("optimizer", ["adam", "sgd"], key="exp_optim")

    if st.button("Train now", type="primary"):
        rng = np.random.RandomState(0)
        samples_per_bucket = 40
        lengths = list(range(MIN_LEN, MAX_LEN + 1))

        datasets = {}
        for L in lengths:
            X_a, y_a = make_pattern_batch(rng, L, samples_per_bucket, "arithmetic")
            X_g, y_g = make_pattern_batch(rng, L, samples_per_bucket, "geometric")
            datasets[L] = (np.concatenate([X_a, X_g]), np.concatenate([y_a, y_g]))

        layer = LSTMLayer(input_size=1, hidden_size=hidden_size, rng=rng)
        Wy = rng.randn(hidden_size, 1) * 0.1
        by = np.zeros(1)
        lr = 0.01 if optimizer_name == "adam" else 0.05
        optimizer = Adam(lr=lr) if optimizer_name == "adam" else SGD(lr=lr)

        chart_placeholder = st.empty()
        progress = st.progress(0.0)
        losses = []

        for epoch in range(epochs):
            epoch_loss, num_batches = 0.0, 0
            for L in rng.permutation(lengths):
                x_batch, y_batch = datasets[L]

                h_seq, _, caches = layer.forward(x_batch)
                h_last = h_seq[:, -1, :]
                pred = h_last @ Wy + by

                loss, dpred = mse_loss(pred, y_batch)
                epoch_loss += loss
                num_batches += 1

                dWy = h_last.T @ dpred
                dby = dpred.sum(axis=0)
                dh_last = dpred @ Wy.T

                dh_seq = np.zeros_like(h_seq)
                dh_seq[:, -1, :] = dh_last
                _, _, _, grads = layer.backward(dh_seq, caches)

                optimizer.step(
                    {"Wx": layer.cell.Wx, "Wh": layer.cell.Wh, "b": layer.cell.b},
                    {"Wx": grads["Wx"], "Wh": grads["Wh"], "b": grads["b"]},
                )
                optimizer.step({"Wy": Wy, "by": by}, {"Wy": dWy, "by": dby})

            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            if epoch % max(1, epochs // 30) == 0 or epoch == epochs - 1:
                chart_placeholder.line_chart(losses)
                progress.progress((epoch + 1) / epochs)

        progress.empty()
        st.success(f"Trained. Final training loss: {losses[-1]:.6f}")
        st.session_state["seq_model"] = (layer, Wy, by)

    if "seq_model" not in st.session_state:
        st.info("Click **Train now** first — then a box will appear below to type your own sequence.")
        return

    layer, Wy, by = st.session_state["seq_model"]

    st.markdown("---")
    st.subheader("Now try it yourself")
    st.markdown(f"Type **any number of values** (2 or more — trained on lengths "
                f"{MIN_LEN}–{MAX_LEN}, but any length works), comma-separated. Try an "
                "additive pattern (`2, 4, 6, 8` → always +2) or a multiplicative one "
                "(`2, 4, 8, 16` → always ×2, within roughly a ×0.6–×1.6 ratio per step "
                "for best accuracy).")

    default = "2, 4, 6, 8"
    user_input = st.text_input("Enter your numbers:", value=default)

    if st.button("Predict next number", type="primary"):
        try:
            values = [float(v.strip()) for v in user_input.split(",") if v.strip() != ""]
        except ValueError:
            st.error("Please enter numbers separated by commas, e.g. `1, 3, 5, 7, 9`.")
            return
        if len(values) < 2:
            st.error("Enter at least 2 numbers.")
            return

        scale = max(max(abs(v) for v in values), 1e-3)
        normalized = np.array(values, dtype=float) / scale
        x = normalized[None, :, None]
        h_seq, _, _ = layer.forward(x)
        pred_norm = float((h_seq[:, -1, :] @ Wy + by).item())
        pred = pred_norm * scale

        st.metric("LSTM's predicted next value", f"{pred:.3f}")

        n_vals = len(values)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(range(n_vals), values, marker="o", label="your input")
        ax.plot([n_vals], [pred], marker="*", markersize=16, color="red", label="LSTM prediction")
        ax.legend()
        ax.set_xlabel("position in sequence")
        st.pyplot(fig)

        if not (MIN_LEN <= n_vals <= MAX_LEN):
            st.caption(f"Note: you entered {n_vals} numbers, outside the trained length range "
                       f"({MIN_LEN}–{MAX_LEN}) — it still works, but may be less accurate.")
        st.caption("Try a sequence that's neither additive nor multiplicative — the prediction "
                   "will be less accurate, which is a good talking point about what the model "
                   "actually learned versus what it's just guessing at.")


# ---------------------------------------------------------------- Step 7
def step_tests():
    st.markdown("The implementation is verified by 13 tests: unit tests on activation "
                "functions/init, shape tests, and numerical gradient checks for both the "
                "single cell and the full unrolled sequence (BPTT).")

    st.code("pytest tests/ -v", language="bash")

    if st.button("Run test suite now", type="primary"):
        with st.spinner("Running pytest..."):
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/", "-v", "--no-header"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                capture_output=True,
                text=True,
            )
        st.code(result.stdout[-4000:] or result.stderr[-4000:], language="text")
        if result.returncode == 0:
            st.success("All tests passed. ✅")
        else:
            st.error("Some tests failed.")


PAGES = [
    step_overview,
    step_ops,
    step_cell,
    step_layer,
    step_backward,
    step_loss_optim,
    step_experiment,
    step_tests,
]

PAGES[step]()
