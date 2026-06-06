# Mechanistic Interpretability in Perceptual Metrics

A Python toolkit for decomposing an AlexNet CNN into basis-aware component maps and visualising how convolutional kernels contribute to intermediate activations. The decomposition is algebraically exact through linear layers and ReLU, and uses a numerically stable softmax approximation at MaxPool — giving an additive, attributable representation of every activation in the network's feature extractor.

Note: This project benefited from implementation assistance and code review suggestions generated using Anthropic's Claude and GitHub Copilot.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Output Layout](#output-layout)
5. [How the Code is Organised](#how-the-code-is-organised)
6. [Concept: Basis-Path Decomposition](#concept-basis-path-decomposition)
7. [Editing Guidance](#editing-guidance)
8. [Notes](#notes)

---

## Project Structure

```
requirements.txt          Python package dependencies
scripts/
├── kernel_decomp.py      basis construction, kernel projection, component-map generation, pruning
├── maxpool_decomp.py     softmax MaxPool decomposition and adaptive α selection
├── relu_decomp.py        exact ReLU decomposition using grounding gates
├── pipeline.py           model loading, grounding, decomposed forward pass, attribution,
│                         output serialisation, and visualisation
└── main.py               command-line interface entry point
```

Each script has a focused responsibility. `pipeline.py` is the orchestrator; the three decomp modules are independently testable — each has a `__main__` self-test block — and can be imported in isolation.

---

## Installation

Create a Python environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate          # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

Key dependencies: `torch`, `torchvision`, `numpy`, `Pillow`, `matplotlib`.

---

## Usage

Run the core pipeline on a single image:

```bash
python scripts/main.py path/to/image.jpg
```

Process every supported image in a folder (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`):

```bash
python scripts/main.py --input-dir path/to/images --output-dir ./results
```

When `--input-dir` is given, each image gets its own subdirectory inside `--output-dir` named after the image stem. With a positional image path, all outputs go directly into `--output-dir`.

### Common options

| Flag | Default | Description |
|---|---|---|
| `--output-dir` / `-o` | `./decomp_outputs` | Root directory for all saved outputs. |
| `--basis` / `-b` | `haar` | Kernel decomposition basis: `haar`, `dct`, or `none`. `none` skips decomposition (K=1 component per channel). |
| `--prune N` | off | After each conv layer, keep only the top-N components per output channel (ranked by mean absolute activation); zero the rest. Reduces memory at the cost of some reconstruction fidelity. |
| `--top-k K` | `20` | Number of top-scoring components recorded per channel in attributions and visualisations. |
| `--path-epsilon EPS` | `1e-6` | Attribution components with mean absolute activation below this threshold are zeroed before recording. |
| `--no-save` | off | Skip writing `grounding.pkl`, `components.pkl`, and `attributions.json`. |
| `--no-visualise` | off | Skip all figure generation. |
| `--no-pretrained` | off | Use a randomly initialised AlexNet instead of ImageNet-pretrained weights. |
| `--image-path PATH` | input image | Path to the original image used for conv1 RGB overlay panels. |
| `--max-channels N` | `96` | Maximum number of channels shown per layer in visualisations. |

**MaxPool α controls.** The softmax sharpness α is estimated automatically per pooling layer; manual overrides are available for reproducibility or ablation:

| Flag | Default | Description |
|---|---|---|
| `--alpha-pool0 A` | auto | Force a fixed α for pool1 (after conv1). |
| `--alpha-pool1 A` | auto | Force a fixed α for pool2 (after conv2). |
| `--alpha-auto-target T` | `30.0` | Target value for α × median\_gap during automatic selection. |
| `--alpha-max A` | `200.0` | Hard ceiling on α (overflow guard). |
| `--alpha-fallback A` | `50.0` | α used when gap estimation fails (near-tied pooling inputs). |

For the full flag list:

```bash
python scripts/main.py --help
```

---

## Output Layout

Each image produces one output folder containing:

```
<output-dir>/
├── grounding.pkl            saved grounding activations (conv outputs, ReLU gates, pool inputs)
├── components.pkl           decomposed component tensors (B, C, H, W, K) per layer + metadata
├── attributions.json        per-channel top-K attribution summaries for all conv layers
├── stats/                   CSV tables for the top components per layer, ranked by mean |z|
├── heatmaps/                component attribution heatmaps per channel
├── dominant_maps/           dominant component spatial maps per output channel
├── topk_maps/               per-channel top-K component panels
├── conv1_rgb_panels/        optional RGB overlay panels for conv1
└── summary_dominant_pq.png  dominant (p,q) scatter across layers
```

**File formats.** `grounding.pkl` and `components.pkl` are standard Python pickles (protocol `HIGHEST_PROTOCOL`). Component tensors inside `components.pkl` are stored as numpy arrays for portability and are restored to `torch.Tensor` when loaded via `load_outputs`. `attributions.json` serialises all numpy arrays as nested Python lists and is human-readable.

**Loading outputs in Python:**

```python
from pipeline import load_outputs

grounding, decomp_results, attributions = load_outputs("./decomp_outputs")

# Component tensor for conv3: shape (B, C, H, W, K)
comp_conv3 = decomp_results["layer_components"]["conv3"]

# Dominant component for channel 0 at conv3
top_flat = attributions["conv3"]["top_k_indices"][0][0]
N = attributions["conv3"]["basis_N"]     # sqrt(K), e.g. 3 for conv3
p, q = divmod(top_flat, N)              # row-major flat index → (p, q)
```

---

## How the Code is Organised

### Pipeline flow

1. `scripts/main.py` parses CLI arguments and calls `run_images`.
2. `scripts/pipeline.py` runs the full workflow for each image via `run_one`:
   - `preprocess_image` — resize to 256, centre-crop to 224, convert to tensor, apply ImageNet normalisation.
   - `collect_grounding_values` — standard AlexNet forward pass; record all intermediate activations needed by the decomposed pass.
   - `decomposed_forward` — propagate basis component maps through conv, ReLU, and MaxPool using the three decomp modules.
   - `compute_attributions` — rank basis components by mean absolute contribution per channel for each conv layer.
   - `save_outputs` / `run_visualisations` — persist results and generate figures.

### Modular responsibilities

**`kernel_decomp.py`** — basis construction, kernel projection, and component-map generation.

- `dct_basis(N)` and `haar_basis(N)` build the 1-D orthonormal basis matrix of shape `(N, N)` where each row is one basis vector. Both satisfy `U @ U.T == I_N`.
- `build_2d_basis(N)` forms the 2-D basis by outer product: `Psi[p, q, x, y] = U[p, x] * U[q, y]`, used for reference; the separable structure means the full `(N, N, N, N)` tensor is never materialised in the hot path.
- `project_kernel(K, U)` projects a `(Co, Ci, N, N)` kernel onto the 1-D basis via two sequential einsum contractions — first across rows, then across columns — returning projection coefficients of shape `(Co, Ci, N, N)`. Using two contractions avoids forming the explicit 2-D outer product and keeps intermediate sizes at `O(N·Co·Ci·N)`.
- `reconstruct_from_coeffs(coeffs, U)` inverts the projection for round-trip verification. Maximum absolute error should be below `1e-5` for both Haar and DCT bases at all supported kernel sizes.
- `basis_component_maps(X, K, U, padding, stride)` computes the full `(B, Co, Ho, Wo, N²)` component tensor for one conv layer. For each `(p, q)` pair it builds `ψ_pq = outer(u_p, u_q)`, convolves `X` with it via grouped depthwise convolution (one group per input channel), then mixes channels using the `(Co, Ci)` projection coefficients to produce one `(B, Co, Ho, Wo)` map.
- `prune_components(x_comp, prune)` scores components by `mean |z|` over `(B, H, W)` independently per output channel, keeps the top-`prune`, and zeros the rest. Scoring per channel (not globally) prevents components that dominate one channel from being kept across all channels at the expense of channel-specific components.

**`relu_decomp.py`** — exact ReLU decomposition via grounding gates.

- `relu_decompose(z_components, Z_original)` multiplies every component map by the binary gate `1[Z_j > 0]`, where the gate is read from the grounding activation `Z_original` (the pre-ReLU output recorded in Pass 1). The decomposition is algebraically exact: summing the gated components over the K dimension recovers `ReLU(Z_j)` exactly. Uses strict inequality, consistent with PyTorch's default ReLU.

**`maxpool_decomp.py`** — softmax-approximated additive MaxPool decomposition.

- `estimate_alpha(p_original, kernel_size, stride, ...)` unfolds `p_original` into pooling windows, computes the median gap between the maximum and second-maximum activation in each window, and sets `α = target_gap_product / median_gap`. It then clamps α downward so that `α × ||p||_inf < 600` (preventing float32 overflow) and upward at `alpha_max`. If the gap is too small to estimate reliably, `alpha_fallback` is used.
- `maxpool_decompose(p_components, p_original, kernel_size, stride, padding, alpha)` decomposes MaxPool additively. It uses `F.unfold` to extract pooling windows from both `p_original` and the merged `(C×K)` component tensor without data copies, computes numerically stable softmax weights per window (log-sum-exp with per-window max subtraction), and contracts the weights against the unfolded components to produce the `(B, C, H_out, W_out, K)` output.

**`pipeline.py`** — orchestration for all remaining stages.

- `load_alexnet(pretrained)` loads AlexNet from torchvision and moves it to the active device (CUDA if available, otherwise CPU) in eval mode.
- `collect_grounding_values(image_tensor, model)` walks `model.features` layer by layer and records: `conv_outputs` (post-conv pre-ReLU activations), `relu_gates` (boolean `Z > 0` tensors), `relu_outputs` (post-ReLU activations), and `pool_inputs` (activations immediately before each MaxPool). AlexNet features index: `0:Conv1 1:ReLU 2:Pool  3:Conv2 4:ReLU 5:Pool  6:Conv3 7:ReLU  8:Conv4 9:ReLU  10:Conv5 11:ReLU 12:Pool`.
- `decomposed_forward(image_tensor, model, grounding, ...)` is Pass 2. It maintains a component tensor `x_comp` of shape `(B, C, H, W, K)`. At each `Conv2d` it calls `basis_component_maps`; for layers after the first conv it accumulates over the `K_prev` incoming components (summing `basis_component_maps` calls — valid by linearity of convolution). At each `ReLU` it calls `relu_decompose` with the matching grounding gate. At each `MaxPool` it calls `estimate_alpha` (or reads a manual override) then `maxpool_decompose`. With `basis_type="none"` the component tensor is set to `grounding["conv_outputs"][i].unsqueeze(-1)` at each conv (K=1), bypassing all basis math while keeping the rest of the pipeline intact.
- `compute_attributions(layer_components, target_layer, ...)` computes `mean |z|` over `(B, H, W)` for every `(channel, component)` pair, zeros entries below `path_epsilon`, and returns the top-K indices and values per channel alongside the flat attribution matrix.
- `compute_decomposition_error(decomp_results, grounding, layer, conv_index)` applies the appropriate ReLU gate to the stored pre-ReLU components, sums over K, and compares against `grounding["relu_outputs"][conv_index]` using normalised L2. A value below 0.01 (1%) indicates high-fidelity reconstruction.
- `save_outputs` / `load_outputs` pickle grounding and components (converting tensors to numpy arrays for portability) and JSON-serialise attributions.
- `run_visualisations` generates all figure types using stored `layer_components`.

---

## Concept: Basis-Path Decomposition

### The core idea

A convolutional layer computes `Z = K * X` (summed over input channels). Any kernel `K^(c)` defined on an `N×N` spatial grid can be expanded exactly in any orthonormal basis `{ψ_i}` of `R^(N×N)`:

```
K^(c) = Σ_i  <K^(c), ψ_i> · ψ_i
```

Substituting into the channel-wise convolution and summing over input channels yields:

```
K * X = Σ_i  z_i          (algebraically exact, no approximation)

z_i  =  Σ_c  <K^(c), ψ_i> · (ψ_i * X^(c))
```

Each `z_i` is a **basis component feature map**: the independent contribution of basis element `ψ_i` to the full conv output. Applying this expansion recursively across all layers gives a **basis-path decomposition** where every output is an additive sum of terms indexed by tuples `(i_1, i_2, …, i_L)` — one basis index chosen per layer. The scalar weight attached to each tuple is the attribution of that basis path to the final output.

### Two-pass execution

Processing one image requires exactly two forward passes.

**Pass 1 — Grounding** (`collect_grounding_values`). A standard, unmodified AlexNet forward pass. Pre-ReLU activations `Z_j`, pre-MaxPool activations `p_i`, ReLU gates `1[Z_j > 0]`, and post-ReLU activations are all recorded and stored. These grounding values drive Pass 2's nonlinear decompositions; reading them from a single clean pass is numerically safer than re-summing many potentially-cancelling component values.

**Pass 2 — Decomposed forward** (`decomposed_forward`). The pipeline carries a component tensor `x_comp` of shape `(B, C, H, W, K)` through the network:

- **Conv2d:** Project the kernel onto the basis, compute `basis_component_maps` to produce the K-component representation of this conv's output. At the first conv the raw image is the input; at subsequent convs each of the `K_prev` incoming components is independently convolved and the results are summed (linearity of convolution).
- **ReLU:** Gate every component by `1[Z_j > 0]` from the grounding pass. Exact.
- **MaxPool:** Contract each component at each pooling window position with `softmax(α·p)_i` from the grounding pass. Approximate.

### Separable 2-D basis

Both Haar and DCT bases are **separable**: `ψ_pq(x,y) = u_p(x) · u_q(y)`. Convolving with `ψ_pq` factors into two sequential 1-D convolutions (row-wise then column-wise), reducing cost from `O(N²·HW)` to `O(2N·HW)` per basis element — a speedup by a factor of `N/2`. `basis_component_maps` exploits this directly; `project_kernel` uses two einsum contractions for the same reason, never materialising the full 2-D outer product.

### ReLU decomposition — exact

For a neuron `j` with pre-activation `Z_j = Σ_i z_ij` (sum of basis components):

```
f(z_ij)  =  z_ij · 1[Z_j > 0]
```

**Correctness:** `Σ_i f(z_ij) = 1[Z_j > 0] · Σ_i z_ij = ReLU(Z_j)` exactly, because `Σ_i z_ij = Z_j` by linearity of the basis expansion. No approximation is introduced at any ReLU layer. If the neuron is inactive (`Z_j ≤ 0`), all components are zeroed. If active, each component passes through unchanged — including components that may individually be negative.

### MaxPool decomposition — approximate

The max of a pooling window can be written as `MaxPool = Σ_i p_i · 1[p_i = p_max]`. The indicator is not additively decomposable, so it is approximated by a softmax:

```
1[p_i = p_max]  ≈  softmax(α·p)_i  =  exp(α·p_i) / Σ_m exp(α·p_m)
```

The approximation error is `O(exp(−α · gap))` where `gap = p_max − second_max`. For `α · gap ≥ 30` the error is below `1e-13`. Substituting the basis expansion `p_i = Σ_j p_ij` gives the additive MaxPool attribution:

```
MaxPool  ≈  Σ_j  ( Σ_i  p_ij · softmax(α·p)_i )
```

The softmax uses the log-sum-exp trick internally (subtracting `max(p)` per window before exponentiating) so float32 does not overflow even for large α. When two pooling inputs are exactly tied, the softmax distributes weight evenly between them — a fractional attribution consistent with the additive decomposition.

**Approximation summary:**

| Operation | Exact or approximate |
|---|---|
| Linear convolution | **Exact** |
| ReLU + grounding gate | **Exact** |
| MaxPool + softmax + grounding | **Approximate** — error `O(exp(−α·gap))` per layer |

### Generalised Haar basis

Performs recursive binary splitting of `[0, N−1]`. Each internal node `[lo, hi]` emits one wavelet vector: `+1` on the left half, `−|L|/|R|` on the right half (then normalised). The scaling vector `u_0 = (1/√N)(1,…,1)` is first. Remaining vectors follow depth-first ordering: parent wavelet before its left subtree, left subtree before right. This ordering is what `haar_basis` in `kernel_decomp.py` produces via `_recurse` and determines the correspondence between flat index `k` and spatial scale.

Haar basis functions are **spatially localised** (non-zero only over a contiguous interval), making them well-suited to the hierarchical, scale-based feature-building intuition of deep CNNs. The `(p,q)=(0,0)` element is the global mean; higher indices correspond to finer-scale horizontal, vertical, or diagonal detail.

### DCT-II basis

```
u_k(n) = α_k · cos(π(2n+1)k / 2N),   α_0 = 1/√N,   α_k = √(2/N)  for k > 0
```

DCT basis functions are **globally supported** (non-zero everywhere) and ordered by spatial frequency. The `(p,q)=(0,0)` element is the DC component; increasing indices correspond to higher spatial frequencies. DCT is preferable when frequency-domain interpretation is desired — for example, studying the network's sensitivity to texture versus shape.

### AlexNet layer geometry and basis sizes

| Layer | Type | Kernel | Pad | Stride | Basis N | K = N² |
|---|---|---|---|---|---|---|
| conv1 | Conv | 11×11 | 2 | 4 | 11 | 121 |
| pool1 | MaxPool | 3×3 | 0 | 2 | — | — |
| conv2 | Conv | 5×5 | 2 | 1 | 5 | 25 |
| pool2 | MaxPool | 3×3 | 0 | 2 | — | — |
| conv3 | Conv | 3×3 | 1 | 1 | 3 | 9 |
| conv4 | Conv | 3×3 | 1 | 1 | 3 | 9 |
| conv5 | Conv | 3×3 | 1 | 1 | 3 | 9 |

### Attribution and the (p,q) index

After the decomposed forward pass each layer stores `(B, C, H, W, K)`. Attribution scores components per output channel as `mean_{b,h,w} |z[b, c, h, w, k]|`. The flat index `k` maps to `(p, q)` in row-major order via `p, q = divmod(k, N)`. In the Haar basis `(p,q)=(0,0)` is the global mean; larger `p` or `q` correspond to finer-scale horizontal or vertical detail. In DCT they correspond to increasing horizontal and vertical spatial frequency.

---

## Editing Guidance

- **Add new model architectures** by updating `scripts/pipeline.py`: extend `collect_grounding_values` to walk the new `model.features` structure recording the same keys (`conv_outputs`, `relu_gates`, `relu_outputs`, `pool_inputs`), and update `decomposed_forward` to handle the new layer sequence and geometry constants.

- **Add new visualisations** inside `scripts/pipeline.py` using the stored `layer_components` structure: `{layer_name: Tensor (B, C, H, W, K)}`. Call new functions from `run_visualisations`. Use `interp_label(p, q, basis_type)` to produce human-readable component labels (`(mean,mean)` / `(s1,s2)` for Haar; `(DC,DC)` / `(f1,f2)` for DCT).

- **Add a new basis** by extending `get_1d_basis` in `kernel_decomp.py` to accept a new `basis_type` string and return a valid `(N, N)` orthonormal matrix. All downstream code (`project_kernel`, `basis_component_maps`, `reconstruct_from_coeffs`) is basis-agnostic and will work unchanged.

- **Keep changes local to one module** when adding behaviour, and preserve the existing pipeline API through `scripts/main.py`.

---

## Notes

- The code is intentionally built around `torch.Tensor` and CPU-friendly output serialisation for portability. CUDA is used automatically when available but is not required.
- The pipeline is modular so you can replace one stage — pooling strategy, basis choice, attribution scoring — without rewriting the rest.
- Each decomp module (`kernel_decomp.py`, `relu_decomp.py`, `maxpool_decomp.py`) has a `_run_tests()` self-test. Run with `python scripts/<module>.py` to verify correctness after any changes.
- Component tensors can be large: at conv1, `(B, C, H, W, K)` with K=121 is 121× larger than the activation tensor. Use `--prune` to reduce K per channel after each conv, or process one image at a time (`B=1`) to manage memory.
- Grounding values are per-image — `collect_grounding_values` must be called once per image before `decomposed_forward`. Together, the `grounding` and `decomp_results` dicts are sufficient to reproduce any intermediate activation or attribution downstream.
