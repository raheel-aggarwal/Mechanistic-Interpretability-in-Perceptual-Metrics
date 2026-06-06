# Mechanistic Interpretability in Perceptual Metrics

A lightweight Python toolkit for decomposing an AlexNet CNN into basis-aware
component maps and visualising how convolutional kernels contribute to
intermediate activations.

## Project structure

- `requirements.txt` — Python package dependencies.
- `scripts/` — core Python implementation.
  - `kernel_decomp.py` — basis construction, kernel projection, and component-map generation.
  - `maxpool_decomp.py` — softmax MaxPool decomposition and adaptive α selection.
  - `relu_decomp.py` — exact ReLU decomposition using grounding gates.
  - `pipeline.py` — model utilities, grounding, decomposed forward pass, attribution, output serialization, and visualisation.
  - `main.py` — command-line interface entry point.

## Installation

Create a Python environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run the core pipeline on a single image:

```bash
python scripts/main.py path/to/image.jpg
```

Process every supported image in a folder:

```bash
python scripts/main.py --input-dir path/to/images --output-dir ./results
```

### Common options

- `--output-dir`: root output folder.
- `--basis`: `haar`, `dct`, or `none`.
- `--prune`: keep only the top-N components per channel.
- `--top-k`: number of top components to record and visualise.
- `--path-epsilon`: attribution threshold for small components.
- `--no-save`: skip saving `grounding.pkl`, `components.pkl`, and `attributions.json`.
- `--no-visualise`: skip figure generation.
- `--no-pretrained`: use a randomly initialised AlexNet.

For full flag descriptions, run:

```bash
python scripts/main.py --help
```

## Output layout

Each image produces one output folder containing:

- `grounding.pkl` — saved grounding activations.
- `components.pkl` — decomposed component tensors and metadata.
- `attributions.json` — attribution summaries.
- `stats/` — CSV tables for the top components per layer.
- `heatmaps/` — component attribution heatmaps.
- `dominant_maps/` — dominant component spatial maps per channel.
- `topk_maps/` — per-channel top-K component panels.
- `conv1_rgb_panels/` — optional RGB overlay panels for conv1.
- `summary_dominant_pq.png` — dominant (p,q) scatter across layers.

## How the code is organised

### Pipeline flow

1. `scripts/main.py` parses CLI arguments.
2. `scripts/pipeline.py` runs the full workflow:
   - `preprocess_image` → prepare the input tensor.
   - `collect_grounding_values` → record activations from a standard AlexNet pass.
   - `decomposed_forward` → propagate component maps through conv, ReLU, and MaxPool.
   - `compute_attributions` → rank basis components by mean absolute contribution.
   - `save_outputs` / `run_visualisations` → persist results.

### Modular responsibilities

- `kernel_decomp.py`: basis construction, kernel projection, and component map generation.
- `relu_decomp.py`: exact nonlinear ReLU gate propagation.
- `maxpool_decomp.py`: approximate max pooling through softmax weights.
- `pipeline.py`: orchestration for preprocessing, grounding, decomposed forward pass, attribution, persistence, and visualisation.

## Editing guidance

- Add new model architectures by updating `scripts/pipeline.py` and the
  grounding / decomposed forward logic there.
- Add new visualisations inside `scripts/pipeline.py` using the stored
  `layer_components` structure: `(B, C, H, W, K)`.
- Keep changes local to one module when adding behaviour, and preserve the
  existing pipeline API through `scripts/main.py`.

## Notes

- The code is intentionally built around `torch.Tensor` and CPU-friendly
  output serialization for portability.
- The pipeline is modular so you can replace one stage (e.g. pooling or basis
  choice) without rewriting the whole system.
