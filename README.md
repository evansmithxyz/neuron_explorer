# Neuron Explorer

A local web app for looking inside a small language model. It runs
[`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) on your own machine, records what
every MLP neuron and mixer head does on a prompt, and lets you turn them up or down and see how
the answer changes.

## What it does

- **Neuron map.** Every MLP neuron in every layer, as a 3D scene or a 2D grid, coloured by mean
  activation, by direct push toward the next token, or by the difference an edit made.
- **Neuron detail.** The tokens a neuron promotes and suppresses, its activation on each word of
  the run, and the passages in `corpus.txt` where it fires hardest.
- **Logit lens.** What the model would say if it stopped after each layer.
- **Pathways.** Traces an output token back through the neurons and heads that pushed it, and on
  to the earlier words those heads were reading.
- **Edits.** Scale a single neuron, a group of neurons, or a mixer head. Edits are saved in
  `edits.json`, can be stored as named presets, and can be checked against a suite of test prompts.
- **Steering vectors.** Run the same prompts under two system prompts, take the difference in the
  residual stream, and add it back at any layer with a strength you choose.
- **Chat features.** Multi-turn conversations, thinking mode, image input, and pretend tools whose
  results you type yourself.
- **Export.** Bake neuron and head edits into a full copy of the model weights. Steering vectors
  are written alongside with a small loader script.

## Requirements

- Python 3.13
- Node.js (only to fetch three.js for the 3D map)
- An NVIDIA GPU is strongly recommended. The model loads at full precision when about 10.5 GB of
  GPU memory is free, with 4-bit weights when about 4.2 GB is free, and on the CPU otherwise.

## Setup

These commands use [uv](https://docs.astral.sh/uv/) and the CUDA 12.8 build of PyTorch. Adjust the
index URL for a different CUDA version, or drop it for a CPU-only install.

```bash
uv venv --python 3.13
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install transformers accelerate bitsandbytes safetensors fastapi uvicorn numpy pillow
npm install
```

The model itself (several GB) is downloaded from Hugging Face the first time the server starts.

## Running

On Windows, double-click `run.bat`. It opens the browser and starts the server.

On any platform:

```bash
.venv/Scripts/python server.py      # Windows
.venv/bin/python server.py          # Linux / macOS
```

Then open <http://127.0.0.1:8765>. The first load takes a minute or two.

## Configuration

Set these environment variables before starting the server:

| Variable | Default | Meaning |
|---|---|---|
| `NE_MODEL` | `Qwen/Qwen3.5-4B` | Hugging Face model id. Other models must share this one's layer layout. |
| `NE_DEVICE` | `auto` | `auto` uses the GPU when one is available; `cpu` forces the CPU. |
| `NE_PRECISION` | `auto` | `full` or `4bit` to override the automatic choice on a GPU. |

Other limits (stored runs, conversation length, image size, lookup size) are constants at the top
of `backend.py`.

## Project layout

| Path | Purpose |
|---|---|
| `server.py` | FastAPI routes: validates requests and calls the backend |
| `backend.py` | The `Explorer` class: model loading, hooks, analysis, export |
| `static/index.html` | The whole UI: markup, styles and script |
| `static/map3d.js` | The three.js 3D neuron map |
| `corpus.txt` | Passages used to find what each neuron responds to (one per line) |
| `edits.json`, `presets/` | The current edit list and saved presets |
| `vectors/` | Saved steering vectors |
| `exports/` | Exported models (not tracked) |
| `lookup.npz`, `lookup.json` | The built neuron lookup (not tracked; rebuild it from the UI) |

## Notes

- The server listens on `127.0.0.1` only and has no authentication. Do not expose it to a network.
- Runs are kept in memory, so they are lost when the server restarts. Edits, presets and vectors
  are saved to disk.
- With 4-bit weights, activations are a close approximation of the full model, not exact.

## License

[MIT](LICENSE)
