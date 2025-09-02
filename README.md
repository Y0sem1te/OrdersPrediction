# OrdersPrediction

This repository contains code for predicting whether incoming orders in the next few hours are sufficient to fill a cabinet at a designated FBA warehouse. The implementation includes a preprocessing pipeline, PyTorch model training, inference utilities, and small experiment runners used during model development.

## Repository layout (relevant parts)

- `ai_peigui/` — main module with preprocessing, training and prediction code.
	- `train.py` — training pipeline (data load → preprocess → train → save weights & scalers).
	- `predict.py` — inference utilities and model definition used for serving/prediction.
	- `preprocess/` — preprocessing helpers and intermediate data (multiple `process*.py` scripts).
	- `state/` — serialized artifacts (feature scalers, fbas mapping).
	- `weights/` — saved model checkpoints (state_dict .pt files).
	- `utils/preview.py` — small script to quickly preview JSON transition files.
	- `al_/` — experimental scripts and diagnostics (compatibility checks, experiment runners, comparison tools and reports).

Other helper files (data examples, preview JSONs) may be present in the project tree.

## Quick start

Prerequisites

- Python 3.8+ with the following (typical) packages:
	- torch
	- numpy, pandas, scikit-learn
	- matplotlib, tqdm

Check `ai_peigui/requirements.txt` (if present) or install manually using pip in your environment.

1) Prepare data

- Place your preprocessed transition files under `ai_peigui/preprocess/transition_data/` or `ai_peigui/transition_data/` depending on your pipeline. Common filenames used by scripts:
	- `output.json`, `output-FBAfilter.json`, `output-FBAfilter-delete.json`, `new_costs.json`.

2) Preview data

Run the preview tool to inspect and extract a small sample:

```powershell
python ai_peigui/utils/preview.py --mode delete
```

Options:
- `--mode` — one of `output, fba, delete, costs` (maps to default filenames)
- `--src` / `--dst` — explicitly set source/destination paths
- `-n` — number of items to include in preview

Preview files are saved under `ai_peigui/preprocess/preview/` by default.

3) Train

The main training entrypoint is `ai_peigui/train.py`. It expects the dataset JSON to be at `ai_peigui/data/data_dropout.json` by default (see `train.py` to change the path). Example:

```powershell
python ai_peigui/train.py
```

Training will:
- extract and save `state/fbas_mapping.pkl` (FBAS name → index mapping)
- save `state/feature_scalers.pkl` (scalers for numeric features)
- save model checkpoints under `weights/model_checkpoints/` and final weights under `weights/final_model.pt`

Notes:
- The training implementation saves model `state_dict` (PyTorch) and uses atomic file replace for critical artifacts.

4) Predict / Inference

Use `ai_peigui/predict.py` to load the saved scalers/mappings and weights and make predictions. Confirm you pass the right `fbas_mapping.pkl` and `feature_scalers.pkl` located under `ai_peigui/state/`.

5) Experiments and diagnostics

- The `ai_peigui/al_/` folder contains experimental runners and utilities the developer used during model iteration:
	- compatibility checker: `al_/check_ckpt_vs_model.py`
	- experiment runners: `al_/exp_A_run.py`, `al_/exp_B_run.py`, `al_/exp_C_run.py`, `al_/exp_D_run.py`
	- comparison helper: `al_/compare_models.py`

Run experiments from the repo root so imports work correctly, for example:

```powershell
python ai_peigui/al_/exp_D_run.py
```

## Recommendations & troubleshooting

- If checkpoint loading fails with missing / unexpected keys, run `ai_peigui/al_/check_ckpt_vs_model.py` to compare checkpoint keys vs the current model.
- Keep `state/` (pickles) and `weights/` (checkpoints) together with the code that expects them — scripts use relative paths under `ai_peigui/`.
- If predictions differ a lot from training metrics, check label distribution and consider running the label-sampling check (script under `al_` can be added) or run 5-fold cross validation.

## Contact / Next steps

If you want, I can:
- add a small CLI wrapper to run end-to-end preprocess → train → predict; or
- generate a Dockerfile / environment spec for reproducible runs.

---

This README is a concise guide to the repository and the scripts found under `ai_peigui/` and `ai_peigui/al_/`. If you want a more detailed developer guide (examples for each preprocessing step, expected JSON schema, or a reproducible experiment manifest), tell me which part to expand and I will add it.
