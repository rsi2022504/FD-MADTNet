# FD-MADTNet: Two-Stage Image Deraining

This repository contains the training and inference scripts supplied for the
FD-MADTNet project.

## Repository contents

- `train.py` — two-stage deraining model, training loop, checkpoint saving, and validation.
- `inference.py` — checkpoint loading, image restoration, PSNR/SSIM evaluation, and result export.

## Requirements

- Python 3.9 or later (the supplied code has not been tested across Python versions)
- PyTorch and Torchvision versions compatible with one another
- NumPy
- Pillow
- tqdm
- scikit-image

Install the Python dependencies:

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# Linux/macOS:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install the appropriate PyTorch build for your CUDA version from
https://pytorch.org/get-started/locally/ if GPU acceleration is required.

## Dataset layout

The default training and validation paths expected by `train.py` are:

```text
datasets/
├── Rain13K/
│   ├── input/
│   └── target/
└── Rain100H/
    ├── rainy/
    └── gt/
```

The inference script uses `datasets/Test100/input` and
`datasets/Test100/target` by default. Change the paths in the scripts or set
the environment variables described below.

**Important:** the supplied dataset classes sort the input and target file
lists separately and pair them by position. Ensure that corresponding rainy
and ground-truth images have matching filenames and sort order. The scripts
do not currently verify filename-by-filename correspondence.

Do not commit research datasets unless their licenses explicitly permit
redistribution. Keep datasets outside the repository and use the `.gitignore`
rules provided here.

## Training

From the repository root:

```bash
python train.py
```

Default paths can be overridden with environment variables:

| Variable | Default |
|---|---|
| `TRAIN_RAIN_DIR` | `datasets/Rain13K/input` |
| `TRAIN_GT_DIR` | `datasets/Rain13K/target` |
| `TEST_RAIN_DIR` | `datasets/Rain100H/rainy` |
| `TEST_GT_DIR` | `datasets/Rain100H/gt` |
| `MODEL_SAVE_PATH` | `checkpoints/best_model.pth` |

For example, in PowerShell:

```powershell
$env:TRAIN_RAIN_DIR="D:/datasets/Rain13K/input"
$env:TRAIN_GT_DIR="D:/datasets/Rain13K/target"
$env:TEST_RAIN_DIR="D:/datasets/Rain100H/rainy"
$env:TEST_GT_DIR="D:/datasets/Rain100H/gt"
python train.py
```

Review the `CONFIG` section in `train.py` before running. In particular,
confirm `BATCH_SIZE`, `EPOCHS`, `PATCH_SIZE`, phase boundaries, and loss
weights match the experiment you intend to reproduce.

## Inference

Place the trained checkpoint at `checkpoints/best_model.pth`, or override
`CHECKPOINT_PATH`, then run:

```bash
python inference.py
```

Environment variables:

| Variable | Default |
|---|---|
| `TEST_RAIN_DIR` | `datasets/Test100/input` |
| `TEST_GT_DIR` | `datasets/Test100/target` |
| `CHECKPOINT_PATH` | `checkpoints/best_model.pth` |
| `OUTPUT_DIR` | `outputs/inference` |

The script saves derained images, optional Stage-1 outputs, a per-image
`results.csv`, and a `summary.txt`.

**Evaluation note:** the supplied inference script resizes both rainy and
ground-truth images to 480 × 320 before evaluation. Preserve this behavior
only if it matches the evaluation protocol used for the reported results.
Otherwise, revise the preprocessing consistently and document the protocol.

## Checkpoints and generated files

Model checkpoints, datasets, logs, and inference outputs are excluded from Git
by `.gitignore`. If you intend to publish pretrained weights, host them
separately (for example, using a release or an approved research data
repository) and add the download instructions here.

## Reproducibility and current limitations

- The scripts were syntax-checked during repository preparation, but a full
  training/inference run was not performed in this environment.
- Confirm the PyTorch/Torchvision/CUDA combination on your machine.
- The training file's introductory comments describe an earlier phase schedule;
  the active configuration currently sets `EPOCHS = 60`, `PHASE1_END = 20`,
  and `PHASE2_END = 40`. Verify these values against the exact experiment
  you want to release.
- Confirm the dataset pairing and evaluation resizing protocol before
  presenting results as reproducible.

## Citation

If this code is associated with a paper, replace this section with the final
bibliographic details and BibTeX entry before making the repository public.

## License

No license has been added because the project owner must choose the terms
under which the code may be reused. Add a license file before public release
if you want to grant others explicit reuse rights.
