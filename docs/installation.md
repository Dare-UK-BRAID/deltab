# Installation

## Requirements

deltab requires Python 3.8 or later and the following packages (installed automatically via pip):

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations, linear algebra (pseudoinverse, SVD) |
| `scikit-learn` | PCA decomposition |
| `scipy` | Kernel density estimation (used in example plots) |
| `matplotlib` | Visualisation (examples only) |
| `pandas` | Tabular output (examples only) |

## Install from PyPI

```bash
pip install deltab
```

## Install from source

```bash
git clone https://github.com/your-org/deltab.git
cd deltab
pip install -e .
```

The `-e` flag installs in editable mode, so local changes to the source are reflected immediately without reinstalling.

## Verify installation

```bash
deltab --version
```

```bash
python -c "from deltab import BrainDelta; print('OK')"
```

## TRE / HPC environments

In TRE environments without internet access, install from a pre-downloaded wheel or from a local clone. If using pip with `--break-system-packages` is required:

```bash
pip install deltab --break-system-packages
```

Or create a virtual environment first:

```bash
python3 -m venv /path/to/venv
source /path/to/venv/bin/activate
pip install deltab
```
