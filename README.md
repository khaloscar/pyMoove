# pyMoove

Tools for analysing and predicting cow feeding displacement events using sensor data.

## What is pyMoove?

Modern dairy farming relies on data to monitor aspects such as the health and growth of dairy cows.

When a cow is feeding, it can be displaced by another cow. Understanding these feeding-displacement interactions is important for ensuring that all cows have sufficient access to food.

Current approaches often use cameras to identify displacement events. This project aims to analyse and predict these events using sensor data.

pyMoove is a collection of Python tools developed to support the analysis and prediction of cow feeding displacement events.

## Related project: moo4feed

This project builds upon and makes use of data and concepts related to the moo4feed R package.


## Project structure

```text
pyMoove/
│
├── data/                            # Local directory for raw and processed datasets (git-ignored)
│
├── scripts/
│   ├── clean_rtls_anispace.R
│   └── data_cleaning.py
│
├── src/
│   └── pyMoove/
│       └── __init__.py
│
├── tests/
│   └── test_example.py
│
├── .github/
│   └── workflows/
│       └── test.yml
│
├── README.md
├── pyproject.toml
└── .gitignore

```

### Directory and file descriptions

| File / directory | Description |
|---|---|
| `data/` | Local directory for raw and processed datasets (git-ignored). |
| `src/pyMoove/` | Main Python package containing the library's functionality. |
| `__init__.py` | Defines the package's public interface. |
| `tests/` | Contains automated tests for the library. |
| `.github/workflows/` | Contains GitHub Actions workflows for automated testing. |
| `README.md` | Project overview, documentation and development instructions. |
| `pyproject.toml` | Project metadata, dependencies and build configuration. |
| `.gitignore` | Specifies files and directories Git should ignore. |

**GitHub:** https://github.com/skysheng7/moo4feed

moo4feed is an R package developed by a team at the University of British Columbia. It provides tools for extracting feeding and drinking data collected from livestock farming systems.

**Website:** https://www.skysheng.io/moo4feed/

**Tutorials:** https://www.skysheng.io/moo4feed/articles/index.html

## Development setup

### Requirements

- Python 3.10 or later
- Git

### Setting up the environment

Clone the repository:

```bash
git clone https://github.com/khaloscar/pyMoove
cd pyMoove
```

Create and activate a virtual environment:

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Install the project and development dependencies:

```bash
pip install -e ".[dev]"
```

### Running tests

Run the test suite using pytest:

```bash
pytest
```

