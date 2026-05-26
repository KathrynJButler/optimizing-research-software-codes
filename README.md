# Optimizing Research Software Codes Project

The Optimizing Research Software Codes project aims to improve the performance of an existing research tool used for Global Navigation Satellite System (GNSS) data analysis.  

Specifically, the inherited RINEX2DB Python program converts RINEX observation and navigation files into `.csv` outputs. Our goal is to reduce computation time by 60% while maintaining accuracy.

---

## Team Roster & Contacts

| Name | Role | Contact |
|------|------|----------|
| [Dr. Jihye Park](https://engineering.oregonstate.edu/people/jihye-park) | Project Partner | Jihye.Park@oregonstate.edu |
| Kathryn Butler | Team Member | butlekat@oregonstate.edu |
| Joseph Schaab | Team Member | schaabj@oregonstate.edu |
| Birat Thapa | TA | thapabi@oregonstate.edu |

---

## Repository Overview

| Directory | Description |
|----------------|-------------|
| `.github/workflows/` | Code analysis folder (currently disabled). |
| `Datasets/COVL_1sec/` | Sample input datasets (observation and navigation files). |
| `Testing/` | Testing functions and data go here (currently empty). |
| `documetation/` | Architecture, contributing, and progress reports. |
| `gnss_python-main/` | Main program folder containing `rnx2db.py` and supporting scripts. |

---
## Prerequisites

- **OS:** Windows, macOS, or Linux
- **Python 3.8+** — [python.org/downloads](https://python.org/downloads)
- **Git** (optional) — or download ZIP directly from GitHub
- **Choose one:**
  - **VSCode** (recommended) — [code.visualstudio.com](https://code.visualstudio.com), install the Python extension
  - **Anaconda** (alternative) — [anaconda.com](https://anaconda.com)
- No accounts, API keys, or credentials required!

---

## Installation and Build Steps

**1. Clone the repository**
```bash
git clone https://github.com/KathrynJButler/optimizing-research-software-codes.git
cd optimizing-research-software-codes/gnss_python-main
```

**2. Create and activate a virtual environment**
```bash
python -m venv rnx2db_env

# Windows
rnx2db_env\Scripts\activate

# macOS / Linux
source rnx2db_env/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
# If you see cache errors, run this first:
pip cache purge
```

**4. Add your input files**

Place your files in the correct folders before running:
- `observation/` — observation files (e.g. `NOME1190_rnx2.25O`)
- `navigation/` — navigation files (e.g. `NOME1190.25P`)

> **VSCode users:** Open the project folder and select `rnx2db_env` as your interpreter (Ctrl+Shift+P → "Python: Select Interpreter").  
> **Anaconda users:** See the Anaconda setup section at the bottom of this document.

---

## How to Run the Test Suite

To verify that the optimized output matches the original, run both versions and compare:

```bash
# Windows
fc [original_output.csv] [optimized_output.csv]

# macOS / Linux
diff [original_output.csv] [optimized_output.csv]
```

Our team has tested against multiple datasets of varying sizes to confirm that outputs are identical.

---

## How to Run a Local Development Environment

**1. Activate your virtual environment**
```bash
# Windows
rnx2db_env\Scripts\activate

# macOS / Linux
source rnx2db_env/bin/activate
```

**2. Run the program**
```bash
python rnx2db.py
```

Output will appear in `output/[STATION]/`, named after your input observation file. For example, `output/NOME/NOME1190_rnx2.25O.csv`.

**3. Configure input files**

Edit `config.yaml` to point to your files:
```yaml
rinex:
  singlefile:
    observation:
      - .\observation\[YOURFILE].25O
    navigation:
      - .\navigation\[YOURFILE].25P
  output_dir: .\output
```

The station name and position are read automatically from the observation file header. If the position is not found, the program will warn you and fall back to `config.yaml`:
```yaml
station:
  # if the station position can't be found in the obs header, manually input it here
  info: [NULL, NULL, NULL]
```

---

## How to Deploy

This project runs locally and does not require a server deployment. To transfer to a new machine:

1. Clone or copy the repository to the new machine
2. Follow the installation steps above
3. Copy your `observation/` and `navigation/` data files
4. Run `python rnx2db.py`

---

## Resolving Common Pitfalls

| Problem | Cause | Fix |
|---|---|---|
| `python: command not found` | Virtual environment not activated | Run the `activate` command for your OS before running the script |
| `ModuleNotFoundError` | Dependencies not installed | Run `pip install -r requirements.txt` with the virtual environment active |
| `APPROX POSITION XYZ not found` warning | Position missing from obs file header | Enter coordinates manually in `config.yaml` under `station.info` |
| Output goes to wrong folder | Old output path in config | Update `output_dir` in `config.yaml` to `.\output` or your desired folder |
| `Permission denied` on gfzrnx (macOS/Linux) | Executable bit not set | Run `chmod +x gfzrnx/gfzrnx_*` in the project folder |
| Large file conversion is slow | Version 3 RINEX file being downgraded | Expected. This is a one-time step. Grab a coffee! |
| Warning log fills with messages | Normal operation | Warnings are routed to `warning_log.txt` to keep the console clean |

---

## License
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the [GNU General Public License](https://www.gnu.org/licenses/gpl-3.0.txt) for more details.

---

© 2026 Optimizing Research Software Codes Team
