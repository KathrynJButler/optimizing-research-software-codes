# Optimizing Research Software Codes Project

The Optimizing Research Software Codes project aims to improve the performance of an existing research tool used for Global Navigation Satellite System (GNSS) data analysis.  

Specifically, the inherited RINEX2DB Python program converts RINEX observation and navigation files into `.csv` outputs. Our goal is to reduce computation time by 60% while maintaining accuracy.

---

## Team Roster & Contacts

| Name | Role | Contact |
|------|------|----------|
| [Dr. Jihye Park](https://engineering.oregonstate.edu/people/jihye-park) | Project Partner | Jihye.Park@oregonstate.edu |
| Kathryn Butler | Team Member | butlekat@oregonstate.edu |
| Michael McAllister | Team Member | mcallmic@oregonstate.edu |
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

## Setup & Installation
[Current Setup and Run Instructions](https://github.com/KathrynJButler/optimizing-research-software-codes/blob/main/documentation/How_to_Run.pdf)

This process will be updated after further progress is made to reduce runtime.

---

## Execution & Testing
Guidelines for executing performance benchmarks and validation tests will be included soon!

---

## License
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the [GNU General Public License](https://www.gnu.org/licenses/gpl-3.0.txt) for more details.

---

## Version
**v0.1 (Setup Verified)** — Repository cloned, build confirmed, baseline runtime recorded.

**v1.1 (First Runtime Optimization Complete)** — Removed unnecessary loop and improved the runtime of an average of 10%.

---

© 2026 Optimizing Research Software Codes Team
