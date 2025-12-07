# rnx2db Architecture Overview

This document is a simple explanation of what's inside rnx2db.py. It is meant to help new contributors (or anyone unfamiliar with GNSS/RINEX processing) understand the basics of the file quickly.

---

## rnx2db.py Responsibilities

1. Load and normalize the input files
2. Convert files if they’re too large or the wrong version
3. Calculate timing information
4. Compute satellite positions and angles
5. Produce a compact and organized CSV database

---

## Major Components

### Initialization (`__init__`)

* Stores the list of nav/obs files
* Loads configuration settings
* Runs preprocessing on each file
* Loads everything into Pandas DataFrames

This sets up all the data the rest of the process needs.

### File Preprocessing (`_preprocess_obs` and `_preprocess_nav`)

* Checks whether the file is too large to process efficiently
* Checks whether it's in RINEX version 3 format

If both conditions are true, it converts the file to version 2 using the external GFZRNX utility.

### Determining Observation Version (`_get_obs_version`)

A tiny helper that reads the first line of the observation file and returns whether it’s RINEX version 2 or 3. 

### Converting Obs/Nav Files (`_obs_to_version_2` and `_nav_to_version_2`)

Both functions call the GFZRNX executable with the correct flags to produce a version‑2 file.
They handle operating‑system differences and log errors if conversion fails.

### Parsing Everything (`parse`)

The main function.

1. Filters the observation data by constellation (GPS, GLONASS, Galileo, etc.)
2. Adds empty columns for new fields like position, signal strength, Doppler, etc.
3. Computes a standardized timestamp called MJS (Modified Julian Seconds)
4. Computes satellite positions using the navigation data
5. Filters out invalid entries (e.g., missing X/Y/Z)
6. Sorts, organizes, and writes the final CSV

---

## Output

Writes everything into a CSV file whose location is determined by:

* The station configuration
* The year/day of year
* Optionally the hour‑minute index

The output contains the final processed observations, including:

* Satellite ID
* Time
* Position (X, Y, Z)
* Clock correction
* Azimuth/Elevation
* Frequency band
* Doppler
* Carrier phase
* Signal strength