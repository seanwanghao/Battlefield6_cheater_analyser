# BF6 Anti-Cheat Forensic Screener

## Project Description

BF6 Anti-Cheat Forensic Screener is a statistical analysis tool designed
to collect Battlefield 6 match history from Tracker.gg and identify
anomalous performance patterns using multiple forensic algorithms. It
automates match extraction, computes risk indicators, and produces
structured forensic reports to assist manual review. The tool does not
prove cheating but helps prioritize accounts and matches that exhibit
statistically unusual behavior.

------------------------------------------------------------------------

## Overview

This project uses browser automation and statistical analysis to screen
Battlefield 6 player match history for anomaly signals commonly
associated with suspicious performance patterns.

The main script file is:

Analyser.py

Outputs:

-   matches.csv --- extracted match-level stats
-   report.json --- anomaly scores, probability estimates, and
    diagnostics
-   meta.json --- scraping metadata including extracted player_id

Important: This tool provides screening signals, NOT proof.

------------------------------------------------------------------------

## How it works

Pipeline:

1.  Playwright launches Chromium browser
2.  Opens Tracker.gg match history page
3.  Extracts player_id from document.title
4.  Auto Clicks Load More repeatedly
5.  Extracts match stats from DOM
6.  Classifies matches into Battle Royale and Multiplayer
7.  Computes statistical anomaly signals
8.  Outputs forensic reports

------------------------------------------------------------------------

## Algorithms

### FPS cheat-risk score

Measures frequency of extreme performance:

-   KD ≥ 10 share
-   KD ≥ 20 share
-   KPM ≥ 3 share
-   KPM ≥ 4 share
-   Kills ≥ 60 share
-   Kills ≥ 80 share

Score computed using weighted severity mapping.

------------------------------------------------------------------------

### Statistical anomaly score

Combines:

-   FPS anomaly component
-   Robust outlier component
-   Heaping component
-   Benford deviation component

------------------------------------------------------------------------

### Cheat probability estimate

Heuristic probability combining:

-   FPS score
-   anomaly score
-   sample size
-   triggered indicators

------------------------------------------------------------------------

### Estimated model accuracy

Confidence estimate based on:

-   sample size
-   signal strength
-   data quality

------------------------------------------------------------------------

### Outlier detection

Uses Median Absolute Deviation (MAD):

z = 0.6745 × (x − median) / MAD

------------------------------------------------------------------------

### Heaping detection

Detects abnormal rounding patterns in values.

------------------------------------------------------------------------

### Benford analysis

Checks digit distribution when statistically valid.

------------------------------------------------------------------------

## Installation

Install dependencies:

pip install pandas playwright python -m playwright install chromium

------------------------------------------------------------------------

## Usage

Basic run:

python Analyser.py --url "TRACKER_URL"

Recommended run:

python Analyser.py --url "TRACKER_URL" --max-clicks 80 --slow --headless

Options:

--max-clicks number of Load More clicks --outdir output directory
--headless run without browser window --slow slower scraping

------------------------------------------------------------------------

## Output files

matches.csv Match-level data

report.json Full analysis report

meta.json Scraping metadata

------------------------------------------------------------------------

## Interpretation

Higher scores indicate stronger anomaly signals.

This tool assists investigation but does not confirm cheating.

------------------------------------------------------------------------

## Limitations

This tool detects statistical anomalies only.

Tracker.gg layout changes may require script updates.

------------------------------------------------------------------------

## Workflow

1.  Run tool
2.  Wait for the script caputre records from Tracker.gg
3.  When the records was fully captured, brower will be automatically closed
4.  Review report in the terminal 
5.  Review report.json
6.  Inspect suspicious matches in matches.csv
7.  Perform manual verification
