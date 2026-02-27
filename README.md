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

Important: This tool provides screening signals, NOT proof. Pro Players might be identified as high cheat probability

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

Final score:

FPS_score = 100 × Σ(weight × severity)

------------------------------------------------------------------------

### Statistical anomaly score

Combines:

-   FPS anomaly component
-   Robust outlier component
-   Heaping component
-   Benford deviation component

Risk_score = 100 × (0.50×FPS + 0.25×Outliers + 0.15×Heaping +
0.10×Benford)

------------------------------------------------------------------------

### Cheat probability estimate

Heuristic probability combining:

-   FPS score
-   anomaly score
-   sample size
-   triggered indicators

Clamped to 0--100%.

------------------------------------------------------------------------

### Estimated model accuracy

Confidence estimate based on:

-   sample size
-   signal strength
-   data quality

This is confidence, not true accuracy.

------------------------------------------------------------------------

### Outlier detection

Uses Median Absolute Deviation (MAD):

z = 0.6745 × (x − median) / MAD

Used to detect extreme match-level anomalies.

------------------------------------------------------------------------

### Heaping detection

Detects excessive rounding patterns such as values ending in 0, 5, or
00.

------------------------------------------------------------------------

### Benford analysis

Applied only when:

-   sample size sufficient
-   digit span sufficient

Used as weak supporting signal.

------------------------------------------------------------------------

### Data sanity checks

Detects scraping errors using:

zero_share_kills zero_share_deaths

High values trigger reliability warning.

------------------------------------------------------------------------

## Installation

Install dependencies:

```bash
pip install pandas playwright
```
Install the Playwright Chromium browser:

```bash
python -m playwright install chromium
```
------------------------------------------------------------------------

## Usage

Basic run:

python Analyser.py --url "TRACKER_URL"

Recommended run:

```bash
python Analyser.py --url "TRACKER_URL" --max-clicks 80 --slow
```

Options:

--max-clicks <number of Load More clicks (default 100)> <br>
--outdir <output directory (default current directory) <br>
--headless <run without browser window(not recommand)>  <br>
--slow <slower scraping (recommand)>

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

This tool does NOT prove cheating. It identifies statistical anomalies
only.

DOM scraping may break if Tracker.gg changes layout.

With limited access to players' records, more data, such as headshot%, gun accuracy, may need to be added in identifyer in the future

------------------------------------------------------------------------

## Workflow

1.  Find the player you would like to check on Tracker.gg and go to the "Sessions" page and copy the URL (it should be "https://tracker.gg/bf6/profile/<user_id_number>/matches")
2.  Run the tool and paste the URL to the commandline
3.  Wait for the script to capture records from Tracker.gg
4.  When the records were fully captured, brower will be automatically closed
5.  Review the report in the terminal 
6.  Review report.json
7.  Inspect suspicious matches in matches.csv
8.  Perform manual verification
