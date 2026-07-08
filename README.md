# Cosmic Muon Lifetime Analysis

Python analysis pipeline for reconstructing the lifetime of atmospheric cosmic-ray muons from real scintillation detector waveform data.

---

## Overview

Cosmic-ray muons are continuously produced in the Earth's atmosphere through interactions of high-energy cosmic rays with atmospheric nuclei. A fraction of these muons lose enough kinetic energy to stop inside a scintillation detector before decaying into an electron or positron and neutrinos.

This project reconstructs the time delay between the stopping muon and its subsequent decay using real oscilloscope waveforms recorded in a laboratory setup.

The goal of this repository is not only to estimate the muon lifetime, but also to demonstrate a realistic detector-analysis workflow, including waveform loading, trigger reconstruction, delayed pulse detection, diagnostics and result reporting.

---

## Features

- Automatic oscilloscope CSV parsing
- Trigger reconstruction from coincidence channel
- Dual-polarity pulse detection
- Peak-prominence based event selection
- Early-time veto for prompt electronic features
- Muon lifetime reconstruction
- Lifetime histogram generation
- Detector diagnostic tools
- Pulse statistics
- Analysis summary report

---

## Experimental Setup

The experimental setup consists of two plastic scintillation detectors coupled to photomultiplier tubes.

The recorded oscilloscope channels are:

| Channel | Description |
|---|---|
| CH1 | Analog scintillator waveform |
| CH2 | Coincidence trigger signal |

The coincidence signal defines the trigger time of the event. A delayed pulse in CH1 is then searched for and interpreted as a decay-electron candidate.

The dataset consists of real oscilloscope waveforms acquired during a cosmic muon lifetime measurement.

---

## Detector Algorithm

The reconstruction algorithm proceeds as follows:

1. Load waveform data from CSV files.
2. Determine the trigger time from CH2.
3. Apply an early-time veto to suppress prompt features.
4. Search CH1 for delayed pulse candidates.
5. Detect both positive and negative pulse candidates.
6. Rank candidates using peak prominence.
7. Select the most prominent delayed candidate.
8. Compute the reconstructed lifetime.

The current detector is a prominence-based, dual-polarity detector with an early-time veto.

---

## Project Structure

```text
muon-lifetime-analysis/

├── data/
│   └── raw/
│
├── src/
│   ├── io.py
│   ├── detector.py
│   ├── lifetime.py
│   └── analysis.py
│
├── diagnostics/
│   ├── plot_detector_diagnostics.py
│   ├── pulse_statistics.py
│   ├── summary_report.py
│   ├── mle_lifetime.py
│   └── publication_histogram.py
│
├── results/
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Current Detector Configuration

| Parameter | Value |
|---|---:|
| Trigger channel | CH2 |
| Decay search channel | CH1 |
| Early-time veto | 0.8 μs |
| Pulse search | Dual polarity |
| Selection metric | Peak prominence |
| Maximum accepted lifetime | 10 μs |

---

## Results

Current reconstruction result:

| Quantity | Value |
|---|---:|
| Input waveforms | 180 |
| Reconstructed events | 179 |
| Detection efficiency | 99.4 % |
| Mean reconstructed lifetime | 2.099 μs |
| Standard deviation | 0.765 μs |

The reconstructed mean lifetime is close to the accepted free-muon lifetime of approximately 2.2 μs.

However, this analysis should be interpreted as a detector-development study rather than a precision lifetime measurement.

No detector acceptance correction, efficiency correction, background subtraction or systematic uncertainty estimate has been applied.

---

## Lifetime Histogram

![Muon lifetime histogram](docs/publication_lifetime_hist.png)

---

## Diagnostics

Several diagnostic tools are included in the repository:

| Script | Purpose |
|---|---|
| `plot_detector_diagnostics.py` | Visual inspection of selected pulse candidates |
| `pulse_statistics.py` | Pulse height, width, prominence and polarity statistics |
| `summary_report.py` | Generates a text summary of the analysis |
| `mle_lifetime.py` | Experimental maximum-likelihood style estimator |
| `publication_histogram.py` | Produces a polished histogram for documentation |

These tools were used to understand the detector behaviour and evaluate the reconstruction performance.

---

## Limitations

The present analysis intentionally remains simple.

The following effects are not included:

- detector acceptance corrections
- detector efficiency modelling
- background subtraction
- systematic uncertainty estimation
- full maximum-likelihood lifetime extraction
- pulse-shape template fitting

The reconstructed lifetime should therefore be interpreted as the output of the current detector algorithm, not as a precision measurement of the physical muon lifetime.

---

## Future Work

Possible future improvements include:

- pulse-shape discrimination
- template matching
- adaptive detector thresholds
- background estimation
- detector efficiency studies
- uncertainty propagation
- full statistical lifetime fit
- ROOT/C++ implementation
- large-scale ROOT dataset analysis

---

## Requirements

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the main analysis:

```bash
python main.py
```

Generate the analysis summary:

```bash
python diagnostics/summary_report.py
```

Generate the publication-style histogram:

```bash
python diagnostics/publication_histogram.py
```

---

## Educational Purpose

This repository was developed as a scientific programming and detector-analysis project using real experimental waveform data.

It demonstrates the structure of a small particle-physics analysis pipeline, including data loading, detector development, event reconstruction, diagnostics and result reporting.