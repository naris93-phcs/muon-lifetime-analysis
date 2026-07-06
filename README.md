# Muon Lifetime Analysis

This project analyzes cosmic ray muon decay events and extracts the muon lifetime.

## Physics

The muon decay follows an exponential law:

N(t) = N0 exp(-t / τ)

where τ is the muon lifetime.

## Method

- Load waveform data from oscilloscope CSV files
- Detect trigger and decay signals
- Compute time difference per event
- Build lifetime distribution
- Extract mean lifetime

## Result

Typical result:
τ ≈ 2.2 μs

## Requirements

pip install -r requirements.txt

## Run

python main.py
python main.py --debug