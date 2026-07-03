# Muon Lifetime Analysis

This project analyzes raw detector data from a muon decay experiment.

## What it does
- Loads waveform data (TIME, CH1, CH2)
- Detects muon events
- Uses coincidence signal (CH2) for filtering
- Extracts muon decay times
- Fits exponential decay to estimate lifetime

## Physics model
Muon decay follows:
N(t) = N0 * exp(-t / τ)

where τ is the muon lifetime.

## Goal
Estimate the muon lifetime from experimental data.
