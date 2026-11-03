# 0004: Tube masking is the only masking strategy with a cross-channel block

Status: accepted, 2026-11-03

## Context

The cross-channel block (Stage B) mixes tokens across channels at each time patch and assumes a complete channel grid at every time patch it sees. Random masking on the time-by-channel patch grid violates that assumption. Two separate implementations have failed on exactly this point and both trained without error: in `fibermind-fm` v0.2, Stage B was skipped during masked training (confirmed: 0 of 12 parameter tensors moved in 20 steps) while still being applied, untrained, at inference; in the June `das_mae.py` on `main`, the block runs on the gathered kept tokens, so token j of channel 1 is mixed with token j of channel 2 even though they come from different time steps, scrambling the moveout it is meant to learn.

## Decision

Tube masking is the only masking strategy permitted when a cross-channel block is present: whole time patches are masked and every channel is kept at the time patches that survive. Config validation refuses any other masking strategy if Stage B is configured. A unit test asserts that every Stage B parameter tensor changes after twenty optimiser steps on synthetic data, for both MAE and JEPA.

## Alternatives considered

- Random patch masking with Stage B made tolerant of partial grids (masked attention over present channels). Rejected: it changes what Stage B computes between training and inference and gives up the structural guarantee that bought the fix in August.
- Drop Stage B. Rejected: moveout and apparent velocity are the physics that distinguish a train from a noise burst, and are the reason the block exists.

## Consequences

The mask ratio applies to time patches, so the effective masking is coarser than random patch masking at the same ratio; the ratio is a config value to be tuned, not assumed from image MAE defaults. Random patch masking remains available for temporal-only ablations. The same constraint applies to the VideoMAE family, which is the precedent.
