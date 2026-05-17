# Feedforward Evaluation Report

## Summary
- Strategy count: 2
- Query counts: [8, 16, 32, 64, 96, 128]
- Output lengths: [200]

## Relative To Baseline 350W
- ff_decode_tbt_guarded_pid: mean energy saving=11.43%, mean TBT increase=3.94%

## PID Guard Behavior
- ff_decode_tbt_guarded_pid: mean PID updates=2.07, mean prefill delta=0.69 W, mean decode delta=0.12 W, mean feedback TBT=72.32 ms