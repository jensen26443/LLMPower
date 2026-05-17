# Feedforward Evaluation Report

## Summary
- Strategy count: 2
- Query counts: [8, 16, 32, 64, 96, 128]
- Output lengths: [100]

## Relative To Baseline 350W
- ff_decode_tbt_guarded_pid: mean energy saving=10.32%, mean TBT increase=4.02%

## PID Guard Behavior
- ff_decode_tbt_guarded_pid: mean PID updates=1.01, mean prefill delta=0.63 W, mean decode delta=0.01 W, mean feedback TBT=71.39 ms