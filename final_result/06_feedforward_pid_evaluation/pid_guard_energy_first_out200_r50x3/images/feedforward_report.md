# Feedforward Evaluation Report

## Summary
- Strategy count: 2
- Query counts: [8, 16, 32, 64, 96, 128]
- Output lengths: [200]

## Relative To Baseline 350W
- ff_decode_tbt_guarded_pid: mean energy saving=11.41%, mean TBT increase=3.94%

## PID Guard Behavior
- ff_decode_tbt_guarded_pid: mean PID updates=2.15, mean prefill delta=0.90 W, mean decode delta=0.07 W, mean feedback TBT=71.45 ms