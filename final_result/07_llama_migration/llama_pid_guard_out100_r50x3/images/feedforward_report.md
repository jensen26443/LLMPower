# Feedforward Evaluation Report

## Summary
- Strategy count: 2
- Query counts: [8, 16, 32, 64, 96, 128]
- Output lengths: [100]

## Relative To Baseline 350W
- ff_decode_tbt_guarded_pid: mean energy saving=13.28%, mean TBT increase=5.34%

## PID Guard Behavior
- ff_decode_tbt_guarded_pid: mean PID updates=1.00, mean prefill delta=0.57 W, mean decode delta=0.00 W, mean feedback TBT=69.18 ms