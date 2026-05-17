# Concurrent Prefill-Only Evaluation Report

## Summary
- Strategy count: 4
- Query counts: [8, 16, 32, 64, 103, 112, 119]
- Suspicious batches filtered: 75

## Relative To Baseline 350W
- prefill_manual_buckets: mean energy saving=8.72%, mean TTFT increase=2.71%
- prefill_token_fit: mean energy saving=3.83%, mean TTFT increase=0.96%
- prefill_token_fit_plus25w: mean energy saving=1.76%, mean TTFT increase=-0.05%

## Recommended Prefill Buckets
- q=8, C=225: recommend 193W (strategy=prefill_token_fit, TTFT 3.15%, Energy 5.59%)
- q=16, C=504: recommend 200W (strategy=prefill_manual_buckets, TTFT -0.49%, Energy 9.90%)
- q=32, C=1581: recommend 220W (strategy=prefill_manual_buckets, TTFT 3.79%, Energy 7.73%)
- q=64, C=2175: recommend 220W (strategy=prefill_manual_buckets, TTFT 3.10%, Energy 12.25%)
- q=103, C=6053: recommend 260W (strategy=prefill_manual_buckets, TTFT 2.72%, Energy 6.98%)
- q=112, C=11106: recommend 260W (strategy=prefill_manual_buckets, TTFT 3.95%, Energy 7.92%)
- q=119, C=20295: recommend 260W (strategy=prefill_manual_buckets, TTFT 3.47%, Energy 11.35%)

## Fit Results
{
  "baseline_350w": {
    "avg_ttft_ms": {
      "function": "linear",
      "params": [
        0.08040540946843204,
        250.57387359937508
      ],
      "r2": 0.9927553339193953
    },
    "avg_energy_j": {
      "function": "linear",
      "params": [
        0.046853810635354765,
        22.93955704191304
      ],
      "r2": 0.9988639864382275
    },
    "avg_power_w": {
      "function": "sqrt",
      "params": [
        1.1804965001221035,
        98.53083582461984
      ],
      "r2": 0.9944447179319204
    }
  },
  "prefill_manual_buckets": {
    "avg_ttft_ms": {
      "function": "linear",
      "params": [
        0.08348873247241057,
        255.70946416412704
      ],
      "r2": 0.9924207657021649
    },
    "avg_energy_j": {
      "function": "linear",
      "params": [
        0.04173772226846961,
        24.283460708049052
      ],
      "r2": 0.9997251431890071
    },
    "avg_power_w": {
      "function": "sqrt",
      "params": [
        0.9915953431496064,
        92.88195610628196
      ],
      "r2": 0.987703854357057
    }
  },
  "prefill_token_fit": {
    "avg_ttft_ms": {
      "function": "linear",
      "params": [
        0.08109443837176525,
        252.4671232702255
      ],
      "r2": 0.9933278767394768
    },
    "avg_energy_j": {
      "function": "linear",
      "params": [
        0.04588823594835323,
        21.027613830191207
      ],
      "r2": 0.9991440128745162
    },
    "avg_power_w": {
      "function": "sqrt",
      "params": [
        1.2048767897957053,
        89.74522978130933
      ],
      "r2": 0.9948329629255303
    }
  },
  "prefill_token_fit_plus25w": {
    "avg_ttft_ms": {
      "function": "linear",
      "params": [
        0.08049164149828403,
        250.41315421657222
      ],
      "r2": 0.9925288292125268
    },
    "avg_energy_j": {
      "function": "linear",
      "params": [
        0.04693088866485562,
        20.86047586999866
      ],
      "r2": 0.9988512356720683
    },
    "avg_power_w": {
      "function": "sqrt",
      "params": [
        1.2275727532760448,
        95.71125384231749
      ],
      "r2": 0.9962264113104656
    }
  }
}