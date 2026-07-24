# DLB Incremental Cost

LB ON covers planner through source pack.
Rail transfer, destination forwarding, and combine are excluded.

- Source-forwarded messages: 4000
- Planner CUDA P50: 3143.26 us
- Save-plan CUDA P50: 31.22 us
- Publish-count CUDA P50: 28.70 us

| dtype | OFF total | ON total | LB increment |
|---|---:|---:|---:|
| bfloat16 | 40.27 us | 3251.30 us | 3211.02 us |
| fp8 | 32.46 us | 3237.71 us | 3205.25 us |

| dtype | pack OFF | pack ON | pack delta |
|---|---:|---:|---:|
| bfloat16 | 40.27 us | 48.11 us | 7.84 us |
| fp8 | 32.46 us | 34.53 us | 2.06 us |
