# DLB Incremental Cost

LB ON covers planner through source pack.
Rail transfer, destination forwarding, and combine are excluded.

- Source-forwarded messages: 4000
- Planner CUDA P50: 196.46 us
- Save-plan CUDA P50: 30.34 us
- Publish-count CUDA P50: 28.37 us

| dtype | OFF total | ON total | LB increment |
|---|---:|---:|---:|
| bfloat16 | 43.81 us | 309.12 us | 265.31 us |
| fp8 | 35.98 us | 292.91 us | 256.93 us |

| dtype | pack OFF | pack ON | pack delta |
|---|---:|---:|---:|
| bfloat16 | 43.81 us | 53.95 us | 10.14 us |
| fp8 | 35.98 us | 37.74 us | 1.76 us |
