# DLB Incremental Cost

LB ON covers planner through source pack.
Rail transfer, destination forwarding, and combine are excluded.

- Source-forwarded messages: 4000
- Planner CUDA P50: 227.42 us
- Save-plan CUDA P50: 15.54 us
- Publish-count CUDA P50: 14.62 us

| dtype | OFF total | ON total | LB increment |
|---|---:|---:|---:|
| bfloat16 | 39.58 us | 309.46 us | 269.87 us |
| fp8 | 31.54 us | 294.34 us | 262.80 us |

| dtype | pack OFF | pack ON | pack delta |
|---|---:|---:|---:|
| bfloat16 | 39.58 us | 51.87 us | 12.29 us |
| fp8 | 31.54 us | 36.75 us | 5.22 us |
