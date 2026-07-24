# DLB Incremental Cost

Only the DLB planner and source-pack ON/OFF delta are reported.
Rail transfer, destination forwarding, and combine are excluded.

- Source-forwarded messages: 4000
- Planner CUDA P50: 230.67 us

| dtype | pack OFF | pack ON | pack delta | total DLB delta |
|---|---:|---:|---:|---:|
| bfloat16 | 40.38 us | 51.09 us | 10.70 us | 241.38 us |
| fp8 | 32.75 us | 37.23 us | 4.48 us | 235.15 us |
