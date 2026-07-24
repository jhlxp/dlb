# DLB Incremental Cost

Only the DLB planner and source-pack ON/OFF delta are reported.
Rail transfer, destination forwarding, and combine are excluded.

- Source-forwarded messages: 4000
- Planner CUDA time: 796.61 us

| dtype | pack OFF | pack ON | pack delta | total DLB delta |
|---|---:|---:|---:|---:|
| bfloat16 | 40.51 us | 52.78 us | 12.27 us | 808.88 us |
| fp8 | 32.08 us | 36.32 us | 4.24 us | 800.85 us |
