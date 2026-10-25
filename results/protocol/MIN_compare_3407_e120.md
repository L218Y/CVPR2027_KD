| arm | official mIoU | foreground mIoU | day mIoU | night mIoU | night person | guardrail | d mIoU vs ref [95% CI] |
|---|---|---|---|---|---|---|---|
| ce | 55.32 | 50.09 | 54.83 | 51.86 | 45.427 | 29.275 | +0.00 [+0.00, +0.00] |
| cbce | 56.30 | 51.28 | 53.66 | 53.40 | 42.9 | 33.852 | +0.96 [-0.52, +2.50] |
| uniform_kd | 55.55 | 50.34 | 56.16 | 51.33 | 45.663 | 29.67 | +0.19 [-0.72, +1.05] |
| cbce_kd | 56.73 | 51.71 | 55.57 | 53.32 | 45.384 | 25.729 | +1.46 [+0.09, +3.02]* |
| correctness_kd | 57.32 | 52.41 | 55.27 | 53.83 | 41.079 | 31.403 | +1.94 [+0.04, +3.57]* |

* significant at the 95% level under an image-level paired bootstrap.
