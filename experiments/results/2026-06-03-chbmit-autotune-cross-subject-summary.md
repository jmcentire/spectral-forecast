# CHB-MIT autotune cross-subject summary

Labels are not used for candidate selection. Phase rates are post-hoc overlays on the selected held-out validation run.

| Subject | Files | Val accepted | Delta | z effect | p_ge | p floor | Pos files | Config | Ictal pos rate | Pre | Post | Inter | Top phases I/P/Post/Inter |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| chb01 | 14 | yes | 937.3 | 15.73 | 0.0099 | 0.0099 | 1.00 | 1024/256 t=2.5 min=3 | 15/16=0.938 | 0.278 | 0.259 | 0.136 | 9/9/8/58 |
| chb02 | 14 | yes | 1127.9 | 4.98 | 0.0099 | 0.0099 | 1.00 | 1024/256 t=3.0 min=3 | 5/6=0.833 | 0.216 | 0.319 | 0.049 | 2/6/5/38 |
| chb03 | 14 | yes | 1526.6 | 8.27 | 0.0099 | 0.0099 | 0.86 | 1024/256 t=3.0 min=3 | 10/14=0.714 | 0.369 | 0.560 | 0.423 | 5/6/17/56 |

Use empirical p as a floor-limited bound when null exceedances are zero. The z column is a standardized effect size, not a normal-theory p-value.
