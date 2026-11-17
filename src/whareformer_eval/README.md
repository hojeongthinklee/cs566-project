# Whareformer evaluation code (unmodified copy)

`evaluation.py` (from `src/utils/evaluation.py`) and `evaluate.py` (from `src/scripts/evaluate.py`) are copied unchanged
from the official Whareformer repository and are covered by its `LICENSE` (included here). They implement the OSNOM
benchmark metrics used for every number we report: mPCL (dense evaluation on EPIC-KITCHENS, sparse key frames on HD-EPIC)
and IDF1.

GraphMem's scorer wrapper (`src/analysis/milestones/m3/wf_common.py`) imports these functions from a Whareformer
checkout placed next to the code (`src/analysis/milestones/Whareformer`), which is also where the benchmark's released
observations live.
