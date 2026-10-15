# Source training

The helpers implement one optimization step. Supply the data loaders, epoch loop and source-validation routine separately. Targets are not used for training or checkpoint selection.

## Inputs

Each batch contains `detector`, a CPU float tensor shaped B,P,T,17,2 in normalized screen coordinates, and `target`, centre-frame camera-relative 3D supervision shaped B,P,17,3 in millimetres. The model output is in metres. Use T=243 for the paper configuration and the joint order in DATASETS.md. No 2D supervision is required by the two-module loss.

## Handoff

```python
import torch
from figra import FiGRA, make_optimizer, train_step

generator = torch.Generator().manual_seed(100042)
stage1 = FiGRA(backbone, arm="fam", stage="stage1").to(device)
optimizer = make_optimizer(stage1)
for source_batch in source_loader:
    record = train_step(stage1, source_batch, optimizer, generator)

# Run source validation and restore its selected state, not necessarily the last epoch.
torch.save(stage1.adapter_state(), "fam_source_selected.pt")

stage2 = FiGRA(identically_loaded_backbone, stage="stage2").to(device)
state = torch.load("fam_source_selected.pt", map_location=device, weights_only=True)
stage2.fam.load_state_dict(state["fam"], strict=True)
optimizer = make_optimizer(stage2)
for source_batch in source_loader:
    record = train_step(stage2, source_batch, optimizer, generator)
torch.save(stage2.adapter_state(), "figra_adapters.pt")
```

Load only trusted checkpoint files. Save the model dimensions, source-selection settings and upstream-checkpoint identity separately with the experiment. Adapter-only states do not include the upstream weights. Rebuild the optimizer after changing the training stage.

## Objectives and freezing

Stage 1 minimizes root-relative position loss plus 0.5 times the bone-scaled position loss and 0.01 times the mean absolute FAM gate. The FAM query offset and gate consume constant zero vectors, not visibility or cross-person signals.

Stage 2 uses position loss plus 0.5 times bone-scaled loss, 0.05 times the soft 150 mm threshold loss and 0.01 times the coordinate-residual RMS. The helpers retain the original numerical RMS floor. Source perturbations are applied with a CPU generator. AdamW defaults are learning rate 0.0002 and weight decay 0.0001.

The recorded stage-2 protocol uses seeds 42, 123 and 2026, batch size 32, at most 30 epochs, 100,000 source windows per epoch and patience 5. MPI-derived and 3DPW source sampling is 0.7/0.3. Select using the sum of clean and fixed-corrupted source PCK and AUC, subject to clean PCK/AUC being no more than 0.50 percentage points below the frozen-FAM reference and corrupted PCK/AUC improving by at least 0.25 points. The caller must implement the validation and eligibility rule; `train_step` alone does not perform selection.

Synthetic tests verify parameter counts, freezing, gradients, save/restore and person independence. They do not reproduce the manuscript benchmarks. Obtain the original assets and follow the manuscript for the reported experimental settings.
