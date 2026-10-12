# FiGRA

Module and source-training interfaces for *FiGRA Feature Geometry Residual Adaptation for Monocular 3D Pose Lifting under Observation Shift*.

FiGRA adapts joint features before a fixed pose-regression head and refines the resulting pelvis-relative coordinates. FAM and KGR add 67,553 and 17,347 parameters, respectively, for a total of 84,900. The demonstrated backbone is GraphMLP. Each person's track is processed independently.

## Setup

Use Python 3.11 or later and install a suitable [PyTorch](https://pytorch.org/get-started/locally/) build, then run

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python examples/smoke.py
```

The tests and smoke example use a synthetic interface fixture. They check software behavior, not pose-estimation accuracy.

## External backbone

Obtain GraphMLP and its 243-frame checkpoint from the [official repository](https://github.com/Vegetebird/GraphMLP). Construct and load that model separately. The expected pre-head interface is defined in the upstream `model/graphmlp.py` file.

```python
from figra import FiGRA, make_optimizer, train_step

model = FiGRA(pretrained_backbone, arm="fam", stage="stage1").to(device)
optimizer = make_optimizer(model)
record = train_step(model, source_batch, optimizer, source_generator)
```

FAM attaches to the input of the external `.head`. Use separate model instances for concurrent inference workers.

## Two-stage adaptation

1. Train `arm="fam", stage="stage1"` and select a FAM checkpoint using source validation.
2. Restore that FAM state into `arm="figra", stage="stage2"` with an identically initialized pretrained backbone.
3. Optimize only KGR. The pretrained backbone and FAM remain fixed.

Stage 1 updates 67,553 parameters and stage 2 updates 17,347. The adaptation budget counts their union, 84,900. `stage="joint"` exposes simultaneous FAM/KGR updates for a continuation control; initialize that control from the same selected FAM checkpoint. It is not a from-scratch joint-training result.

See [TRAINING.md](TRAINING.md) for losses, source selection and checkpoint handoff, and [DATASETS.md](DATASETS.md) for input shapes and official data sources.

## Release scope

This repository provides the FAM and KGR implementation, an external-backbone wrapper, source losses and perturbations, optimization helpers and synthetic tests. It does not include the GraphMLP implementation, comparison-method implementations, datasets, weights, raw predictions or the complete data-preparation and benchmark-evaluation pipeline. Numerical evidence accompanying the manuscript is provided separately in its supplementary material.

No publication venue or acceptance is claimed. Cite the underlying GraphMLP and dataset publications in [REFERENCES.bib](REFERENCES.bib). Dependency attribution is in [THIRD_PARTY.md](THIRD_PARTY.md).

## License

The FiGRA implementation is available under the [MIT license](LICENSE). External code, data and weights retain their original terms.
