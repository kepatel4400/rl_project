# rl_project
Robot Learning (ROLE) course project

## Dependencies

The pinned environment targets Linux x86-64. Python 3.11 has a prebuilt
PyBullet wheel. On Python 3.12 or 3.13, build PyBullet with NumPy support for
the simulation scripts. The PyTorch build uses CUDA 13. For GPU acceleration,
use a compatible NVIDIA driver from the R580 series or newer.

The simulation scripts also import `urdf_models` for the object meshes. Make
the directory containing that package available on `PYTHONPATH`.
[PyBullet URDF models](https://github.com/GPT-RL/pybullet-URDF-models/tree/0f00cd5f7a4f5a5db893c2141c890f9b2e995431)

## Loading checkpoints

Stable Baselines3 checkpoint archives contain pickle data that can execute code
when loaded. Use `--trust-checkpoint` only for a checkpoint whose source you
trust. When using a YAML configuration, verify the checkpoint path it selects
before passing the flag on the command line.

```bash
bash run_eval.sh path/to/model.zip --trust-checkpoint
bash run_rl_eval_sweep.sh --trust-checkpoint
```

The training scripts accept `--trust-checkpoint` alongside `--load-model` for
resuming from an existing checkpoint. Fresh training needs no flag.

# TODO:
- [ ] Upload my paper
- [ ] Add a few output GIFs
- [ ] Add result table in the paper
