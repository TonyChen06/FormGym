# FieldFinder training experiments

Minimal changes to [FormGym](https://github.com/mtoles/FormGym) for three treatments: balanced coordinate cross-entropy, answer-region spatial supervision, and their combination. Balanced CE assigns half the loss to coordinate tokens and half to other tokens (including padding). The spatial head is training-only; inference is unchanged.

## Run

Install `requirements.txt` in a CUDA environment. Place the processed FUNSD, XFUND, and FormNLU data at the paths in `tool/config.yaml` (data is not bundled). Run from the repository root:

```bash
# Balanced CE
torchrun --standalone --nproc_per_node=4 -m tool.train_florence --epochs 6 --epochs_per_eval 6 --balanced_ce --note balanced_ce

# Spatial supervision
torchrun --standalone --nproc_per_node=4 -m tool.train_florence --epochs 6 --epochs_per_eval 6 --spatial_loss_weight 0.3 --note spatial

# Both
torchrun --standalone --nproc_per_node=4 -m tool.train_florence --epochs 6 --epochs_per_eval 6 --balanced_ce --spatial_loss_weight 0.3 --note combined
```

Each run starts from `microsoft/Florence-2-large-ft`, uses global batch size 8 across four GPUs, and evaluates the full configured validation sets after six epochs. Use fresh training, without PEFT or checkpoint resumption. For a control, omit both loss flags.

Checkpoints are saved under `tool/checkpoints/`. Copy `validation_metrics.json` and `validation_predictions.jsonl` before the next run, since these two files are overwritten.
