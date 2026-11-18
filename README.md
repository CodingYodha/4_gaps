# 4_gaps

Experiment code for my first-author paper on destructive rank collapse: **[arXiv:2610.24571](https://arxiv.org/abs/2610.24571)**. The paper comes from my research internship at IIT Roorkee.

## Problem

Very deep networks without residual connections are hard to train. The paper asks why residual connections are needed, by tracking how the rank (geometry) of hidden representations changes during training, with and without residual connections.

## How it works

The notebooks train many models while logging how their hidden representations change over training. The same kind of experiment is repeated across architectures, datasets and controls.

| Group | Notebooks | What they test |
|---|---|---|
| MLPs | `cifar-mlp-full`, `death-collapse-mlp-resnet-batchnorm`, `mlp-norm-resnet-cure-run` | Plain MLPs vs residual and normalized versions |
| CNNs / ResNets | `cnn-resnet-drc`, `cnn-drc-resnet66` | Convolutional networks with and without residual connections |
| Vision Transformers | `vit-drc`, `vit-drc-cifar-full` | The same question for ViTs |
| Language models | `nano-gpt-drc`, `with-residual-wikitext103-drc-attention` | A small GPT and attention models on WikiText-103 |
| Signal-death runs | `dead-signal-cifar`, `dead-signal-scout-run` | When the signal through the network dies out |
| Fix ("cure") runs | `cure_run_CIFAR_EXP2`, `cure_run_MNIST_EXP2_fixed_NaN_Metrics` | Testing fixes on CIFAR-10 and MNIST |
| Controls | `destructuve-collapse-synthetic-init-optimizer-control-exp` | Synthetic data with different initializations and optimizers |
| Early experiments | `gap-1`, `gap-1-v2`, `gap-1-cifar`, `gap_1_bottleneck/` | First versions of the setup |
| Plots | `cifar-mini-plot-run` | Plotting runs |

## Results

The full results and figures are in the paper: [arXiv:2610.24571](https://arxiv.org/abs/2610.24571).

## How to run

The notebooks need a GPU. To run one locally:

```bash
pip install torch torchvision numpy matplotlib jupyter
jupyter notebook cnn-resnet-drc.ipynb
```

`gap_1_bottleneck/` has its own `requirements.txt`:

```bash
pip install -r gap_1_bottleneck/requirements.txt
```
