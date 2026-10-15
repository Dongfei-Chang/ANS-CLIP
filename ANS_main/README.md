# How to Run

## Hardware

All experiments were conducted on a single NVIDIA GeForce RTX 5090 GPU.


## How to Install

Create a conda environment and install the required dependencies as follows:

```
# Create a conda environment
conda create -n dassl python=3.10

# Activate the environment
conda activate dassl


# Install the GPU-enabled version of PyTorch.
# For example:
pip install torch==2.9.1+cu128 torchvision==0.24.1+cu128 torchaudio==2.9.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128


```

After that, run `pip install -r requirements.txt` in the `ANS_main/` directory to install the additional packages required by [CLIP](https://github.com/openai/CLIP). Make sure that the `dassl` environment is activated before running this command. Then, you are ready to go.

Follow [DATASETS.md](DATASETS.md) to install the datasets.


## Generalization From Base to New Classes

```bash
cd scripts/
bash base2new.sh
```
