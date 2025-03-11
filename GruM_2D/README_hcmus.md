# HCMUS GraphAI

Use [uv](https://docs.astral.sh/uv/getting-started/installation/)

```sh
uv sync

source .venv/bin/activate
```

Prepare

```
# ORCA
cd evaluation/orca 
g++ -O2 -std=c++11 -o orca orca.cpp

# graph-tool
# https://graph-tool.skewed.de/installation.html#native-installation
sudo apt install python3-graph-tool
brew install graph-tool
```

Download then put

- [`data/planar_64_200.pt`](https://drive.google.com/drive/folders/13esonTpioCzUAYBmPyeLSjXlDoemXXQB)
- [`checkpoints/planar/planar.pth`](https://drive.google.com/drive/folders/16lTBrYEqUncuck7k9YDxWuNjTM_PZ4vl?usp=sharing)

Run

```sh
# Prepare data
python3 ./data/data_generators.py --dataset planar --mmd

# Sampling
CUDA_VISIBLE_DEVICES=0 python3 main.py --type sample --config planar
python3 main.py --type sample --config planar
```
