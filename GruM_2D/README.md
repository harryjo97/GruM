# GruM code for general graph generation / 2D molecule generation

<p align="middle">
  <img src="../assets/(Fig16-2) Planar_process.gif" width="45%" />
  <img src="../assets/(Fig17-1) SBM_process.gif" width="45%" /> 
</p>

## Dependencies

Please refer to the **Dependencies** in [READEME.md](../README.md) of the root directory.

<!-- Create environment with **Python 3.9.15** and **Pytorch 1.12.1**. Please use the following command to install the requirements:

```
pip install -r requirements.txt
conda install -c conda-forge graph-tool=2.45
conda install -c conda-forge rdkit=2023.03.2
``` -->

## Running Experiments

### 1. Dataset preparations

We provide four **general graph datasets** (Planar, SBM, and Proteins) and two **molecular graph datasets** (QM9 and ZINC250k). 

Download the datasets from the following links and <u>move the dataset to `data` directory</u>:

+ Planar (`planar_64_200.pt`): [https://drive.google.com/drive/folders/13esonTpioCzUAYBmPyeLSjXlDoemXXQB?usp=sharing](https://drive.google.com/drive/folders/13esonTpioCzUAYBmPyeLSjXlDoemXXQB?usp=sharing)

+ SBM (`sbm_200.pt`): [https://drive.google.com/drive/folders/1imzwi4a0cpVvE_Vyiwl7JCtkr13hv9Da?usp=sharing](https://drive.google.com/drive/folders/1imzwi4a0cpVvE_Vyiwl7JCtkr13hv9Da?usp=sharing)

+ Proteins (`proteins_100_500.pt`): [https://drive.google.com/drive/folders/1IawmycfhX49IlGZC5l5A58CJTxIY1iFr?usp=sharing](https://drive.google.com/drive/folders/1IawmycfhX49IlGZC5l5A58CJTxIY1iFr?usp=sharing).

We provide the commands for generating general graph datasets as follows:

```
python data/data_generators.py --dataset <dataset> --mmd
```
where `<dataset>` is one of the general graph datasets: `planar`, `sbm`, and `proteins`.
This will create the `<dataset>.pkl` file in the `data` directory.


To <u>preprocess the molecular graph datasets</u> for training models, run the following command:

```
python data/preprocess.py --dataset <dataset>
python data/preprocess_for_nspdk.py --dataset <dataset>
```
where ```<dataset>``` is one of the 2d molecule datasets: ```qm9``` or ```zinc250k```.

For the evaluation of general graph generation tasks, run the following command to <u>compile the ORCA program</u> (see http://www.biolab.si/supp/orca/orca.html):

```
cd evaluation/orca 
g++ -O2 -std=c++11 -o orca orca.cpp
```

### 2. Configurations

The configurations are provided in the `config/` directory in `YAML` format. 
Hyperparameters used in the experiments are specified in the Appendix C of our paper.


### 3. Training

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --type train --config <dataset> --seed 42
```
where ```<dataset>``` is one of the experiment configs in ```config/*.yaml```

### 4. Generation and evaluation

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --type sample --config <dataset>
```
where ```<dataset>``` is one of the experiment configs in ```config/*.yaml```

## Pretrained checkpoints

We provide checkpoints of the pretrained models in the following links:

+ Planar: [https://drive.google.com/drive/folders/16lTBrYEqUncuck7k9YDxWuNjTM_PZ4vl?usp=sharing](https://drive.google.com/drive/folders/16lTBrYEqUncuck7k9YDxWuNjTM_PZ4vl?usp=sharing)

+ SBM: [https://drive.google.com/drive/folders/1XXcmcexRgGou-DPrbs8LgGWUUAdnnu34?usp=sharing](https://drive.google.com/drive/folders/1XXcmcexRgGou-DPrbs8LgGWUUAdnnu34?usp=sharing)

+ Proteins: [https://drive.google.com/drive/folders/10N_tp4E0sXLqnNbSupg1cOFurs67vCz0?usp=sharing](https://drive.google.com/drive/folders/10N_tp4E0sXLqnNbSupg1cOFurs67vCz0?usp=sharing)

+ QM9: [https://drive.google.com/drive/folders/1RokFFheV648c23KWt3tngh_ZFO3uYe1-?usp=sharing](https://drive.google.com/drive/folders/1RokFFheV648c23KWt3tngh_ZFO3uYe1-?usp=sharing)

+ ZINC250k: [https://drive.google.com/drive/folders/1-W0z3xQEz9To_ewJtLutEjU4SAvDZnAn?usp=sharing](https://drive.google.com/drive/folders/1-W0z3xQEz9To_ewJtLutEjU4SAvDZnAn?usp=sharing)

To use the checkpoints, please download the model checkpoints (`<dataset>.pth`) and move to the `checkpoints/<dataset>`.

## Generated graphs and molecules

We provide generated graphs or molecules of the pretrained models in the `generated_graphs/` directory, which are used for the reported results of the paper.

## Citation

If you found the provided code with our paper useful in your work, we kindly request that you cite our work.

```BibTex
@article{jo2024GruM,
  author    = {Jaehyeong Jo and
               Dongki Kim and
               Sung Ju Hwang},
  title     = {Graph Generation with Diffusion Mixture},
  journal   = {arXiv:2302.03596},
  year      = {2024},
  url       = {https://arxiv.org/abs/2302.03596}
}
```