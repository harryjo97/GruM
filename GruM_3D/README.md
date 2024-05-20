# GruM code for 3D Molecule Generation

<p align="middle">
  <img src="../assets/(Fig19-2) GEOM_DRUGS_process.gif" width="80%" />
</p>

## Dependencies

Please refer to the **Dependencies** in [READEME.md](../README.md) of the root directory.

<!-- GruM for 3D molecule generation is built in **Python 3.9.15** and **PyTorch 1.12.1**.
Please run the following commands after installing PyTorch.
```sh
pip install -r requirements.txt
conda install pyg -c pyg
conda install -c conda-forge rdkit=2022.03.2
``` -->

## Running Experiments


### 1. Dataset

QM9 dataset will be downloaded automatically.

For the GEOM-DRUGS dataset, please run the following commands:
```sh
cd data/GEOM-DRUGS/raw/
wget https://dataverse.harvard.edu/api/access/datafile/4360331
tar -xzvf 4360331
```

### 2. Configurations

The configurations are provided on the `config/` directory in `YAML` format. 
Hyperparameters used in the experiments are specified in the Appendix C of our paper.


### 3. Training

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --type train --config <dataset>
```
where ```<dataset>``` is one of the experiment configs in ```config/*.yaml```

### 4. Generation and Evaluation

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --type sample --config <dataset>
```
where ```<dataset>``` is one of the experiment configs in ```config/*.yaml```

## Pretrained checkpoints

We provide checkpoints of the pretrained models in the following links:

+ QM9_3D: [https://drive.google.com/drive/folders/18EUee5Im4Yguv89f128eTrDj-aOz6xuR?usp=sharing](https://drive.google.com/drive/folders/18EUee5Im4Yguv89f128eTrDj-aOz6xuR?usp=sharing)

+ GEOM_DRUGS: [https://drive.google.com/drive/folders/17SU5KIrWzneFsQwb5-q2Gh39Y9cQ2MKJ?usp=sharing](https://drive.google.com/drive/folders/17SU5KIrWzneFsQwb5-q2Gh39Y9cQ2MKJ?usp=sharing) 

To use the checkpoints, please download the model checkpoints (`<dataset>.pth`) and move to the `checkpoints/<dataset>`.

## Generated molecules

We further provide generated molecules of the pretrained models on the `generated_samples/` directory, which are used for the reported results of the paper.

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