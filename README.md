# Graph Generation with Destination-Predicting Diffusion Mixture

Official Code Repository for the paper [Graph Generation with Destination-Predicting Diffusion Mixture](https://arxiv.org/abs/2302.03596).

In this repository, we implement the *Destination-Predicting Diffusion Mixture* (DruM).

<p align="middle">
  <img src="assets/(Fig14-2) Planar_process.gif" width="45%" />
  <img src="assets/(Fig15-1) SBM_process.gif" width="45%" /> 
</p>

<p align="middle">
  <img src="assets/(Fig18-2) GEOM-DRUGS_process.gif" width="80%" /> 
</p>


## Why DruM?

+ Previous diffusion models cannot model the graph topology as they learn to denoise at each step w/o the knowledge of the destination of the generative process.

+ To fix such a myopic behavior, we propose a novel graph generation framework that captures the graph topology by directly predicting the destination of the generative process modeled by a mixture of endpoint-conditioned diffusion processes.

+ Our method significantly outperforms previous graph diffusion models on the generation of diverse real and synthetic graphs, as well as on 2D/3D molecule generation tasks.

## Code structure

We provide <u>two separate projects</u> of DruM for <u>three types of graph generation tasks</u>:

- General graph 
- 2D molecule 
- 3D molecule 

Each projects consists of the following:

```
DruM_2D : Code for general graph generation / 2D molecule generation 
```

```
DruM_3D : Code for 3D molecule generation
```

We provide the details in README.md of each projects.


## Dependencies

Create environment with **Python 3.9.15** and **Pytorch 1.12.1**. 
Use the following command to install the requirements:

```
pip install -r requirements.txt
conda install pyg -c pyg
conda install -c conda-forge graph-tool=2.45
conda install -c conda-forge rdkit=2022.03.2
```


## Citation

If you found the provided code with our paper useful in your work, we kindly request that you cite our work.

```BibTex
@article{jo2022DruM,
  author    = {Jaehyeong Jo and
               Dongki Kim and
               Sung Ju Hwang},
  title     = {Graph Generation with Destination-Driven Diffusion Mixture},
  journal   = {arXiv:2302.03596},
  year      = {2023},
  url       = {https://arxiv.org/abs/2302.03596}
}
```
