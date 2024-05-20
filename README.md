# Graph Generation with Diffusion Mixture

Official Code Repository for the paper [Graph Generation with Diffusion Mixture](https://arxiv.org/abs/2302.03596) (ICML 2024).

In this repository, we implement the *Graph Diffusion Mixture* (GruM).

<p align="middle">
  <img src="assets/(Fig16-2) Planar_process.gif" width="45%" />
  <img src="assets/(Fig17-1) SBM_process.gif" width="45%" /> 
</p>

<p align="middle">
  <img src="assets/(Fig19-2) GEOM_DRUGS_process.gif" width="80%" /> 
</p>


## Why GruM?

+ Previous diffusion models cannot accurately model the graph structures as they learn to denoise at each step without considering the topology of the graphs to be generated.

+ To fix such a myopic behavior of previous diffusion models, we propose a new graph generation framework that captures the graph structures by directly predicting the final graph of the diffusion process modeled by a mixture of endpoint-conditioned diffusion processes.

+ Our method significantly outperforms previous graph diffusion models on the generation of diverse real and synthetic graphs, as well as on 2D/3D molecule generation tasks.

## Code structure

We provide <u>two separate projects</u> of GruM for <u>three types of graph generation tasks</u>:

- General graph 
- 2D molecule 
- 3D molecule 

Each projects consists of the following:

```
GruM_2D : Code for general graph generation / 2D molecule generation 
```

```
GruM_3D : Code for 3D molecule generation
```

We provide the details in README.md for each projects.


## Dependencies

Create an environment with **Python 3.9.15** and **Pytorch 1.12.1**. 
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
