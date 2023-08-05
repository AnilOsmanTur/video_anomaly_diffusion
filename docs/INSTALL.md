# Installation

## Diffusion model

We used [k-diffusion](https://github.com/crowsonkb/k-diffusion) to implement the diffusion model with some changes.

The required packages for **diffusion model** are in the file `environment.yml`, and you can run the following command to install the environment

```
conda env create -f environment.yml
conda activate diffuser

```

## Autoencoder model

The required packages for **autoencoder model** are in the file `environment_m1.yml`, and you can run the following command to install the environment

```
conda env create -f environment_m1.yml
conda activate anomaly

```



### Note:
- **The above commands are for reference only**, please configure your own environment according to your needs.
