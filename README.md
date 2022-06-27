# PyTorch Implementation of Differentiable SDE Solvers ![Python package](https://github.com/google-research/torchsde/workflows/Python%20package/badge.svg?branch=dev)
This library provides [stochastic differential equation (SDE)](https://en.wikipedia.org/wiki/Stochastic_differential_equation) solvers with GPU support and efficient backpropagation.

---
<p align="center">
  <img width="600" height="450" src="./assets/latent_sde.gif">
</p>

## Installation
```shell script
pip install torchsde
```

**Requirements:** Python >=3.6 and PyTorch >=1.6.0.

## Documentation
Available [here](./DOCUMENTATION.md).

## Examples
### Quick example
```python
import torch
import torchsde

batch_size, state_size, brownian_size = 32, 3, 2
t_size = 20

class SDE(torch.nn.Module):
    noise_type = 'general'
    sde_type = 'ito'

    def __init__(self):
        super().__init__()
        self.mu = torch.nn.Linear(state_size, 
                                  state_size)
        self.sigma = torch.nn.Linear(state_size, 
                                     state_size * brownian_size)

    # Drift
    def f(self, t, y):
        return self.mu(y)  # shape (batch_size, state_size)

    # Diffusion
    def g(self, t, y):
        return self.sigma(y).view(batch_size, 
                                  state_size, 
                                  brownian_size)

sde = SDE()
y0 = torch.full((batch_size, state_size), 0.1)
ts = torch.linspace(0, 1, t_size)
# Initial state y0, the SDE is solved over the interval [ts[0], ts[-1]].
# ys will have shape (t_size, batch_size, state_size)
ys = torchsde.sdeint(sde, y0, ts)
```


### Latent SDE

[`examples/latent_sde.py`](examples/latent_sde.py) learns a *latent stochastic differential equation*, as in Section 5 of [\[1\]](https://arxiv.org/pdf/2001.01328.pdf).
The example fits an SDE to data, whilst regularizing it to be like an [Ornstein-Uhlenbeck](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process) prior process.
The model can be loosely viewed as a [variational autoencoder](https://en.wikipedia.org/wiki/Autoencoder#Variational_autoencoder_(VAE)) with its prior and approximate posterior being SDEs. This example can be run via
```shell script
python -m examples.latent_sde --train-dir <TRAIN_DIR>
```
The program outputs figures to the path specified by `<TRAIN_DIR>`.

Some notes for the code:
1. the encoder maps the trajectory information xs into the latent vector.
2. Solve the latent dynamics to get zs and project zs into xs_.
3. Define the maximum likelihood loss to learn the generative model.

