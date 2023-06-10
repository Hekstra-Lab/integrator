# integrator
Proof of concept profile fitting and integration with neural nets


### Development Installation
With CUDA:
```
conda create -yn torch python
conda activate torch
conda install -yc conda-forge mamba
mamba install -y torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -e .
```

CPU only:
```
conda create -yn torch python
conda activate torch
conda install -yc conda-forge mamba
mamba install -y pytorch torchvision torchaudio cpuonly -c pytorch
pip install -e .
```


### Proof of Concept Example
This tiny example shows that for a single image, the integrator
correlates very well with precognition integration results

```
conda activate torch
cd examples
python integrate.py
```
