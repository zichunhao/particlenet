# `ParticleNet` for Anomaly Detection on the `JetNet` Dataset

## Introduction
This is a training script for training a `ParticleNet` model [[arXiv:1902.08570](https://arxiv.org/abs/1902.08570)] for anomaly detection on the [JetNet](https://jetnet.readthedocs.io/en/latest/pages/contents.html) dataset, used as a baseline model for the [Lorentz Group Equivariant Autoencoder](https://github.com/zichunhao/lgn-autoencoder). The script is based on the code in [Raghav](https://www.raghavkansal.com/)'s repository "[Graph Generative Adversarial Networks for Sparse Data Generation](https://github.com/rkansal47/mnist_graph_gan/tree/master/jets)."


## Data
The default dataset is JetNet. To download and preprocess the data for training, run the following command:
```bash
python3 data/get_data.py
```
Flags:
- `--path-data`: Path to the directory where the data will be stored. Default: `data/particlenet`.
- `--test-size`: Fraction of the data to be used for testing. Default: `0.2`.
- `--bkg-type`: Type of background to use. Default: `['g', 'q']`.
- `--sig-type`: Type of signal to use. Default: `['w', 't', 'z']`.
- `--mask`: Whether to use mask as a feature. Default: `False`.

## Training
To train the model, run the following command:
```bash
python3 main.py \
--path-results 'results' \
--path-data-train-sig 'data/particlenet/sig_train.pt' \
--path-data-train-bkg 'data/particlenet/bkg_train.pt' \
--path-data-test-sig 'data/particlenet/sig_test.pt' \
--path-data-test-bkg 'data/particlenet/bkg_test.pt' 
```
The results will be stored in the `results/` directory. More flags and their meanings can be found in `main.py`.

## References
- Huilin Qu and Loukas Gouskos, "Jet Tagging via Particle Clouds", *Phys. Rev. D*, doi:[10.1103/PhysRevD.101.056019](https://doi.org/10.1103/PhysRevD.101.056019), [arXiv:1902.08570](https://arxiv.org/abs/1902.08570).
- Raghav Kansal, [JetNet](https://github.com/rkansal47/JetNet) Library.
- Raghav Kansal, [Graph Generative Adversarial Networks for Sparse Data Generation](https://github.com/rkansal47/mnist_graph_gan).
- Zichun Hao, [Lorentz Group Equivariant Autoencoder](https://github.com/zichunhao/lgn-autoencoder).
