# Stochastic-Discriminator-Augmentation
Simple implementation of Training Generative Adversarial Networks with Limited Data (NeurIPS 2020)

## Usage
### Setup
```
pip install pytorch torchvision numpy tqdm
```

### Train
#### Without Data Augmentation
```
python train.py
```
#### With Conventional Data Augmentation
```
python train.py --mode 1
```
#### With Stochastic Discriminator Augmentation
```
python train.py --mode 2
```

## Experimental Results
<div align="center"><img src="https://github.com/drumpt/Stochastic-Discriminator-Augmentation/blob/main/imgs/result_1.png" width="800"></div>
<div align="center"><img src="https://github.com/drumpt/Stochastic-Discriminator-Augmentation/blob/main/imgs/result_2.png" width="800"></div>
<div align="center"><img src="https://github.com/drumpt/Stochastic-Discriminator-Augmentation/blob/main/imgs/result_3.png" width="800"></div>

## References
- [Training Generative Adversarial Networks with Limited Data (NeurIPS 2020)](https://arxiv.org/pdf/2006.06676.pdf)
