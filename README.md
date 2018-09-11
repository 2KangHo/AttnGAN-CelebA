# AttnGAN-CelebA

Pytorch implementation for AttnGAN with CelebA dataset.

> **Architecture**
<img src="img/framework.png" width="900px" height="350px"/>


## Dependencies

- python 2.7
- Pytorch
- In addition, please add the project folder to PYTHONPATH and `pip install` the following packages:
  - `python-dateutil`
  - `easydict`
  - `pandas`
  - `torchfile`
  - `nltk`
  - `scikit-image`
  - `pyyaml`


## Data

- Download our preprocessed text for [CelebA](https://drive.google.com/open?id=1N5NLcqjV6IL_ZWwdm2mdZQ6dUyphxnVb) and extract them to `data/CelebA/`
  - file directory example: `data/CelebA/text/0/000012.txt`
- Download the preprocessed [CelebA](https://drive.google.com/open?id=1d_XYCGnXE8AmrKM6Ioo-7hRHhb-Dc04F) image data and extract them to `data/CelebA/`
  - file directory example: `data/CelebA/images/000012.jpg`


## Training

1. Pre-train DAMSM models:
    > For CelebA dataset: `python pretrain_DAMSM.py --cfg cfg/DAMSM/CelebA.yml --gpu 0`

2. Train AttnGAN models:
    > For CelebA dataset: `python main.py --cfg cfg/CelebA_attn2.yml --gpu 0`

- `*.yml` files are example configuration files for training/evaluation our models.


## Sampling

- Run `python main.py --cfg cfg/eval_CelebA.yml --gpu 0` to generate examples from captions in files listed in "./data/CelebA/example_filenames.txt". Results are saved to `DAMSMencoders/`.
- Input your own sentence in "./data/CelebA/example_captions.txt" if you want to generate images from customized sentences.


## Validation

- To generate images for all captions in the validation dataset, change B_VALIDATION to True in the eval_*.yml. and then run `python main.py --cfg cfg/eval_CelebA.yml --gpu 0`


## result - attention map

- 60epoch
<img src="img/attention_maps3300.png" width="685px" height="594px"/>


## Reference

- [AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks](https://arxiv.org/abs/1711.10485v1)
