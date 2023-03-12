# Efficient Personalized Speech Enhancement through Self-Supervised Learning

[Aswin Sivaraman](https://actuallyaswin.github.io/) and [Minje Kim](https://saige.sice.indiana.edu) (Indiana University)

<!-- [![Demo](https://img.shields.io/badge/Web-Demo-blue)]() -->
[![Paper](https://img.shields.io/badge/Web-Paper-blue)](https://ieeexplore.ieee.org/abstract/document/9794565/)

## Abstract
This work presents self-supervised learning (SSL) methods for developing monaural speaker-specific (i.e., personalized) speech enhancement (SE) models. While generalist models must broadly address many speakers, specialist models can adapt their enhancement function towards a particular speaker's voice, expecting to solve a narrower problem. Hence, specialists are capable of achieving more optimal performance in addition to reducing computational complexity. However, naive personalization methods can require clean speech from the target user, which is inconvenient to acquire, e.g., due to subpar recording conditions. To this end, we pose personalization as either a zero-shot learning (ZSL) task, in which no additional clean speech of the target speaker is used for training, or a few-shot learning (FSL) task, in which the goal is to minimize the duration of the clean speech used for transfer learning. With this paper, we propose self-supervised learning methods as a solution to both zero- and few-shot personalization tasks. The proposed methods are designed to learn the personalized speech features from unlabeled data (i.e., in-the-wild noisy recordings from the target user) without knowing the corresponding clean sources. Our experiments investigate three different self-supervised learning mechanisms. The results show that self-supervised models achieve zero-shot and few-shot personalization using fewer model parameters and less clean data from the target user, achieving the data efficiency and model compression goals.

<figure>
    <img src="docs/images/pse_ssl_overview.png"
         alt="Overview of Self-Supervised PSE Methods">
    <figcaption><small><em>An overview of the baseline and proposed personalization methods. With the baseline, the SE model is first pretrained using a large speaker-agnostic dataset as a generalist and then finetuned using clean speech signals of the test user. This method relies entirely on the finetuning process for personalization. On the other hand, the proposed methods provide various SSL options to pretrain the model using noisy, but speaker-specific speech, which serve a better initialization point for the subsequent finetuning process, leading to better SE performance. The pretrained models can also conduct a certain level of SE as a ZSL model, while the FSL-based finetuning tends to improve the pretrained model.</small></em></figcaption>
</figure>


## Proposed Methods

### Pseudo Speech Enhancement (PseudoSE)
<img src="docs/images/waveforms_pseudose.png" alt="Pseudo Speech Enhancement" width="400px">

### Contrastive Mixtures (CM)
<img src="docs/images/waveforms_cm.png" alt="Contrastive Mixtures" width="400px">

### Data Purification (DP)
<img src="docs/images/dp_overview.png" alt="Data Purification">

Note that DP may be applied onto the loss functions of either PseudoSE or CM.

## Installation & Usage

Use pip to install the necessary Python packages (e.g., [pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/stable/), [ray[tune]](https://docs.ray.io/en/latest/tune/), and [asteroid](https://asteroid-team.github.io/)).

```
pip install -r requirements.txt
```

Additionally, the following datasets must be downloaded and un-zipped:
+ [Librispeech](http://www.openslr.org/12/)
+ [FSD50K](https://zenodo.org/record/4060432)
+ [MUSAN](http://www.openslr.org/17/)

We define a _generalist_ model as one which is speaker-agnostic; it is trained to enhance the voices of many different speakers. A _specialist_ model is one which is trained to enhance the voice of a single speaker. In this experiment, we train specialist models entirely using degraded recordings (sampling a single speaker from **Librispeech** and unseen noises from **FSD50K**).

To train generalist models, first modify `code/conf_generalists.yaml` with the correct folder paths, then run:
```
python code/train_generalists.py
```
Similarly, to train specialist models, first modify `code/conf_specialists.yaml` with the correct folder paths, then run:
```
python code/train_specialists.py
```

Each YAML configuration file defines the experiment search space, and all values provided in a list expand the search space. For example, the provided `conf_generalists.yaml` will run four different experiments:

1. *{batch_size=64, model_name=convtasnet, model_size=tiny, distance_func=snr}*
2. *{batch_size=64, model_name=convtasnet, model_size=small, distance_func=snr}*
3. *{batch_size=64, model_name=convtasnet, model_size=medium, distance_func=snr}*
4. *{batch_size=64, model_name=convtasnet, model_size=large, distance_func=snr}*

The experiments may be run across multiple GPUs and CPUs, which can be specified in the above YAML files.

## Citation

```
@article{SivaramanA2022jstsp,
  title={{Efficient Personalized Speech Enhancement Through Self-Supervised Learning}}, 
  author={Sivaraman, Aswin and Kim, Minje},
  journal={{IEEE Journal of Selected Topics in Signal Processing}}, 
  year={2022},
  volume={16},
  number={6},
  pages={1342-1356}
}
```
