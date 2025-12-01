<div align="center">
<h1> SEPNet (IEEE TCSVT 2024) </h1>
<h3>Polyp Segmentation via Semantic Enhanced Perceptual Network</h3>

Tong Wang<sup>1</sup>, Xiaoming Qi<sup>1</sup>, and Guanyu Yang<sup>1,\*</sup>

<sup>1</sup> Southeast University
<small><span style="color:#E63946; font-weight:bold;">*</span> indicates corresponding authors</small>

[[`Paper`](https://ieeexplore.ieee.org/document/10608167)]
</div>
<!-- ## Preface -->

<!-- - This repository provides code for _"**Polyp Segmentation via Semantic Enhanced Perceptual Network**_" IEEE TCSVT-2024.
- [Our paper](https://ieeexplore.ieee.org/document/10608167) is published online. 
- If you have any questions about our paper, feel free to contact me.
>  **Authors:** [Tong Wang](https://wangtong627.github.io/), [Xiaoming Qi](https://jerryqseu.github.io/) & [Guanyu Yang](https://cs.seu.edu.cn/gyyang/main.htm). -->

## News

- [Aug/16/2024] We have open-sourced the [model weight](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/tong_wang_mbzuai_ac_ae/ET9u--Dah4JMhKeeJb9dGqcBd6kC9Vx1rSREPq7RqU5qzQ?e=GYPNCN) and [prediction results](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/tong_wang_mbzuai_ac_ae/EXz63_SesOxLh6VKyIDdLJQBd5pp0987i5qVijbH4KIG4w?e=rjQB7d
). 
- [Jul/24/2024] Our paper has been accepted by IEEE Transactions on Circuits and Systems for Video Technology (IEEE TCSVT).

## Overview

### Introduction

Accurate polyp segmentation is crucial for precise diagnosis and prevention of colorectal cancer. However, precise polyp segmentation still faces challenges, mainly due to the similarity of polyps to their surroundings in terms of color, shape, texture, and other aspects, making it difficult to learn accurate semantics.

To address this issue, we propose a novel semantic enhanced perceptual network (SEPNet) for polyp segmentation, which enhances polyp semantics to guide the exploration of polyp features. Specifically, we propose the Polyp Semantic Enhancement (PSE) module, which utilizes a coarse segmentation map as a basis and selects kernels to extract semantic information from corresponding regions, thereby enhancing the discriminability of polyp features highly similar to the background. Furthermore, we design a plug-and-play semantic guidance structure for the PSE, leveraging accurate semantic information to guide scale perception and context fusion, thereby enhancing feature discriminability.
Additionally, we propose a Multi-scale Adaptive Perception (MAP) module, which enhances the flexibility of receptive fields by increasing the interaction of information between neighboring receptive field branches and dynamically adjusting the size of the perception domain based on the contribution of each scale branch.
Finally, we construct the Contextual Representation Calibration (CRC) module, which calibrates contextual representations by introducing an additional branch network to supplement details.

Extensive experiments demonstrate that SEPNet outperforms 15 sota methods on five challenging datasets across six standard metrics.

### Qualitative Results

![](https://github.com/wangtong627/SEPNet/blob/main/qualitative_results.png)
_Figure: Qualitative Results._

---
## Installation and Usage 🛠️

Following are the steps to set up the environment and use the project code:

### 1. Prerequisites of Environment

We recommend using **Anaconda** to manage the environment. The code has been tested with **Python 3.10**, **PyTorch 1.13.1 (with CUDA 11.7)**, and **Torchvision 0.14.1**.

1.  **Create and activate the Conda environment:**
    ```bash
    conda create -n SEPNet python=3.10
    conda activate SEPNet
    ```

2.  **Install PyTorch and Torchvision:**
    For the specified versions (e.g., CUDA 11.7), you can install them using the following command (check the official PyTorch website for the latest or specific installation instructions for your system):
    ```bash
    # Example for PyTorch 1.13.1 with CUDA 11.7
    pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --extra-index-url [https://download.pytorch.org/whl/cu117](https://download.pytorch.org/whl/cu117)
    ```

3.  **Install other dependencies:**
    The following libraries are required. Note that specific older versions were used for development and are recommended for reproducibility.
    ```bash
    pip install tensorboardX tqdm Pillow==6.2.2 scipy==1.2.2 opencv-python==3.4.2.17
    # If a general tensorboardX is not enough, you might need to install tnt depending on the implementation:
    # pip install git+[https://github.com/pytorch/tnt.git@master](https://github.com/pytorch/tnt.git@master)
    ```

### 2. Training

To train the SEPNet model:

```bash
python myTrain.py
```
(Note: Ensure you have downloaded and set up the training dataset and pre-trained backbone weights as mentioned in the Dataset and Weight sections.)


### 2. Testing

To test the model, first download the pre-trained weights (SEPNet model) from the links in the Weight section and place it correctly in your project structure (e.g., in the checkpoint folder).

Then, run the testing script:

```bash
python myTest.py
```


## Code

The structure of the project is as follows.
```markdown
- checkpoint
- lib
    - backbones
        - efficientnet.py
        - efficientnet_utils.py
        - __init__.py
        - pvtv2.py
        - res2net.py
        - resnet.py
    - __init__.py
    - model_for_eval.py
    - model_for_train.py
    - modules
        - cbr_block.py
        - crc_module.py
        - encoder.py
        - get_logit.py
        - __init__.py
        - map_module.py
        - pse_module.py
- measure
    - eval_list.py
    - eval_metrics.py
    - __init__.py
    - metric.py
- myTest.py
- myTrain.py
- result
- utils
    - dataloader.py
    - __init__.py
    - loss_func.py
    - trainer_for_six_logits.py
    - utils.py
```

## Dataset
- downloading **testing dataset**, which can be found in this [Google Drive Link (327.2MB)](https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao/view?usp=sharing). It contains five sub-datsets: CVC-300 (60 test samples), CVC-ClinicDB (62 test samples), CVC-ColonDB (380 test samples), ETIS-LaribPolypDB (196 test samples), Kvasir (100 test samples).
- downloading **training dataset**, which can be found in this [Google Drive Link (399.5MB)](https://drive.google.com/file/d/1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb/view?usp=sharing). It contains two sub-datasets: Kvasir-SEG (900 train samples) and CVC-ClinicDB (550 train samples).

## Weight 
- During training, it is necessary to load the pre-trained parameters of the backbone network, and the weights of PVT-V2-B2 can be downloaded from [**SEU_Pan**](https://pan.seu.edu.cn:443/#/link/0775D9F57116CE2267D091181D1C86E7) or [**OneDrive**](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/tong_wang_mbzuai_ac_ae/EbgpfL8aDBxDqJgSGv3YlXABYr8atQUnrKrbKqMI7310bg?e=7BKi6m).
- You can also choose to directly load our trained model weights for direct inference, the weight of our proposed SEPNet can be downloaded at [**SEU_Pan**](https://pan.seu.edu.cn:443/#/link/A29A7D77DF2E47541397FFD38AD7A334) or [**OneDrive**](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/tong_wang_mbzuai_ac_ae/ET9u--Dah4JMhKeeJb9dGqcBd6kC9Vx1rSREPq7RqU5qzQ?e=GYPNCN).

## Prediction Results
- You can also directly download our prediction results for evaluation. The prediction map of our proposed SEPNet can be downloaded at [**SEU_Pan**](https://pan.seu.edu.cn:443/#/link/0FADA6A9BC151291FD009934F7BC4294) or [**OneDrive**](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/tong_wang_mbzuai_ac_ae/EXz63_SesOxLh6VKyIDdLJQBd5pp0987i5qVijbH4KIG4w?e=rjQB7d).

## Citation

```
@article{wang2024polyp,
title={Polyp Segmentation via Semantic Enhanced Perceptual Network},
author={Wang, Tong and Qi, Xiaoming and Yang, Guanyu},
journal={IEEE Transactions on Circuits and Systems for Video Technology},
year={2024},
publisher={IEEE}
}
```

