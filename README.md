<div align="center">
<h1> 🩺 SEPNet (IEEE TCSVT 2024) </h1>
<h3>Polyp Segmentation via Semantic Enhanced Perceptual Network</h3>

  [Tong Wang](https://wangtong627.github.io/)<sup>1</sup>,
  [Xiaoming Qi](https://jerryqseu.github.io/)<sup>1</sup>,
  [Guanyu Yang](https://cs.seu.edu.cn/gyyang/main.htm)<sup>1,\*</sup>

<sup>1</sup> Southeast University
<small><span style="color:#E63946; font-weight:bold;">*</span> indicates corresponding authors</small>

<!-- [[`Paper`](https://ieeexplore.ieee.org/document/10608167)] -->
[![Paper](https://img.shields.io/badge/Paper-IEEE%20TCSVT2024-blue)](https://ieeexplore.ieee.org/document/10608167)
[![Dataset](https://img.shields.io/badge/Dataset-Google%20Drive-2C786C)](https://drive.google.com/drive/folders/1y1T-2mF4d4x_05S-fR8z-t3N1Xl5Zz7V?usp=sharing)
[![Checkpoint](https://img.shields.io/badge/Checkpoint-OneDrive-orange)](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/tong_wang_mbzuai_ac_ae/ET9u--Dah4JMhKeeJb9dGqcBd6kC9Vx1rSREPq7RqU5qzQ?e=GYPNCN)
[![Prediction Maps](https://img.shields.io/badge/Prediction_Maps-OneDrive-purple)](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/tong_wang_mbzuai_ac_ae/EXz63_SesOxLh6VKyIDdLJQBd5pp0987i5qVijbH4KIG4w?e=rjQB7d)

[![SUN-SEG Checkpoint](https://img.shields.io/badge/SUN--SEG_Checkpoint-OneDrive-orange)](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/tong_wang_mbzuai_ac_ae/IgDHwJcR4jysT7BhUd0MIcOkAYcoMrzVWPI6zjHO63k71UA?e=IeaZ5T)
[![SUN-SEG Prediction Maps](https://img.shields.io/badge/SUN--SEG_Prediction_Maps-OneDrive-purple)](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/tong_wang_mbzuai_ac_ae/IQBSSR0AxkqHR4ElpmvRPFhRAbSwULSzdD8-Fcfa5mXpeeg?e=YEtapl)

</div>
<!-- ## Preface -->

<!-- - This repository provides code for _"**Polyp Segmentation via Semantic Enhanced Perceptual Network**_" IEEE TCSVT-2024.
- [Our paper](https://ieeexplore.ieee.org/document/10608167) is published online. 
- If you have any questions about our paper, feel free to contact me.
>  **Authors:** [Tong Wang](https://wangtong627.github.io/), [Xiaoming Qi](https://jerryqseu.github.io/) & [Guanyu Yang](https://cs.seu.edu.cn/gyyang/main.htm). -->

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="DiverseVAR">
</p>

## 📢 News
- **[Dec 25, 2025]** Released **SUN-SEG dataset** model weights and prediction maps.
  - [SUN-SEG Model Weights (OneDrive)](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/tong_wang_mbzuai_ac_ae/IgDHwJcR4jysT7BhUd0MIcOkAYcoMrzVWPI6zjHO63k71UA?e=IeaZ5T)
  - [SUN-SEG Prediction Maps (OneDrive)](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/tong_wang_mbzuai_ac_ae/IQBSSR0AxkqHR4ElpmvRPFhRAbSwULSzdD8-Fcfa5mXpeeg?e=YEtapl)
- **[Aug 16, 2024]** Released [model weights](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/tong_wang_mbzuai_ac_ae/ET9u--Dah4JMhKeeJb9dGqcBd6kC9Vx1rSREPq7RqU5qzQ?e=GYPNCN) and [prediction maps](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/tong_wang_mbzuai_ac_ae/EXz63_SesOxLh6VKyIDdLJQBd5pp0987i5qVijbH4KIG4w?e=rjQB7d).
- **[Jul 24, 2024]** [Paper](https://ieeexplore.ieee.org/document/10608167) accepted by **IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)**.

## 📌 Overview
Accurate polyp segmentation is crucial for precise diagnosis and prevention of colorectal cancer. However, precise polyp segmentation still faces challenges, mainly due to the similarity of polyps to their surroundings in terms of color, shape, texture, and other aspects, making it difficult to learn accurate semantics.

To address this issue, we propose a novel semantic enhanced perceptual network (SEPNet) for polyp segmentation, which enhances polyp semantics to guide the exploration of polyp features. Specifically, we propose the Polyp Semantic Enhancement (PSE) module, which utilizes a coarse segmentation map as a basis and selects kernels to extract semantic information from corresponding regions, thereby enhancing the discriminability of polyp features highly similar to the background. Furthermore, we design a plug-and-play semantic guidance structure for the PSE, leveraging accurate semantic information to guide scale perception and context fusion, thereby enhancing feature discriminability.
Additionally, we propose a Multi-scale Adaptive Perception (MAP) module, which enhances the flexibility of receptive fields by increasing the interaction of information between neighboring receptive field branches and dynamically adjusting the size of the perception domain based on the contribution of each scale branch.
Finally, we construct the Contextual Representation Calibration (CRC) module, which calibrates contextual representations by introducing an additional branch network to supplement details.

Extensive experiments demonstrate that SEPNet outperforms 15 sota methods on five challenging datasets across six standard metrics.


## 🛠️ Installation & Usage 

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
    pip install timm==0.5.4 \
            scipy \
            numpy==1.26.0 \
            opencv-python==4.7.0.72 \
            tqdm \
            scikit-learn \
            tensorboard \
            six \
            Pillow==6.2.2
    ```

### 2. Training

To train the SEPNet model:

```bash
python myTrain.py
```
(Note: Ensure you have downloaded and set up the training dataset and pre-trained backbone weights as mentioned in the Dataset and Weight sections.)

### 3. Testing

To test the model, first download the pre-trained weights (SEPNet model) from the links in the Weight section and place it correctly in your project structure (e.g., in the checkpoint folder).
Then, run the testing script:

```bash
python myTest.py
```


## 📰 Code

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

## 💡 Dataset & Model Weights & Prediction Results
### Dataset
- downloading **testing dataset**, which can be found in this [Google Drive Link (327.2MB)](https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao/view?usp=sharing). It contains five sub-datsets: CVC-300 (60 test samples), CVC-ClinicDB (62 test samples), CVC-ColonDB (380 test samples), ETIS-LaribPolypDB (196 test samples), Kvasir (100 test samples).
- downloading **training dataset**, which can be found in this [Google Drive Link (399.5MB)](https://drive.google.com/file/d/1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb/view?usp=sharing). It contains two sub-datasets: Kvasir-SEG (900 train samples) and CVC-ClinicDB (550 train samples).

### Weight 
- During training, it is necessary to load the pre-trained parameters of the backbone network, and the weights of PVT-V2-B2 can be downloaded from [**SEU_Pan**](https://pan.seu.edu.cn:443/#/link/0775D9F57116CE2267D091181D1C86E7) or [**OneDrive**](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/tong_wang_mbzuai_ac_ae/EbgpfL8aDBxDqJgSGv3YlXABYr8atQUnrKrbKqMI7310bg?e=7BKi6m).
- You can also choose to directly load our trained model weights for direct inference, the weight of our proposed SEPNet can be downloaded at [**SEU_Pan**](https://pan.seu.edu.cn:443/#/link/A29A7D77DF2E47541397FFD38AD7A334) or [**OneDrive**](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/tong_wang_mbzuai_ac_ae/ET9u--Dah4JMhKeeJb9dGqcBd6kC9Vx1rSREPq7RqU5qzQ?e=GYPNCN).

### Prediction Results
- You can also directly download our prediction results for evaluation. The prediction map of our proposed SEPNet can be downloaded at [**SEU_Pan**](https://pan.seu.edu.cn:443/#/link/0FADA6A9BC151291FD009934F7BC4294) or [**OneDrive**](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/tong_wang_mbzuai_ac_ae/EXz63_SesOxLh6VKyIDdLJQBd5pp0987i5qVijbH4KIG4w?e=rjQB7d).

### SUN-SEG Results

We additionally provide model weights and prediction maps evaluated on the **SUN-SEG dataset**, which further demonstrates the generalization ability of SEPNet.

- **SUN-SEG Model Weights**  
  [OneDrive Download](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/tong_wang_mbzuai_ac_ae/IgDHwJcR4jysT7BhUd0MIcOkAYcoMrzVWPI6zjHO63k71UA?e=IeaZ5T)

- **SUN-SEG Prediction Maps**  
  [OneDrive Download](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/tong_wang_mbzuai_ac_ae/IQBSSR0AxkqHR4ElpmvRPFhRAbSwULSzdD8-Fcfa5mXpeeg?e=YEtapl)


## 📊 Experimental Results on IPS Datasets
### Qualitative Results
![](https://github.com/wangtong627/SEPNet/blob/main/qualitative_results.png)
_Figure: Qualitative Results._

### Quantitative Comparison 
The following tables report the quantitative comparison results on two widely used **seen datasets**,  
**CVC-ClinicDB** and **Kvasir**, following the evaluation protocol in our IEEE TCSVT 2024 paper.  
We adopt **mDice**, **mIoU**, **S-measure (Sα)**, **wFmeasure**, **meanEm**, and **MAE** as evaluation metrics,  
where higher values indicate better performance except for MAE.

#### CVC-ClinicDB (612 images)
| Method | mDice ↑ | mIoU ↑ | Sα ↑ | wFmeasure ↑ | meanEm ↑ | MAE ↓ |
|------|--------|--------|------|------------|----------|-------|
| UNet | 0.823 | 0.755 | 0.889 | 0.811 | 0.914 | 0.019 |
| UNet++ | 0.794 | 0.729 | 0.873 | 0.785 | 0.890 | 0.019 |
| SFA | 0.700 | 0.607 | 0.793 | 0.647 | 0.840 | 0.042 |
| DenseBiasNet | 0.895 | 0.838 | 0.924 | 0.884 | 0.956 | 0.014 |
| PraNet | 0.899 | 0.849 | 0.936 | 0.896 | 0.963 | 0.009 |
| ACSNet | 0.882 | 0.826 | 0.927 | 0.873 | 0.947 | 0.011 |
| MSEG | 0.909 | 0.864 | 0.938 | 0.907 | 0.961 | 0.007 |
| MSNet | 0.918 | 0.869 | 0.946 | 0.913 | 0.973 | 0.008 |
| CCBANet | 0.868 | 0.812 | 0.916 | 0.861 | 0.935 | 0.015 |
| SANet | 0.916 | 0.859 | 0.939 | 0.909 | 0.971 | 0.012 |
| DCRNet | 0.896 | 0.844 | 0.933 | 0.890 | 0.964 | 0.010 |
| LDNet | 0.881 | 0.825 | 0.924 | 0.879 | 0.960 | 0.010 |
| Polyp-Mixer | 0.908 | 0.856 | 0.943 | 0.902 | 0.963 | 0.009 |
| M2SNet | 0.922 | 0.880 | 0.942 | 0.917 | 0.970 | 0.009 |
| CFANet | 0.933 | 0.883 | 0.950 | 0.924 | 0.981 | 0.006 |
| **SEPNet (Ours)** | **0.937** | **0.883** | **0.959** | **0.933** | **0.984** | **0.006** |

#### Kvasir (1000 images)
| Method | mDice ↑ | mIoU ↑ | Sα ↑ | wFmeasure ↑ | meanEm ↑ | MAE ↓ |
|------|--------|--------|------|------------|----------|-------|
| UNet | 0.818 | 0.746 | 0.858 | 0.794 | 0.881 | 0.055 |
| UNet++ | 0.821 | 0.744 | 0.862 | 0.808 | 0.887 | 0.048 |
| SFA | 0.723 | 0.611 | 0.782 | 0.670 | 0.834 | 0.075 |
| DenseBiasNet | 0.856 | 0.778 | 0.875 | 0.833 | 0.922 | 0.043 |
| PraNet | 0.898 | 0.840 | 0.915 | 0.885 | 0.944 | 0.030 |
| ACSNet | 0.898 | 0.838 | 0.920 | 0.882 | 0.941 | 0.032 |
| MSEG | 0.897 | 0.839 | 0.912 | 0.885 | 0.942 | 0.028 |
| MSNet | 0.905 | 0.849 | 0.923 | 0.892 | 0.947 | 0.025 |
| CCBANet | 0.853 | 0.777 | 0.887 | 0.831 | 0.901 | 0.035 |
| SANet | 0.904 | 0.847 | 0.915 | 0.892 | 0.949 | 0.028 |
| DCRNet | 0.886 | 0.825 | 0.911 | 0.868 | 0.933 | 0.035 |
| LDNet | 0.887 | 0.821 | 0.905 | 0.831 | 0.941 | 0.041 |
| Polyp-Mixer | 0.916 | 0.861 | 0.922 | 0.908 | 0.959 | 0.025 |
| M2SNet | 0.912 | 0.861 | 0.924 | 0.901 | 0.953 | 0.025 |
| CFANet | 0.915 | 0.861 | 0.924 | 0.903 | 0.956 | 0.023 |
| **SEPNet (Ours)** | **0.922** | **0.869** | **0.934** | **0.914** | **0.961** | **0.022** |

The following tables present quantitative results on three **unseen datasets**,  
including **CVC-300**, **CVC-ColonDB**, and **ETIS-LaribPolypDB**, which are not used during training.  
These results demonstrate the generalization ability of SEPNet under challenging cross-dataset settings.  
We report **mDice**, **mIoU**, **S-measure (Sα)**, **wFmeasure**, **meanEm**, and **MAE** as evaluation metrics.

#### CVC-300 (60 images)
| Method | mDice ↑ | mIoU ↑ | Sα ↑ | wFmeasure ↑ | meanEm ↑ | MAE ↓ |
|------|--------|--------|------|------------|----------|-------|
| UNet | 0.710 | 0.627 | 0.843 | 0.684 | 0.848 | 0.022 |
| UNet++ | 0.707 | 0.624 | 0.839 | 0.687 | 0.834 | 0.018 |
| SFA | 0.467 | 0.329 | 0.640 | 0.341 | 0.644 | 0.065 |
| DenseBiasNet | 0.809 | 0.724 | 0.880 | 0.769 | 0.921 | 0.019 |
| PraNet | 0.871 | 0.797 | 0.925 | 0.843 | 0.950 | 0.010 |
| ACSNet | 0.863 | 0.787 | 0.923 | 0.825 | 0.939 | 0.013 |
| MSEG | 0.874 | 0.804 | 0.924 | 0.852 | 0.948 | 0.009 |
| MSNet | 0.865 | 0.799 | 0.926 | 0.848 | 0.945 | 0.010 |
| CCBANet | 0.888 | 0.815 | 0.928 | 0.859 | 0.962 | 0.010 |
| SANet | 0.888 | 0.815 | 0.928 | 0.859 | 0.962 | 0.010 |
| DCRNet | 0.856 | 0.788 | 0.921 | 0.830 | 0.943 | 0.010 |
| LDNet | 0.869 | 0.793 | 0.923 | 0.841 | 0.948 | 0.010 |
| Polyp-Mixer | 0.891 | 0.820 | 0.929 | 0.861 | 0.948 | 0.009 |
| M2SNet | 0.869 | 0.807 | 0.939 | 0.881 | 0.965 | 0.009 |
| CFANet | 0.893 | 0.827 | 0.938 | 0.875 | 0.962 | 0.008 |
| **SEPNet (Ours)** | **0.937** | **0.883** | **0.959** | **0.933** | **0.984** | **0.006** |

#### CVC-ColonDB (380 images)
| Method | mDice ↑ | mIoU ↑ | Sα ↑ | wFmeasure ↑ | meanEm ↑ | MAE ↓ |
|------|--------|--------|------|------------|----------|-------|
| UNet | 0.504 | 0.436 | 0.710 | 0.491 | 0.692 | 0.061 |
| UNet++ | 0.482 | 0.408 | 0.693 | 0.467 | 0.680 | 0.064 |
| SFA | 0.456 | 0.337 | 0.629 | 0.366 | 0.604 | 0.095 |
| DenseBiasNet | 0.646 | 0.552 | 0.763 | 0.614 | 0.803 | 0.061 |
| PraNet | 0.712 | 0.640 | 0.820 | 0.809 | 0.847 | 0.043 |
| ACSNet | 0.716 | 0.649 | 0.829 | 0.837 | 0.839 | 0.039 |
| MSEG | 0.735 | 0.666 | 0.834 | 0.820 | 0.859 | 0.038 |
| MSNet | 0.751 | 0.671 | 0.838 | 0.820 | 0.872 | 0.041 |
| CCBANet | 0.706 | 0.626 | 0.812 | 0.768 | 0.852 | 0.042 |
| SANet | 0.753 | 0.670 | 0.837 | 0.826 | 0.869 | 0.043 |
| DCRNet | 0.704 | 0.631 | 0.821 | 0.804 | 0.840 | 0.052 |
| LDNet | 0.740 | 0.652 | 0.830 | 0.817 | 0.876 | 0.046 |
| Polyp-Mixer | 0.791 | 0.706 | 0.862 | 0.768 | 0.893 | 0.039 |
| M2SNet | 0.758 | 0.685 | 0.842 | 0.737 | 0.869 | 0.038 |
| CFANet | 0.743 | 0.665 | 0.835 | 0.728 | 0.869 | 0.039 |
| **SEPNet (Ours)** | **0.859** | **0.740** | **0.878** | **0.857** | **0.926** | **0.026** |

#### ETIS-LaribPolypDB (196 images)
| Method | mDice ↑ | mIoU ↑ | Sα ↑ | wFmeasure ↑ | meanEm ↑ | MAE ↓ |
|------|--------|--------|------|------------|----------|-------|
| UNet | 0.398 | 0.335 | 0.684 | 0.366 | 0.643 | 0.036 |
| UNet++ | 0.401 | 0.344 | 0.683 | 0.390 | 0.629 | 0.035 |
| SFA | 0.297 | 0.217 | 0.557 | 0.231 | 0.531 | 0.109 |
| DenseBiasNet | 0.473 | 0.390 | 0.672 | 0.417 | 0.695 | 0.067 |
| PraNet | 0.628 | 0.567 | 0.794 | 0.700 | 0.808 | 0.031 |
| ACSNet | 0.578 | 0.509 | 0.754 | 0.578 | 0.737 | 0.059 |
| MSEG | 0.700 | 0.630 | 0.828 | 0.671 | 0.855 | 0.015 |
| MSNet | 0.723 | 0.652 | 0.845 | 0.677 | 0.875 | 0.020 |
| CCBANet | 0.559 | 0.483 | 0.751 | 0.513 | 0.783 | 0.038 |
| SANet | 0.750 | 0.654 | 0.849 | 0.685 | 0.881 | 0.015 |
| DCRNet | 0.556 | 0.496 | 0.736 | 0.506 | 0.742 | 0.096 |
| LDNet | 0.645 | 0.551 | 0.788 | 0.600 | 0.841 | 0.041 |
| Polyp-Mixer | 0.759 | 0.676 | 0.863 | 0.711 | 0.875 | 0.017 |
| M2SNet | 0.749 | 0.678 | 0.846 | 0.712 | 0.872 | 0.017 |
| CFANet | 0.732 | 0.655 | 0.845 | 0.693 | 0.881 | 0.016 |
| **SEPNet (Ours)** | **0.803** | **0.721** | **0.882** | **0.771** | **0.934** | **0.014** |

## 📊 Experimental Results on SUN-SEG Dataset

We report quantitative comparisons on the **SUN-SEG benchmark**, following the official evaluation protocol.
The dataset is divided into **Easy/Hard** and **Seen/Unseen** subsets.
### TestEasyDataset / Seen

| Method | Smeasure | adpEm | meanEm | meanFm | wFmeasure | meanSen | meanSpe | meanDice | meanIoU | MAE |
|-------|----------|-------|--------|--------|-----------|---------|---------|----------|---------|-----|
| COSNet | 0.845 | 0.932 | 0.836 | 0.774 | 0.727 | 0.691 | 0.963 | 0.730 | 0.648 | 0.034 |
| PCSA | 0.852 | 0.875 | 0.835 | 0.744 | 0.681 | 0.703 | 0.968 | 0.709 | 0.604 | 0.039 |
| 23DCNN | 0.895 | 0.941 | 0.909 | 0.853 | 0.819 | 0.808 | 0.977 | 0.829 | 0.756 | 0.021 |
| DCFNet | 0.572 | 0.611 | 0.591 | 0.393 | 0.357 | 0.475 | 0.648 | 0.389 | 0.320 | 0.174 |
| AMD | 0.471 | 0.527 | 0.526 | 0.122 | 0.114 | 0.287 | 0.552 | 0.135 | 0.083 | 0.194 |
| PraNet | 0.918 | 0.953 | 0.942 | 0.902 | 0.877 | 0.869 | 0.985 | 0.883 | 0.825 | 0.020 |
| ACSNet | 0.920 | 0.951 | 0.942 | 0.892 | 0.874 | 0.878 | 0.967 | 0.882 | 0.828 | 0.017 |
| SANet | 0.916 | 0.954 | 0.933 | 0.885 | 0.866 | 0.863 | 0.973 | 0.872 | 0.820 | 0.018 |
| **SEPNet** | **0.931** | **0.968** | **0.962** | **0.908** | **0.883** | **0.887** | **0.985** | **0.896** | **0.834** | **0.017** |

### TestEasyDataset / Seen

| Method | Smeasure | adpEm | meanEm | meanFm | wFmeasure | meanSen | meanSpe | meanDice | meanIoU | MAE |
|-------|----------|-------|--------|--------|-----------|---------|---------|----------|---------|-----|
| COSNet | 0.845 | 0.932 | 0.836 | 0.774 | 0.727 | 0.691 | 0.963 | 0.730 | 0.648 | 0.034 |
| PCSA | 0.852 | 0.875 | 0.835 | 0.744 | 0.681 | 0.703 | 0.968 | 0.709 | 0.604 | 0.039 |
| 23DCNN | 0.895 | 0.941 | 0.909 | 0.853 | 0.819 | 0.808 | 0.977 | 0.829 | 0.756 | 0.021 |
| DCFNet | 0.572 | 0.611 | 0.591 | 0.393 | 0.357 | 0.475 | 0.648 | 0.389 | 0.320 | 0.174 |
| AMD | 0.471 | 0.527 | 0.526 | 0.122 | 0.114 | 0.287 | 0.552 | 0.135 | 0.083 | 0.194 |
| PraNet | 0.918 | 0.953 | 0.942 | 0.902 | 0.877 | 0.869 | 0.985 | 0.883 | 0.825 | 0.020 |
| ACSNet | 0.920 | 0.951 | 0.942 | 0.892 | 0.874 | 0.878 | 0.967 | 0.882 | 0.828 | 0.017 |
| SANet | 0.916 | 0.954 | 0.933 | 0.885 | 0.866 | 0.863 | 0.973 | 0.872 | 0.820 | 0.018 |
| **SEPNet** | **0.931** | **0.968** | **0.962** | **0.908** | **0.883** | **0.887** | **0.985** | **0.896** | **0.834** | **0.017** |


### TestHardDataset / Seen

| Method | Smeasure | adpEm | meanEm | meanFm | wFmeasure | meanSen | meanSpe | meanDice | meanIoU | MAE |
|-------|----------|-------|--------|--------|-----------|---------|---------|----------|---------|-----|
| COSNet | 0.785 | 0.894 | 0.772 | 0.683 | 0.626 | 0.594 | 0.955 | 0.633 | 0.541 | 0.046 |
| PCSA | 0.772 | 0.819 | 0.759 | 0.636 | 0.566 | 0.560 | 0.935 | 0.585 | 0.479 | 0.057 |
| 23DCNN | 0.849 | 0.917 | 0.869 | 0.805 | 0.753 | 0.726 | 0.971 | 0.764 | 0.671 | 0.035 |
| DCFNet | 0.603 | 0.640 | 0.602 | 0.427 | 0.385 | 0.467 | 0.740 | 0.411 | 0.340 | 0.135 |
| AMD | 0.480 | 0.528 | 0.536 | 0.124 | 0.115 | 0.250 | 0.576 | 0.129 | 0.079 | 0.171 |
| PraNet | 0.884 | 0.930 | 0.919 | 0.865 | 0.831 | 0.816 | 0.974 | 0.839 | 0.766 | 0.031 |
| ACSNet | 0.872 | 0.919 | 0.910 | 0.835 | 0.806 | 0.814 | 0.975 | 0.820 | 0.748 | 0.036 |
| SANet | 0.874 | 0.924 | 0.905 | 0.844 | 0.810 | 0.801 | 0.969 | 0.820 | 0.748 | 0.033 |
| **SEPNet** | **0.894** | **0.943** | **0.940** | **0.870** | **0.835** | **0.852** | **0.979** | **0.857** | **0.776** | **0.034** |

### TestHardDataset / Unseen

| Method | Smeasure | adpEm | meanEm | meanFm | wFmeasure | meanSen | meanSpe | meanDice | meanIoU | MAE |
|-------|----------|-------|--------|--------|-----------|---------|---------|----------|---------|-----|
| COSNet | 0.670 | 0.825 | 0.627 | 0.506 | 0.443 | 0.380 | 0.851 | 0.438 | 0.353 | 0.070 |
| PCSA | 0.682 | 0.792 | 0.660 | 0.510 | 0.442 | 0.415 | 0.871 | 0.450 | 0.351 | 0.080 |
| 23DCNN | 0.786 | 0.843 | 0.775 | 0.688 | 0.634 | 0.607 | 0.921 | 0.644 | 0.558 | 0.044 |
| DCFNet | 0.514 | 0.556 | 0.522 | 0.303 | 0.263 | 0.364 | 0.615 | 0.290 | 0.225 | 0.185 |
| AMD | 0.472 | 0.578 | 0.527 | 0.141 | 0.128 | 0.213 | 0.551 | 0.135 | 0.086 | 0.183 |
| PraNet | 0.787 | 0.848 | 0.802 | 0.726 | 0.667 | 0.627 | 0.931 | 0.675 | 0.587 | 0.053 |
| ACSNet | 0.762 | 0.799 | 0.776 | 0.657 | 0.610 | 0.600 | 0.852 | 0.624 | 0.547 | 0.053 |
| SANet | 0.753 | 0.817 | 0.736 | 0.633 | 0.590 | 0.562 | 0.856 | 0.595 | 0.527 | 0.055 |
| **SEPNet** | **0.847** | **0.902** | **0.895** | **0.791** | **0.745** | **0.776** | **0.953** | **0.774** | **0.684** | **0.039** |



## 🤝 FAQ

- If you want to improve the usability or any piece of advice, please feel free to contact me directly ([E-mail](tong.wang@mbzuai.ac.ae)).

## 🔍 Citation

```bibtex
@article{wang2024polyp,
title={Polyp Segmentation via Semantic Enhanced Perceptual Network},
author={Wang, Tong and Qi, Xiaoming and Yang, Guanyu},
journal={IEEE Transactions on Circuits and Systems for Video Technology},
year={2024},
publisher={IEEE}
}
```

