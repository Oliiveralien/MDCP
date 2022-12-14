# MDCP

This repo contains the code, results and evaluations of the paper ''Multi-Domain Clustering Pruning: Exploring Space and Frequency Similarity
Based on GAN'.

<img src="https://github.com/Oliiveralien/MDCP/blob/main/figs/teaser.jpeg" width="800" height="500" alt="抖音小程序"/><br/>

Network compression plays an important role in accelerating deep neural networks, especially in the application of edge devices such as unmanned cars and drones. Recently, pruning-based methods have been improved significantly, but they still suffer from low efficiency because most of them only pay attention to feature similarities in the space domain. In this paper, we propose a multi-domain pruning method based on clustering (MDCP) which seamlessly integrates sufficient information extraction and knowledge distillation within a GAN-based framework, to address these aforementioned limitations.
Specifically, 1) to exhaustively analyse the features for reasonable pruning, we perform pruning taking both space and frequency information into account, which considering the spectral statistics to produce more accurate pruning map; 2) to tackle the distance distortion problem caused by feature insufficiency, we further propose a clustering-based measurement mechanism to acquire pruning guidance under extreme conditions; 3) to avoid the dependence on labels and fine-tuning in previous works, a generative adversarial mechanism with two label-level losses is introduced, which further ensures the pruning efficiency and accuracy. Such a multi-domain clustering-based framework along with an adversarial and contrastive learning pattern significantly improves the pruning quality. Comprehensive experiments conducted on four benchmarks demonstrate that our MDCP method performs favorably against existing competitors. On CIFAR-10 dataset, for ResNet-110 network, the MDCP method outperforms the former SOTA method (94.33%) in terms of top-1 accuracy (94.61%) and achieves a maximum parameter pruning rate (73.33%). 

![All text](https://github.com/Oliiveralien/MDCP/blob/main/figs/model.jpeg)

Several pre-trained models on CIFAR-10 can be download here:
[ResNet20](https://drive.google.com/file/d/1-vy6OTjTDbWRIJSuxHAXPwxJ7I8KDQat/view?usp=sharing),
[ResNet34](https://drive.google.com/file/d/1BJvA9ausEdQrmGqYdwpiMyV8EqUzw5KG/view?usp=sharing),
[ResNet56](https://drive.google.com/file/d/1_f8cRv7GxzJamU_8H5ct6AcxMXE3CThM/view?usp=sharing),
[ResNet110](https://drive.google.com/file/d/1R8gl7Q18pIcHrxFkVGS-tYoR8LEfpxyz/view?usp=sharing).

More pre-trained models, running instructions and the completed codes will be avaiable as soon as possible.
