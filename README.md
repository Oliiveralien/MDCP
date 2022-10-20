# MDCP

This repo contains the code, results and evaluations of the paper ''Multi-Domain Clustering Pruning: Exploring Space and Frequency Similarity
Based on GAN'.

![All text]<img src="[https://github.com/Oliiveralien/MDCP/blob/main/figs/teaser.jpeg" width="200" height="200" alt="抖音小程序"/><br/>

Network compression plays an important role in accelerating deep neural networks, especially in the application of edge devices such as unmanned cars and drones. Recently, pruning-based methods have been improved significantly, but they still suffer from low efficiency because most of them only pay attention to feature similarities in the space domain. In this paper, we propose a multi-domain pruning method based on clustering (MDCP) which seamlessly integrates sufficient information extraction and knowledge distillation within a GAN-based framework, to address these aforementioned limitations.
Specifically, 1) to exhaustively analyse the features for reasonable pruning, we perform pruning taking both space and frequency information into account, which considering the spectral statistics to produce more accurate pruning map; 2) to tackle the distance distortion problem caused by feature insufficiency, we further propose a clustering-based measurement mechanism to acquire pruning guidance under extreme conditions; 3) to avoid the dependence on labels and fine-tuning in previous works, a generative adversarial mechanism with two label-level losses is introduced, which further ensures the pruning efficiency and accuracy. Such a multi-domain clustering-based framework along with an adversarial and contrastive learning pattern significantly improves the pruning quality. Comprehensive experiments conducted on three benchmarks demonstrate that our MDCP method performs favorably against existing competitors. On CIFAR-10 dataset, for ResNet-110 network, the MDCP method outperforms the former SOTA method (94.33%) in terms of top-1 accuracy (94.61%) and achieves a maximum parameter pruning rate (73.33%). 

![All text](https://github.com/Oliiveralien/MDCP/blob/main/figs/model.jpeg)

![All text](https://github.com/Oliiveralien/MDCP/blob/main/figs/vis.jpeg)

The completed code will be avaiable as soon as possible.
