# FDM-HDR
# Frequency-Domain Multi-Exposure HDR Imaging Network With Representative Image Features [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10310139)]

Keuntek Lee, Jaehyun Park, Yeong Il Jang and Nam Ik Cho

## Environments
- Ubuntu 18.04
- Pytorch 1.10.1
- CUDA 10.2
- CuDNN 7.6.5
- Python 3.8.3

## Abstract 

Constructing a high dynamic range (HDR) image from multi-exposure low dynamic range (LDR) images is challenging mainly due to two major problems. One is a large misalignment between the LDR images taken at different moments due to camera and object motions. The other is missing content in over- or under-exposed areas, which can be very large in the LDR images with long or short exposure times. These problems lead to ghosting artifacts or saturated regions in the reconstructed HDR image. In this paper, we propose a method using convolutional neural networks (CNNs) to address these problems. Specifically, each LDR image is fed to a two-branch CNN that extracts two kinds of features separately, named content features and global features. Then, another CNN aligns and merges the features to create the HDR output. The content features are used to create spatial attention masks, which are used to align the features in the time domain with exposure-invariant attributes. Further, Fourier coefficient vectors of global features are utilized to modulate intermediate features in the frequency domain during the reconstruction process. Experimental results show that our method achieves state-of-the-art HDR reconstruction performances on several benchmarks. 
<br><be>

### <u>Decomposition Network Architecture</u>

<p align="center"><img src="data/decomposenet.PNG" width="500"></p>

### <u>FDM-HDR Architecture</u>

<p align="center"><img src="data/FDM-HDR.PNG" width="900"></p>

## Experimental Results


<p align="center"><img src="data/visual_result.PNG" width="700"></p>
