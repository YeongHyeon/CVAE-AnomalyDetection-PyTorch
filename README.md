[PyTorch] Anomaly Detection using Convolutional Variational Auto-Encoder (CVAE)
=====

Example of Anomaly Detection using Convolutional Variational Auto-Encoder (CVAE) [<a href="https://github.com/YeongHyeon/CVAE-AnomalyDetection">TensorFlow 1.x</a>] [<a href="https://github.com/YeongHyeon/CVAE-AnomalyDetection-TF2">TensorFlow 2.x</a>].

## Architecture
<div align="center">
  <img src="./figures/vae.png" width="400">  
  <p>Simplified VAE architecture.</p>
</div>

## Problem Definition
<div align="center">
  <img src="./figures/definition.png" width="600">  
  <p>'Class-1' is defined as normal and the others are defined as abnormal.</p>
</div>

## Results

||MNIST|Fashion-MNIST|
|:---|:---:|:---:|
|Reconstruciton of training|<img src="./figures/mnist/restoring.png" width="500">|<img src="./figures/fmnist/restoring.png" width="500">|
|Latent of training|<img src="./figures/mnist/latent_tr.png" width="500">|<img src="./figures/fmnist/latent_tr.png" width="500">|
|Latent walk|<img src="./figures/mnist/latent_walk.png" width="500">|<img src="./figures/fmnist/latent_walk.png" width="500">|
|Latent of test|<img src="./figures/mnist/latent_te_2.png" width="500">|<img src="./figures/fmnist/latent_te_2.png" width="500">|
|Histogram of test|<img src="./figures/mnist/histogram-test.png" width="500">|<img src="./figures/fmnist/histogram-test.png" width="500">|
|AUROC|0.997|0.980|

## Environment
* Python 3.7.4  
* PyTorch 1.1.0   
* Numpy 1.17.1  
* Matplotlib 3.1.1  
* Scikit Learn (sklearn) 0.21.3  

## Reference
[1] Kingma, D. P., & Welling, M. (2013). <a href="https://arxiv.org/abs/1312.6114">Auto-encoding variational bayes</a>. arXiv preprint arXiv:1312.6114.  
[2] <a href="https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence">Kullback Leibler divergence</a>. Wikipedia
