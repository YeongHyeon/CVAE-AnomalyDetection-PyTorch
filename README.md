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

### Training
<div align="center">
  <img src="./figures/restoring.png" width="800">  
  <p>Restoration result by CVAE.</p>
</div>

<div align="center">
  <img src="./figures/latent_tr.png" width="300"><img src="./figures/latent_walk.png" width="250">
  <p>Latent vector space of training set, and reconstruction result of latent space walking.</p>
</div>

### Test
#### z_dim = 2
<div align="center">
  <img src="./figures/latent_te_2.png" width="350"><img src="./figures/test-box_2.png" width="400">    
  <p>Left figure shows latent vector space of test set. Right figure shows box plot with restoration loss of test procedure.</p>
</div>

#### z_dim = 128
<div align="center">
  <img src="./figures/latent_te_128.png" width="350"><img src="./figures/test-box_128.png" width="400"><img src="./figures/histogram-test.png" width="440">
  <p>Latent vector space of test set, box plot with restoration loss, and histogram of restoration loss.</p>
</div>

## Environment
* Python 3.7.4  
* PyTorch 1.1.0   
* Numpy 1.17.1  
* Matplotlib 3.1.1  
* Scikit Learn (sklearn) 0.21.3  

## Reference
[1] Kingma, D. P., & Welling, M. (2013). <a href="https://arxiv.org/abs/1312.6114">Auto-encoding variational bayes</a>. arXiv preprint arXiv:1312.6114.  
[2] <a href="https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence">Kullback Leibler divergence</a>. Wikipedia
