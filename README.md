# Fashion-MNIST VAE
This repository contains an implementation of a Variational Autoencoder (VAE) that generates images from the Fashion-MNIST dataset.

## What is a Variational Autoencoder (VAE)?
A Variational Autoencoder is a type of neural network that can generate new data that is similar to the training data. In the case of image generation, a VAE can learn the features of the input images and generate new images that resemble the original images. VAEs are trained using a probabilistic approach that allows them to generate images that are different from the original images while still being similar.

## Dataset
The Fashion-MNIST dataset contains grayscale images of clothing items, such as t-shirts, dresses, shoes, and bags. The dataset contains 60,000 training images and 10,000 test images, with a size of 28x28 pixels.

## Getting Started
To use this repository, you'll need to have the following prerequisites installed:

- Python 3  
- NumPy  
- TensorFlow  

You can install the dependencies using pip:

          pip install -r requirements.txt

## References
- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. arXiv preprint arXiv:1312.6114. https://arxiv.org/abs/1312.6114

## Reconstructions
<p align="center">
<img src="results/reconstruction.png">
</p>  

## Latent space
<p align="center">
<img src="results/latent_space.png">
</p>  

## Image generation using [-3, 3] * [-3, 3] grid
<p align="center">
<img src="results/generation.png">
</p>
