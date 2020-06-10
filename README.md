# Online Kernel based Generative Adversarial Network
## Abstract:
One of the major breakthroughs in deep learning over the past five years has been the Generative Adversarial Network (GAN), a neural network-based generative model which aims to mimic some underlying distribution given a dataset of samples. In contrast to many supervised problems, where one tries to minimize a simple objective function of the parameters, GAN training is formulated as a min-max problem over a pair of network parameters. While empirically GANs have shown impressive success in several domains, researchers have been puzzled by unusual training behavior, including cycling so-called mode collapse. In this paper, we begin by providing a quantitative method to explore some of the challenges in GAN training, and we show empirically how this relates fundamentally to the parametric nature of the discriminator network. We propose a novel approach that resolves many of these issues by relying on a kernel-based non-parametric discriminator that is highly amenable to online training---we call this the Online Kernel-based Generative Adversarial Networks (OKGAN). We show empirically that OKGANs mitigate a number of training issues, including mode collapse and cycling, and are much more amenable to theoretical guarantees. OKGANs empirically perform dramatically better, with respect to reverse KL-divergence, than other GAN formulations on synthetic data; on classical vision datasets such as MNIST, SVHN, and CelebA, show comparable performance.
## How to run codes
- OKGAN: Go to "OKGAN" folder -> run "Testing_Online_Kernel_GAN.ipynb"
- BourGAN: Go to "BourGAN" folder -> Go to "src" -> run "gaussian_grid.ipynb", "gaussian_ring.ipynb", "gaussian_circle.ipynb" (We reimplement BourGAN based on neural network architectures of PacGAN)
## Results

## Requirements
pytorch 1.4.0

numpy 1.18.1

matplotlib 3.1.1

sklearn 0.22.1

