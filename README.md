# Melkani Lab: Drosophila Cardiac Analysis

This is the official implementation of the segmentation model presented in our paper **Automated evaluation of cardiac contractile dynamics using machine learning in the Drosophila models of aging and dilated cardiomyopathy**

![Visualization of Segmentation Model](./assets/pipeline-resize.gif)

# Getting Started

We provide the following:
- Model architecture and checkpoint 
- A custom Heart class to easily work with *Drosophila* recordings using our model
- A tutorial notebook for how to use the Heart class

# Acknowledgements

The model architecture was based on Malav Bateriwala's implementation of the Attention UNet, whose original repository can be found [here](https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/tree/master). We have made slight modifications to the number of convolution filters and the shape of the convolution kernels. We used our own code for model training and inference.
