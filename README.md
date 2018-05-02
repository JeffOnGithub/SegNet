# SegNet

SegNet is a model of semantic segmentation based on Fully Convolutional Network.

This repository contains the implementation of learning and testing in keras and tensorflow.
Also included is a custom layer implementation of index pooling, a new property of segnet.

![architectire](https://user-images.githubusercontent.com/27678705/33704504-199ba3ea-db70-11e7-8009-dc23aa9770a0.png)

## Architecture
- encoder decoder architecture
- fully convolutional network
- indices pooling

    ![indecespooling](https://user-images.githubusercontent.com/27678705/33704612-81053eec-db70-11e7-9822-01dd48d68314.png)

## Description
This repository is SegNet architecture for Semantic Segmentation.
The repository of other people's segmentation, pooling with indices not implemented.But In this repository we implemented  pooling layer and unpooling layer with indices at MyLayers.py.

Segnet architecture is early Semantic Segmentation model,so acccuracy is low but fast.
In the future, we plan to implement models with high accuracy.(UNet,PSPNet,Pix2Pix ect..)

## Fork (JeffOnGithub)
This repository is used for my master.
I've cleaned up / updated some stuff / adapted it for what i'm working on.
Feel free to use those changes.
