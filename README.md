# DL_MiniProject1
# Description:
A residual neural network (ResNet) is an artificial neural network (ANN) which utilizes skip connections, or shortcuts to jump over some layers.  Typical ResNet models are implemented with double- or triple- layer skips that contain nonlinearities and batch normalization in between. In this project, we propose a ResNet-18 architecture which uses SGD Nesterov optimizer in combination with Manifold-Mixup regularization and Affine Transform data augmentation techniques. We provide comprehensive empirical evidence showing that this residual network performs optimally for CIFAR-10 dataset with a test accuracy of 94.1%.

# Final Configuration used:
Hyperparameter / Method	- Value/ Name

No. of Channels (C)	- 32

No. of Residual Layers (N)	- 4

No. of Residual Blocks (B)	- 2

Convolutional kernel size (F)	- 3

Shortcut kernel size (K)	- 1

Avg pooling kernel size (P)	- 4

Optimizer	- SGD with Nesterov

Data Augmentation	- Affine + Basic Transformations

Regularization	- Manifold Mixup

No. of Trainable Parameters	- 2,797,610


# Usage:
To Train the network and generate the .pt weights file:
python main.py --job_id  ‘<str_id>’

The job id argument is a string which helps in creation of a separate folder which would contain the accuracy plot files and other log files.
