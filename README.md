# Measuring Data Distribution Shifts for Safe Image Classification and Segmentation using Deep Learning

## Abstract


We posit that test data can only be safe to use up to a certain threshold of data distribution shift, beyond which model predictions can no longer be trusted. To that end, we compare CNN, RNN, ResNet18 and ViT (Visual Transformer) architectures on MNIST, CIFAR10 and VOC Segmentation datasets, where the test data is subject to 12 different image perturbation types at 10 different intensity levels each. We consider images as data distributions, and use histogram overlapping, KL Divergence and Bhattacharya Distance to quantify the shift between original and perturbed test image. We investigate the relationships between these three metrics and accuracy, with results showing emerging patterns that may indicate when a metric may be helpful in the evaluation of safe operation under distribution shifts.

## Supporting code

All code was run on Google Colab and the Hyperion computing cluster at City, University of London.

### Training and evaluation

The *scripts/* subdirectory contains the code used to run the experiments, where every dataset and model combination has a set of 3 files e.g.:

```
mnist_cnn_eval.py  
mnist_cnn_train.py  
mnist_cnn_train.sh
```
where *mnist_cnn_train.py* is the training script, *mnist_cnn_train.sh* is the batch script and *mnist_cnn_eval.py* is the evaluation script, used to evaluate the trained model and also to output the evaluation data.

### Figures

The *figures* sub-directory contains code used to generate figures.


