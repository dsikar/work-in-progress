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

We also use *mnist_cnn_train.py* to find adequate calibration values such that a similar range of accuracies are found for each perturbation type e.g. for contrast we find:

| Contrast Level | Accuracy |    BD   |    KL   |    HI   |
|----------------|----------|---------|---------|---------|
|      3.0       | 96.0000  | 0.0235  | 1.7749  | 0.9050  |
|      2.9       | 95.8300  | 0.0234  | 1.7614  | 0.9053  |
|      2.8       | 95.5800  | 0.0233  | 1.7473  | 0.9056  |
|      2.7       | 95.1400  | 0.0232  | 1.7322  | 0.9058  |
|      2.6       | 94.8900  | 0.0231  | 1.7221  | 0.9061  |
|      2.5       | 94.4500  | 0.0230  | 1.7080  | 0.9063  |
|      2.4       | 94.0300  | 0.0229  | 1.6932  | 0.9066  |
|      2.3       | 93.5400  | 0.0228  | 1.6782  | 0.9070  |
|      2.2       | 93.0300  | 0.0226  | 1.6614  | 0.9073  |
|      2.1       | 92.4100  | 0.0225  | 1.6498  | 0.9075  |
|      2.0       | 91.7900  | 0.0224  | 1.6375  | 0.9078  |
|      1.9       | 91.0100  | 0.0223  | 1.6164  | 0.9082  |
|      1.8       | 90.2900  | 0.0221  | 1.6051  | 0.9086  |
|      1.7       | 89.4800  | 0.0219  | 1.5845  | 0.9090  |
|      1.6       | 88.7600  | 0.0218  | 1.5692  | 0.9094  |
|      1.5       | 87.8000  | 0.0217  | 1.5511  | 0.9098  |
|      1.4       | 87.0500  | 0.0215  | 1.5365  | 0.9102  |
|      1.3       | 86.5100  | 0.0214  | 1.5199  | 0.9107  |
|      1.2       | 85.9600  | 0.0212  | 1.4996  | 0.9113  |
|      1.1       | 85.4700  | 0.0210  | 1.4771  | 0.9119  |
|      1.0       | 85.0700  | 0.0208  | 1.4600  | 0.9125  |
|      0.9       | 84.3400  | 0.0207  | 1.4402  | 0.9131  |
|      0.8       | 83.9800  | 0.0205  | 1.4202  | 0.9139  |
|      0.7       | 83.5500  | 0.0203  | 1.3963  | 0.9147  |
|      0.6       | 83.0200  | 0.0200  | 1.3659  | 0.9157  |
|      0.5       | 82.4400  | 0.0197  | 1.3381  | 0.9167  |
|      0.4       | 81.8100  | 0.0195  | 1.3016  | 0.9178  |
|      0.3       | 81.0200  | 0.0192  | 1.2624  | 0.9191  |
|      0.2       | 80.0600  | 0.0188  | 1.2234  | 0.9206  |
|      0.0       | 78.3400  | 0.0179  | 1.1160  | 0.9247  |
|     -0.1       | 77.0800  | 0.0174  | 1.0530  | 0.9272  |
|     -0.2       | 75.2700  | 0.0169  | 0.9794  | 0.9306  |
|     -0.3       | 73.4300  | 0.0158  | 0.8528  | 0.9369  |
|     -0.4       | 70.9200  | 0.0089  | 0.4424  | 0.9652  |
|     -0.5       | 65.2400  | 0.0001  | 0.0018  | 0.9997  |
|     -0.6       | 54.8700  | 0.0001  | 0.0039  | 0.9994  |
|     -0.7       | 38.7700  | 0.0001  | 0.0031  | 0.9995  |
|     -0.8       | 16.6700  | 0.0001  | 0.0025  | 0.9995  |
|     -0.9       | 9.7400   | 0.0001  | 0.0030  | 0.9994  |
|     -1.0       | 9.7400   | -1.3617 | 28.6177 | 0.0033  |

Note the calibration values are used to adjust the accuracy for e.g. the MNIST perturbation level table:


| Lv | Brght | Cntrs | DfBlr | Fog  | Frost | GsNse | IpNse | MtBlr | Pxlat | ShNse | Snow | ZmBlr |
|----|-------|-------|-------|------|-------|-------|-------|-------|-------|-------|------|-------|
| 1  | 98.52 | 79.18 | 98.40 | 98.50| 98.37 | 98.31 | 94.84 | 98.38 | 97.15 | 97.71 | 98.35| 98.38 |
| 2  | 98.45 | 80.06 | 98.31 | 98.50| 98.43 | 98.22 | 92.18 | 96.16 | 97.00 | 95.18 | 98.32| 96.81 |
| 3  | 98.17 | 81.02 | 98.23 | 98.12| 98.40 | 98.24 | 87.99 | 96.89 | 96.62 | 87.61 | 98.32| 96.09 |
| 4  | 97.61 | 81.81 | 98.03 | 96.46| 98.43 | 97.98 | 82.51 | 89.23 | 96.69 | 74.20 | 93.23| 89.72 |
| 5  | 95.67 | 82.44 | 97.99 | 81.90| 98.42 | 97.76 | 76.70 | 88.31 | 96.86 | 60.10 | 92.38| 85.06 |
| 6  | 90.94 | 83.02 | 97.77 | 50.28| 98.45 | 97.31 | 71.22 | 73.84 | 97.50 | 47.16 | 91.27| 67.45 |
| 7  | 82.12 | 83.55 | 97.48 | 17.66| 98.43 | 96.63 | 67.01 | 72.08 | 97.18 | 35.23 | 89.10| 58.05 |
| 8  | 74.65 | 83.98 | 97.15 | 9.99 | 98.46 | 95.48 | 62.95 | 58.16 | 96.61 | 25.73 | 87.72| 36.32 |
| 9  | 63.03 | 84.34 | 96.74 | 9.74 | 98.44 | 93.78 | 58.19 | 56.96 | 96.80 | 19.64 | 85.37| 30.26 |
| 10 | 49.01 | 85.07 | 96.11 | 9.74 | 98.44 | 90.68 | 54.35 | 48.64 | 96.35 | 15.97 | 82.81| 17.89 |

where, ideally, the different levels would results in similar accuracies.
### Figures

The *figures* sub-directory contains code used to generate figures.


