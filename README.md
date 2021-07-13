# CNN-pixel-classification [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of contents
  * [Introduction](#introduction)
  * [Installation](#installation)
  * [Pixel classification tools](#pixel-classification-tools)
  * [Example usage](#example-usage)
    * [Training a new model and pixel classifying data](#training-a-new-model-and-pixel-classifying-data)
    * [Using a pre-trained model to pixel classify your own data](#using-a-pre-trained-model-to-pixel-classify-your-own-data)
  * [Individual functions](#individual-functions)
    * [augmentation.py](#augmentationpy)
    * [train.py](#trainpy)
    * [Visualizing prediction quality as a function of training time (predict_lapse.py)](#visualizing-prediction-quality-as-a-function-of-training-time-predict_lapsepy)
    * [predict.py](#predictpy)
- - - - 

&nbsp;

# Introduction

This package uses the [Segmentation Models Pytorch](https://github.com/qubvel/segmentation_models.pytorch "Segmentation Models Pytorch") toolkit and is geared towards pixel classification of cell edges within microscopy data of epithelial tissues. Pixel classification is a prerequesite step for segmentation and detection of epithelial cells within this type of microscopy data, which can be completed using our other github repository, [eye-patterning](https://github.com/K-D-Gallagher/eye-patterning). We are included a [pre-trained model](#pixel-classifying-new-data-using-pre---trained-model) that can be used to quickly segment your own data; this model is trained on epithelial tissue where cell edges have been labeled with fluorescent protein fushion tags and where images were collected using laser scanning confocal microscopy. We also provide the tools necessary to [train your own model](#training-new-model-and-predicting-on-data) from stratch and use this to pixel classify your own data.

&nbsp;

# Installation
In terminal, navigate to the folder where you would like to locally install files and run the following commands. It will clone this repository, creating a folder called 'CNN-pixel-classification', and will install the necessary python requirements. Note, this code is compatible with Python 3.7.

``` shell script

$ git clone https://github.com/K-D-Gallagher/CNN-pixel-classification.git
$ pip3 install -r /path/to/CNN-pixel-classification/requirements.txt 

```

&nbsp;

# Pixel classification tools

Our package uses the [Segmentation Models Pytorch](https://github.com/qubvel/segmentation_models.pytorch "Segmentation Models Pytorch") package in order to provide a range of CNN architectures and pre-trained encoders, facilitating the discovery and training of the most accurate pixel classification model. By using pre-trained encoders that have been trained on vastly larger datasets than our own, paired with un-trained decoders that can trained to classify pixels in our epithelial data, we are able to achieve higher pixel classification accuracy than could be obtained without using transfer-learning. Segmentation Models Pytorch provides the following architectures and encoders:

#### Architectures:

Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, PAN, DeepLabV3, DeepLabV3+

#### Encoders:

ResNet, ResNeXt, ResNeSt, Res2Ne(X)t, RegNet(x/y), GERNet, SE-Net, SK-Net, SK-ResNe(X)t, DenseNet, Inception, EfficientNet, MobileNet, DPN, VGG

&nbsp;

# Example usage

## Training a new model and pixel classifying data

The following code illustrates the order of operations for training your own model from scratch with your own training data. Note, in order to do this, you will need at least ~100 images and corresponding 'ground truth' labels - i.e. binary images of the same dimensionality as the corresponding raw images where 0s correspond to cell interiors / background padding and 1s correspond to cell edges. A more detailed explanation of each python function can be found below.

``` shell script
# Create a static library of augmented images
> python augmentation.py -imf images -msf masks -a 4 -imfn train_images -mskfn train_masks -s 768 -gs True

# Train the model with the augmented and original images. The UNet++ architecture and inceptionv4 encoder resulted in the most accurate segmentation according to our tests.
> python train.py -e 200 -b 4 -cp Segmentation_test/ -fn train_images/ -mf train_masks/ -en resnet18 -wt imagenet -a unetplusplus

# Monitor the training using Tensorboard
> python tensorboard --logdir=Segmentation_test

# Use predict lapse to determine which epoch produced the best results
> python predict_lapse.py -f Segmentation_test -n test_folder -en resnet18 -wt imagenet -a unetplusplus

# Make predictions using the trained model 
> python predict.py -m Segmentation_test/CP_epoch11.pth -i images/ -t 0.1 -en resnet18 -wt imagenet -a unetplusplus


```

&nbsp;

## Using a pre-trained model to pixel classify your own data

The following code illustrates the order of operations for using our provided pre-trained model in order to pixel classify your own images. There are two requirements for your images: 1) they should be 8-bit and 2) they should be 768 x 768 pixels.

&nbsp;

# Individual functions

## augmentation.py

Augmentation allows you to artificially expand your training library volume by creating copies of all the training images that are randomly perturbed in defined ways, such as through rotations, shears, scale change, and the introduction of noise. Generally, augmentation can improve model performance because the accuracy of CNNs scales with training data volume. Additionally, augmentation can improve the generalizability of your model (it's ability to predict on out of sample data) by increasing the variation in your training data library. We chose to create a static library of augmented images, rather than augmenting during the training process. Therefore, this function should be used prior to the training function.

``` shell script

> python augmentation.py -h
usage: augmentation.py [-h] [--image-folder IMAGE_FOLDER]
                       [--mask-folder MASK_FOLDER] [--aug-size AUG_SIZE]
                       [--im-folder-nm IM_FOLDER_NM]
                       [--msk-folder-nm MSK_FOLDER_NM] [--scale SCALE]
                       [--grayscale]

Create a new augmented dataset

optional arguments:
  -h, --help            show this help message and exit
  --image-folder IMAGE_FOLDER, -imf IMAGE_FOLDER
                        Path to image folder (default: None)
  --mask-folder MASK_FOLDER, -msf MASK_FOLDER
                        Path to mask folder (default: None)
  --aug-size AUG_SIZE, -a AUG_SIZE
                        How many times to augment the original image folder
                        (default: None)
  --im-folder-nm IM_FOLDER_NM, -imfn IM_FOLDER_NM
                        Name for new augmented image folder (default: None)
  --msk-folder-nm MSK_FOLDER_NM, -mskfn MSK_FOLDER_NM
                        Name for new augmented mask folder (default: None)
  --scale SCALE, -s SCALE
                        Dimension to scale ass the images (default: 768)
  --grayscale, -gs      Make all the augmented images grayscale (default:
                        False)
```

&nbsp;

## train.py

```shell script

> python train.py -h     
usage: train.py [-h] [-e E] [-b [B]] [-l [LR]] [-f LOAD] [-s SCALE] [-v VAL] [--classes CLASSES] [--in-channels IN_CHANNELS] [--device DEVICE]
                [-cp CHECKPOINT] [-fn FILE] [-en ENCODER] [-wt WEIGHT]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  -e E, --epochs E      Number of epochs (default: 5)
  -b [B], --batch-size [B]
                        Batch size (default: 1)
  -l [LR], --learning-rate [LR]
                        Learning rate (default: 0.0001)
  -f LOAD, --load LOAD  Load model from a .pth file (default: False)
  -s SCALE, --scale SCALE
                        Downscaling factor of the images (default: 0.5)
  -v VAL, --validation VAL
                        Percent of the data that is used as validation (0-100) (default: 10.0)
  --classes CLASSES, -c CLASSES
                        Model output channels (default: 1)
  --in-channels IN_CHANNELS, -ic IN_CHANNELS
                        Model input channels (default: 1)
  --device DEVICE, -d DEVICE
                        Select device (default: cuda:0)
  -cp CHECKPOINT, --checkpoint CHECKPOINT
                        Name folder for checkpoints (default: checkpoints/)
  -fn FILE, --file FILE
                        Name folder for images (default: None)
  -en ENCODER, --encoder ENCODER
                        Name of encoder (default: resnet34)
  -wt WEIGHT, --weight WEIGHT
                        Encoder weights (default: None)
                                              
```

&nbsp;

## Visualizing prediction quality as a function of training time (predict_lapse.py)

We found that looking at the loss curve and dice coefficient alone was not always the best indicator of a model's accuracy. Therefore, we developed a piece of code that will visualize the output of the same image as it is pixel classified with all the training epochs of a model in the specified folder. By qualitatively evaluating these predictions, paired with evaluation of their loss and dice coefficient curves, you can select the best epoch of your trained model to use for batch predicting the rest of your data.

``` shell script 

> python predict_lapse.py 
usage: predict_lapse.py [-h] --folder FOLDER [-en ENCODER] [-wt WEIGHT] [-a ARCHITECTURE] --input INPUT [INPUT ...] [-n NAME]
predict_lapse.py: error: the following arguments are required: --folder/-f, --input/-i
nathanburg@Nathans-MBP CNN-Architecture-Comparison- % python3 predict_lapse.py -h
usage: predict_lapse.py [-h] --folder FOLDER [-en ENCODER] [-wt WEIGHT] [-a ARCHITECTURE] --input INPUT [INPUT ...] [-n NAME]

Visualize predictions at each epoch

optional arguments:
  -h, --help            show this help message and exit
  --folder FOLDER, -f FOLDER
                        path to model folder (default: None)
  -en ENCODER, --encoder ENCODER
                        Name of encoder (default: resnet34)
  -wt WEIGHT, --weight WEIGHT
                        Encoder weights (default: None)
  -a ARCHITECTURE, --architecture ARCHITECTURE
                        Name of architecture (default: None)
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        filenames of input images (default: None)
  -n NAME, --name NAME  Name for image folder (default: None)
```

&nbsp;

## predict.py

``` shell script

> python predict.py -h
usage: predict.py [-h] [--model FILE] --input INPUT [INPUT ...] [--output INPUT [INPUT ...]] [--viz] [--no-save] [--mask-threshold MASK_THRESHOLD]
                  [--scale SCALE] [--classes CLASSES] [--in-channels IN_CHANNELS] [--device DEVICE] [-en ENCODER] [-wt WEIGHT] [-a ARCHITECTURE]

Predict masks from input images

optional arguments:
  -h, --help            show this help message and exit
  --model FILE, -m FILE
                        Specify the file in which the model is stored (default: MODEL.pth)
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        filenames of input images (default: None)
  --output INPUT [INPUT ...], -o INPUT [INPUT ...]
                        Filenames of ouput images (default: None)
  --viz, -v             Visualize the images as they are processed (default: False)
  --no-save, -n         Do not save the output masks (default: False)
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        Minimum probability value to consider a mask pixel white (default: None)
  --scale SCALE, -s SCALE
                        Scale factor for the input images (default: 0.5)
  --classes CLASSES, -c CLASSES
                        Model output channels (default: 1)
  --in-channels IN_CHANNELS, -ic IN_CHANNELS
                        Model input channels (default: 1)
  --device DEVICE, -d DEVICE
                        Select device (default: cuda:0)
  -en ENCODER, --encoder ENCODER
                        Name of encoder (default: resnet34)
  -wt WEIGHT, --weight WEIGHT
                        Encoder weights (default: None)
  -a ARCHITECTURE, --architecture ARCHITECTURE
                        Name of architecture (default: None)
                        
  ```
  
  

  
