# Qserv_models

The current respository contains three different approaches of quantum error prediction. In this implementation, we make use of the quantum circuit image embeddings in the [Qserv dataset](https://github.com/Jongarde/Qserv_dataset), which porvides us with the image representations of 35000 different circuits. The presented code gathers the reading, training and evaluation phases for the regression problem that is the quantum error prediction, using the aforementioned dataset.

## Models: Training phase

As for the followed approaches, we make use of the models of:

➥ [ResNet](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py): To train the ResNet model (ResNet-50), we modify the last layer (fully connected layer), in order to transform it so that it results in only one output, in accordance to our regression problems.  
➥ [Vit-RGTS](https://github.com/kyegomez/Vit-RGTS): Furtherly explained in [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)  
➥ [DinoV2](https://github.com/facebookresearch/dinov2): Furtherly explained in [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)

## Results: Evaluation phase
