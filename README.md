# Qserv_models

The current respository contains three different approaches of quantum error prediction. In this implementation, we make use of the quantum circuit **image embeddings** in the [Qserv dataset](https://github.com/Jongarde/Qserv_dataset), which porvides us with the image representations of 35,000 different circuits. The presented code gathers the reading, training and evaluation phases for the regression problem that is the quantum error prediction, using the aforementioned dataset.

## Models: Training phase

As for the followed approaches, we make use models such as:

➥ [ResNet](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py): To train the ResNet model (ResNet-50), we modify the last layer (fully connected layer), in order to transform it so that it results in only one output, in accordance to our regression problem.  
➥ [Vit-RGTS](https://github.com/kyegomez/Vit-RGTS): Furtherly explained in [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)  
➥ [DinoV2](https://github.com/facebookresearch/dinov2): Furtherly explained in [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)

## Results: Evaluation phase

The results cover multiple metrics to ensure the robustness of the model evaluation. Metrics such as MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), MAPE (Mean Absolute Percentage Error) and R2 were used in order to evaluate the precision of our models, as shown in the following table.

|            Modelo           |   MAE  |  RMSE  |  MAPE  |   R2   |
|:---------------------------:|:------:|:------:|:------:|:------:|
| ResNet-50 (No weights)      | 0,0403 | 0,0675 | 1,8369 | 0,8828 |
| ResNet-50 (Default weights) | 0,0338 | 0,0622 | 1,2043 | 0,9003 |
| Vit-RGTS                    | 0,0364 | 0,0658 | 1,1132 | 0,8887 |
| DinoV2                      | 0,0379 | 0,0656 | 1,4440 | 0.8891 |
