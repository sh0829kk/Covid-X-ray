# Covid-X-ray

:sparkles:  Author: Keke Lin


*Performs discrete, real, and gentle COVID-19 prediction model and patients' Follow-up condition model under DenseNet-121 and Vision Transformer*.

:octocat:

This repository contains:

1. The train and predict code of DenseNet [DenseNet_train.py](DenseNet_train.py), [DenseNet_predict.py](DenseNet_predict.py) and the train and predict code of Vision Transformer[ViT_train.py](ViT_train.py) and [ViT_predict.py](ViT_predict.py) to show how the program can be used.
2. A data file [raw_dataset](raw_dataset) to restore the data set.
3. The path of train data and test data, and how we unify them in [processed-data](processed-data).


   
## Model Constructure
The overview of our model is constructed as follows
![Compare Plots](/Figure3.png)
![Compare Plots](/ViT.png)

## Training Process
The training loss and comparision between DenseNet121 and Vision Transformer is constructed as follows
![Compare Plots](/TrainLoss.png)
