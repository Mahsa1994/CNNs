# CNNs
The CNNs repository is a collection of python codes used for training deep convolutional neural networks. The repository's primary focus is on implementing and experimenting with convolutional neural networks (CNNs) using Python.

## How to train?
Before running the `main.py` script in the CNNs repository, you'll need to specify some hyperparameters within the script. `main.py` is the primary training script for the CNN models, responsible for training the model on a specific dataset, validating the model with specified validation data, and saving the trained model to a specified location. Additionally, the script allows for hyperparameter tuning, enabling you to adjust the model's parameters and fine-tune it to get better results.

### Prepare your own data-set
To start on your own dataset, first you need to prepare it. Just split your data into the train and validation set and build a table from them.
The table consists of two columns:

+ **image_name:** the path to the image.
+ **tags:** the ground-truth label corresponding to the image.

Here is an example for the `train.csv` and `val.csv` files:

<table>
<tr>
<th>image_name</th>
<th>tags</th>
</tr>
<tr>
<td>
./dataset/train_image1.png
</td>
<td>
label1
</td>
</tr>
<tr>
<td>
./dataset/train_image2.png
</td>
<td>
label2
</td>
</tr>
<tr>
<td>
.....
</td>
<td>
.....
</td>
</tr>
</table>

<table>
<tr>
<th>image_name</th>
<th>tags</th>
</tr>
<tr>
<td>
./dataset/val_image1.png
</td>
<td>
label1
</td>
</tr>
<tr>
<td>
./dataset/val_image2.png
</td>
<td>
label2
</td>
</tr>
<tr>
<td>
.....
</td>
<td>
.....
</td>
</tr>
</table>


### Set hyper-parameters
Based on your data and the desired output, you need to set hyper-parameters. Here is the definition of some parameters:
+ **save_dir:** all checkpoints and logs are saved in this directory.
+ **loss_function:** there are two main loss functions that you can use during the training process.
+ **focalLoss_alpha and focalLoss_gamma:** if you set the loss function to `Focal`, then you need to set these params.
+ **fineTune_enable:** if you want to fine-tune an existing network, you need to set it True.
+ **fineTune_layers:** when you set the `fineTune_enable` True you need to define the layers which you want to train them. In the other words, unfreeze these layers.
+ **fineTune_batchNorm:** if the model has `BatchNorm` layers, you can unfreeze them using this param.

### Define the Network
The `model` could be an existing network or your own model. Here is an example of how to load the pretrained ImageNet model:

```
model = models.resnet50(pretrained=True)
number_feats = model.fc.in_features
model.fc = nn.Linear(in_features=number_feats, out_features=number_of_classes)
```

You may load your own model using below code:

```
my_model = torch.load('./snapshots/model.pth')
model = my_model['state_dict'].module ## when your model has a state_dict 
```

### Start training
Finally, start to train your model using this command:

```python main.py```

To monitor the training process use the log file in the `save_dir`:

```tensorboard --logdir=/path/to/event-file```

### Future work
- [ ] Add inference code
- [ ] Add new models
- [ ] Add normalization
- [ ] Add ONNX convertor

