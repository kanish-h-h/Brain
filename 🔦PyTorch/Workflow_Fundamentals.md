The essence of machine learning and deep learning is to take some data from the past, build an algorithm (like a neural network) to discover patterns in it and use the discoverd patterns to predict the future.



![](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/01_a_pytorch_workflow.png)

| Topic                                               | Contents                                                                                                      |
|-----------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| 1. Getting data ready                              | Data can be almost anything but to get started we're going to create a simple straight line                  |
| 2. Building a model                                | Here we'll create a model to learn patterns in the data, we'll also choose a loss function, optimizer and build a training loop.                                      |
| 3. Fitting the model to data (training)            | We've got data and a model, now let's let the model (try to) find patterns in the (training) data.            |
| 4. Making predictions and evaluating a model (inference) | Our model's found patterns in the data, let's compare its findings to the actual (testing) data.            |
| 5. Saving and loading a model                      | You may want to use your model elsewhere, or come back to it later, here we'll cover that.                    |
| 6. Putting it all together                         | Let's take all of the above and combine it.                                                                  |



```python
what_were_covering = {1: "data (prepare and load)",
    2: "build model",
    3: "fitting the model to data (training)",
    4: "making predictions and evaluating a model (inference)",
    5: "saving and loading a model",
    6: "putting it all together"
}
```

We're going to get `torch`, `torch.nn` (`nn` stands for neural network and this package contains the building blocks for creating neural networks in PyTorch) and `matplotlib`.




```python
import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt

# Check PyTorch version
torch.__version__
```




    '2.2.1+cu121'



## 1. Data (preparing and loading)

![](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/01-machine-learning-a-game-of-two-parts.png)

Machine learning is a game of two parts:

1. Turn your data, whatever it is, into numbers (a representation).
2. Pick or build a model to learn the representation as best as possible.

We'll use [linear regression](https://en.wikipedia.org/wiki/Linear_regression) to create the data with known parameters (things that can be learned by a model) and then we'll use PyTorch to see if we can build model to estimate these parameters using [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent).




```python
# create known parameters
weight = 0.7
bias = 0.3

# create data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

X[:10], y[:10]
```




    (tensor([[0.0000],
             [0.0200],
             [0.0400],
             [0.0600],
             [0.0800],
             [0.1000],
             [0.1200],
             [0.1400],
             [0.1600],
             [0.1800]]),
     tensor([[0.3000],
             [0.3140],
             [0.3280],
             [0.3420],
             [0.3560],
             [0.3700],
             [0.3840],
             [0.3980],
             [0.4120],
             [0.4260]]))



`X` -> Features
`y` -> labels

### Split data into training and test sets


```python
# Create train/test split
train_split = int(0.8 * len(X))   # 80% of data used for training, 20% for testing
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

len(X_train), len(y_train), len(X_test), len(y_test)
```




    (40, 40, 10, 10)



The model we create is going to try and learn the relationship between `X_train` & `y_train` and then we will evaluate what it learns on `X_test` and `y_test`.




```python
def plot_predictions(train_data=X_train,
                      train_labels=y_train,
                      test_data=X_test,
                      test_labels=y_test,
                      predictions=None):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(10,7))
  plt.scatter(train_data, train_labels, c="b", s=4, label="Training Data")
  plt.scatter(test_data, test_labels, c="g", s=4, label="Testing Data")

  if predictions is not None:
    plt.scatter(test_data, predictions, c="r", s=4, label="Prediction")

  plt.legend(prop={"size": 14})
```


```python
plot_predictions();
plt.savefig('Linear_prediction.png')
```


    
![png](Linear_prediction.png)
    


## 2. Build Model

Let's replicate a standard linear regression model using pure PyTorch.




```python
# Create a Linear Regression model class
class LinearRegressionModel(nn.Module):  # <- almost everything in PyTorch is a nn.Module (think of this as neural network lego blocks)
  def __init__(self):
    super().__init__()
    self.weights = nn.Parameter(torch.randn(1, dtype=torch.float),
                                requires_grad=True)
    self.bias = nn.Parameter(torch.randn(1, dtype=torch.float),
                             requires_grad=True)

    # Forward defines the computation in the model
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.weights * x + self.bias
```

### PyTorch model building essentials

PyTorch has four (give or take) essential modules you can use to create almost any kind of neural network you can imagine.

They are `torch.nn`, `torch.optim`, `torch.utils.data.Dataset` and `torch.utils.data.DataLoader`.

| PyTorch module       | What does it do?                                                                                                                                                                         |
|----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `torch.nn`             | Contains all of the building blocks for computational graphs (essentially a series of computations executed in a particular way).                                                        |
| `torch.nn.Parameter`   | Stores tensors that can be used with `nn.Module`. If `requires_grad=True` gradients (used for updating model parameters via [gradient descent](https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html)) are calculated automatically, this is often referred to as "autograd". |
| `torch.nn.Module`      | The base class for all neural network modules, all the building blocks for neural networks are subclasses. If you're building a neural network in PyTorch, your models should subclass `nn.Module`. Requires a `forward()` method be implemented. |
| `torch.optim`          | Contains various optimization algorithms (these tell the model parameters stored in nn.Parameter how to best change to improve gradient descent and in turn reduce the loss).             |
| `def forward()`        | All `nn.Module` subclasses require a `forward()` method, this defines the computation that will take place on the data passed to the particular `nn.Module` (e.g. the linear regression formula above).            |


If the above sounds complex, think of like this, almost everything in a PyTorch neural network comes from `torch.nn`,

- `nn.Module` contains the larger building blocks (layers)
- `nn.Parameter` contains the smaller parameters like weights and biases (put these together to make `nn.Module`(s))
- `forward()` tells the larger blocks how to make calculations on inputs (tensors full of data) within `nn.Module`(s)
- `torch.optim` contains optimization methods on how to improve the parameters within `nn.Parameter` to better represent input data

![](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/01-pytorch-linear-model-annotated.png)

### Checking the contents of a PyTorch Model

 let's create a model instance with the class we've made and check its parameters using [`.parameters()`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.parameters)`.


```python
torch.manual_seed(42)

# Create an instance of a model
model_0 = LinearRegressionModel()

# check the nn.Parameter(s) within nn.Module we created
list(model_0.parameters())
```




    [Parameter containing:
     tensor([0.3367], requires_grad=True),
     Parameter containing:
     tensor([0.1288], requires_grad=True)]



We can also get the state (what the model contains) of the model using [`.state_dict()`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.state_dict).




```python
# List named parameters
model_0.state_dict()
```




    OrderedDict([('weights', tensor([0.3367])), ('bias', tensor([0.1288]))])



### Making predictions using `torch.inference_mode()`


```python
# Make predictions with model
with torch.inference_mode():
    y_preds = model_0(X_test)

# Note: in older PyTorch code you might also see torch.no_grad()
# with torch.no_grad():
#   y_preds = model_0(X_test)
```

As the name suggests, `torch.inference_mode()` is used when using a model for inference (making predictions).

`torch.inference_mode()` turns off a bunch of things (like gradient tracking, which is necessary for training but not for inference) to make forward-passes (data going through the `forward()` method) faster.


```python
# Check the predictions
print(f"Number of testing samples: {len(X_test)}")
print(f"Number of predictions made: {len(y_preds)}")
print(f"Predicted values:\n{y_preds}")
```

    Number of testing samples: 10
    Number of predictions made: 10
    Predicted values:
    tensor([[0.3982],
            [0.4049],
            [0.4116],
            [0.4184],
            [0.4251],
            [0.4318],
            [0.4386],
            [0.4453],
            [0.4520],
            [0.4588]])



```python
plot_predictions(predictions=y_preds)
plt.savefig('y_preds.png')
```


    
![png](y_preds.png)
    



```python
y_test - y_preds
```




    tensor([[0.4618],
            [0.4691],
            [0.4764],
            [0.4836],
            [0.4909],
            [0.4982],
            [0.5054],
            [0.5127],
            [0.5200],
            [0.5272]])



#  Train Model

To fix that, we can update its internal parameters (I also refer to parameters as patterns), the `weights` and `bias` values we set randomly using `nn.Parameter()` and `torch.randn()` to be something that better represents the data.



## Creating a loss function and optimizer in PyTorch

For our model to update its parameters on its own, we'll need to add a few more things to our recipe.And that's a **loss function** as well as an optimizer.

| Function      | What does it do?                                                                                       | Where does it live in PyTorch?                    | Common values                                                                                 |
|---------------|--------------------------------------------------------------------------------------------------------|---------------------------------------------------|-----------------------------------------------------------------------------------------------|
| Loss function| Measures how wrong your model's predictions (e.g., `y_preds`) are compared to the truth labels (e.g., `y_test`). Lower values are better.   | PyTorch provides built-in loss functions in `torch.nn`. | Mean Absolute Error (MAE) for regression problems (`torch.nn.L1Loss()`), Binary Cross Entropy for binary classification problems (`torch.nn.BCELoss()`) |
| Optimizer     | Guides your model on how to update its internal parameters to minimize the loss.                         | Various optimization function implementations are available in `torch.optim`.     | Stochastic Gradient Descent (`torch.optim.SGD()`), Adam optimizer (`torch.optim.Adam()`)   |


Let's create a loss function and an optimizer we can use to help improve our model.

Depending on what kind of problem you're working on will depend on what loss function and what optimizer you use.

However, there are some common values, that are known to work well such as the SGD (stochastic gradient descent) or Adam optimizer. And the MAE (mean absolute error) loss function for regression problems (predicting a number) or binary cross entropy loss function for classification problems (predicting one thing or another).

For our problem, since we're predicting a number, let's use MAE (which is under `torch.nn.L1Loss()`) in PyTorch as our loss function.

![](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/01-mae-loss-annotated.png)

Mean absolute error (MAE, in PyTorch: `torch.nn.L1Loss`) measures the absolute difference between two points (predictions and labels) and then takes the mean across all examples.

And we'll use SGD, `torch.optim.SGD(params, lr)` where:

- `params` is the target model parameters you'd like to optimize (e.g. the `weights` and `bias` values we randomly set before).
- `lr` is the **learning rate** you'd like the optimizer to update the parameters at, higher means the optimizer will try larger updates (these can sometimes be too large and the optimizer will fail to work), lower means the optimizer will try smaller updates (these can sometimes be too small and the optimizer will take too long to find the ideal values). The learning rate is considered a `hyperparameter` (because it's set by a machine learning engineer). Common starting values for the learning rate are `0.01`, `0.001`, `0.0001`, however, these can also be adjusted over time (this is called learning rate scheduling).


```python
# Create the loss function
loss_fn = nn.L1Loss() # MAE loss is same as L1Loss

# Create the optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), # parameters of target model to optimize
                            lr=0.01) # learning rate (how much the optimizer should change parameters at each step, higher=more (less stable), lower=less (might take a long time))
```

## Creating an optimization loop in PyTorch

PyTorch Training loop

| Number | Step name                | What does it do?                                                                                                       | Code example                 |
|--------|--------------------------|------------------------------------------------------------------------------------------------------------------------|------------------------------|
| 1      | Forward pass             | The model goes through all of the training data once, performing its `forward()` function calculations.              | `model(x_train)`             |
| 2      | Calculate the loss       | The model's outputs (predictions) are compared to the ground truth and evaluated to see how wrong they are.            | `loss = loss_fn(y_pred, y_train)` |
| 3      | Zero gradients           | The optimizer's gradients are set to zero (they are accumulated by default) so they can be recalculated for the specific training step. | `optimizer.zero_grad()`      |
| 4      | Perform backpropagation on the loss | Computes the gradient of the loss with respect to every model parameter to be updated (each parameter with `requires_grad=True`). This is known as backpropagation, hence "backwards". | `loss.backward()`            |
| 5      | Update the optimizer (gradient descent) | Update the parameters with `requires_grad=True` with respect to the loss gradients in order to improve them.           | `optimizer.step()`           |


![](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/01-pytorch-training-loop-annotated.png)

> Note: The above is just one example of how the steps could be ordered or described. With experience you'll find making PyTorch training loops can be quite flexible.
>
> And on the ordering of things, the above is a good default order but you may see slightly different orders. Some rules of thumb:
>
>  - Calculate the loss (`loss = ...`) before performing backpropagation on it (`loss.backward()`).
>  - Zero gradients (`optimizer.zero_grad()`) before stepping them (`optimizer.step()`).
>  - Step the optimizer (`optimizer.step()`) after performing backpropagation on the loss (`loss.backward()`).


PyTorch Testing loop

| Number | Step name                        | What does it do?                                                                                                       | Code example        |
|--------|----------------------------------|------------------------------------------------------------------------------------------------------------------------|---------------------|
| 1      | Forward pass                     | The model goes through all of the evaluation data once, performing its `forward()` function calculations.             | `model(x_test)`     |
| 2      | Calculate the loss               | The model's outputs (predictions) are compared to the ground truth and evaluated to see how wrong they are.            | `loss = loss_fn(y_pred, y_test)` |
| 3      | Calculate evaluation metrics     | Alongside the loss value, you may want to calculate other evaluation metrics such as accuracy on the test set.        | Custom functions    |



Notice the testing loop doesn't contain performing backpropagation (`loss.backward()`) or stepping the optimizer (`optimizer.step()`), this is because no parameters in the model are being changed during testing, they've already been calculated. For testing, we're only interested in the output of the forward pass through the model.

![](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/01-pytorch-testing-loop-annotated.png)


Let's put all of the above together and train our model for 100 epochs (forward passes through the data) and we'll evaluate it every 10 epochs.




```python
# setting seed
torch.manual_seed(42)

# set number of epochs
epochs = 100

# create empty loss lists to track value
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
  ### Training

  # put model in training mode
  model_0.train()

  # 1. Forward pass using forward()
  y_pred = model_0(X_train)
  # print(y_pred)

  # 2. calculae the loss
  loss = loss_fn(y_pred, y_train)

  # 3. zero grad of the optimizer
  optimizer.zero_grad()

  # 4. loss backward
  loss.backward()

  # 5. progress the optimizer
  optimizer.step()

  ### Testing

  # Put the model in evaluation mode
  model_0.eval()

  with torch.inference_mode():
    # 1. forward pass on test data
    test_pred = model_0(X_test)

    # 2. calculate loss on test data
    test_loss = loss_fn(test_pred, y_test.type(torch.float))

    # print out what's happening
    if epoch % 10 == 0:
      epoch_count.append(epoch)
      train_loss_values.append(loss.detach().numpy())
      test_loss_values.append(test_loss.detach().numpy())
      print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss}")
```

    Epoch: 0 | MAE Train Loss: 0.31288138031959534 | MAE Test Loss: 0.48106518387794495
    Epoch: 10 | MAE Train Loss: 0.1976713240146637 | MAE Test Loss: 0.3463551998138428
    Epoch: 20 | MAE Train Loss: 0.08908725529909134 | MAE Test Loss: 0.21729660034179688
    Epoch: 30 | MAE Train Loss: 0.053148526698350906 | MAE Test Loss: 0.14464017748832703
    Epoch: 40 | MAE Train Loss: 0.04543796554207802 | MAE Test Loss: 0.11360953003168106
    Epoch: 50 | MAE Train Loss: 0.04167863354086876 | MAE Test Loss: 0.09919948130846024
    Epoch: 60 | MAE Train Loss: 0.03818932920694351 | MAE Test Loss: 0.08886633068323135
    Epoch: 70 | MAE Train Loss: 0.03476089984178543 | MAE Test Loss: 0.0805937647819519
    Epoch: 80 | MAE Train Loss: 0.03132382780313492 | MAE Test Loss: 0.07232122868299484
    Epoch: 90 | MAE Train Loss: 0.02788739837706089 | MAE Test Loss: 0.06473556160926819



```python
# Plot the loss curves
plt.plot(epoch_count, train_loss_values, label="Train Loss")
plt.plot(epoch_count, test_loss_values, label="Test Loss")
plt.title("Training and Test loss curves")
plt.xlabel("Loss")
plt.ylabel("Epochs")
plt.legend();
plt.savefig('linear_regression_loss.png')
```


    
![png](linear_regression_loss.png)
    



```python
# find our model's learned parameters
print(f"The model learned the following values for weight and bias: ")
print(model_0.state_dict())
print("\nAnd the original values for weights and bias are: ")
print(f"weights: {weight}, bias: {bias}")
```

    The model learned the following values for weight and bias: 
    OrderedDict([('weights', tensor([0.5784])), ('bias', tensor([0.3513]))])
    
    And the original values for weights and bias are: 
    weights: 0.7, bias: 0.3


# 4. Making Predictions with trained PyTorch model (inference)

There are three things to remember when making predictions (also called performing inference) with a PyTorch model:

1. Set the model in evaluation mode (`model.eval()`).
2. Make the predictions using the inference mode context manager (with `torch.inference_mode(): ...`).
3. All predictions should be made with objects on the same device (e.g. data and model on GPU only or data and model on CPU only).

The first two items make sure all helpful calculations and settings PyTorch uses behind the scenes during training but aren't necessary for inference are turned off (this results in faster computation). And the third ensures that you won't run into cross-device errors.


```python
# 1. Set the model in evaluation mode
model_0.eval()

# 2. Setup the inference mode
with torch.inference_mode():
  # 3. Make sure the calculations are done with the model and data on the same device
  # in our case, we haven't setup device-agnostic code yet so our data and model are
  # on the CPU by default.
  # model_0.to(device)
  # X_test = X_test.to(device)
  y_preds = model_0(X_test)
y_preds
```




    tensor([[0.8141],
            [0.8256],
            [0.8372],
            [0.8488],
            [0.8603],
            [0.8719],
            [0.8835],
            [0.8950],
            [0.9066],
            [0.9182]])




```python
plot_predictions(predictions=y_preds)
plt.savefig('lr_predicted.png')
```


    
![png](lr_predicted.png)
    


# 5. Saving and Reloding a PyTorch model

| PyTorch method                   | What does it do?                                                                                                                                    |
|----------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `torch.save`                       | Saves a serialized object to disk using Python's pickle utility. Models, tensors, and various other Python objects like dictionaries can be saved.   |
| `torch.load`                       | Uses pickle's unpickling features to deserialize and load pickled Python object files (e.g., models, tensors, or dictionaries) into memory.       |
| `torch.nn.Module.load_state_dict` | Loads a model's parameter dictionary (`model.state_dict()`) using a saved `state_dict()` object.                                                   |


### Saving a PyTorch model's `state_dict()`

The recommended way for saving and loading a model for inference (making predictions) is by saving and loading a model's `state_dict()`.

Let's see how we can do that in a few steps:

1. We'll create a directory for saving models to called models using Python's `pathlib` module.
2. We'll create a file path to save the model to.
3. We'll call `torch.save(obj, f)` where `obj` is the target model's `state_dict()` and `f` is the filename of where to save the model.

**Note:** It's common convention for PyTorch saved models or objects to end with `.pt` or `.pth`, like `saved_model_01.pth`.


```python
from pathlib import Path

# 1. Create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), # only saves model learned parameters
           f=MODEL_SAVE_PATH)
```

    Saving model to: models/01_pytorch_workflow_model_0.pth



```python
# Check the saved file path
!ls -l models/01_pytorch_workflow_model_0.pth
```

    -rw-r--r-- 1 root root 1680 Apr 21 14:58 models/01_pytorch_workflow_model_0.pth


### Loading a saved PyTorch model's `state_dict()`


```python
# insantiate a new instance of our model
loaded_model_0 = LinearRegressionModel()

# load the state_dict of our saved model
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
```




    <All keys matched successfully>




```python
# 1. put the loaded model into evaluation mode
loaded_model_0.eval()

# 2. use the inference mode
with torch.inference_mode():
  loaded_model_preds = loaded_model_0(X_test)  # perform a forward pass to loaded data
```


```python
# comparing previous model predictions with loaded model predictions
y_preds == loaded_model_preds
```




    tensor([[True],
            [True],
            [True],
            [True],
            [True],
            [True],
            [True],
            [True],
            [True],
            [True]])



# 6. Putting it all together


```python
import torch
from torch import nn
import matplotlib.pyplot as plt

torch.__version__
```




    '2.2.1+cu121'




```python
# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
```

    Using device: cpu


## 6.1 Data


```python
# create weight and bias
weight = 0.7
bias = 0.3

X = torch.arange(0, 1, 0.02).unsqueeze(dim=1)
y = weight * X + bias
X[:10], y[:10]
```




    (tensor([[0.0000],
             [0.0200],
             [0.0400],
             [0.0600],
             [0.0800],
             [0.1000],
             [0.1200],
             [0.1400],
             [0.1600],
             [0.1800]]),
     tensor([[0.3000],
             [0.3140],
             [0.3280],
             [0.3420],
             [0.3560],
             [0.3700],
             [0.3840],
             [0.3980],
             [0.4120],
             [0.4260]]))




```python
# split the data
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

len(X_train), len(y_train), len(X_test), len(y_test)
```




    (40, 40, 10, 10)




```python
# visualize the dataset
plot_predictions(X_train, y_train, X_test, y_test)
```


    
![png](Linear_prediction.png)
    


## 6.2 Building a PyTorch linear model

We'll create the same style of model as before except this time, instead of defining the weight and bias parameters of our model manually using `nn.Parameter()`, we'll use `nn.Linear(in_features, out_features)` to do it for us.

![](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/01-pytorch-linear-regression-model-with-nn-Parameter-and-nn-Linear-compared.png)


```python
# Subclass nn.Module to make a model
class LinearRegressionModelV2(nn.Module):
  def __init__(self):
    super().__init__()
    # use nn.Linear() for creating the model parameter
    self.linear_layer = nn.Linear(in_features=1,
                                  out_features=1)

    # define forward computation
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.linear_layer(x)

# seed
torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
model_1, model_1.state_dict()
```




    (LinearRegressionModelV2(
       (linear_layer): Linear(in_features=1, out_features=1, bias=True)
     ),
     OrderedDict([('linear_layer.weight', tensor([[0.7645]])),
                  ('linear_layer.bias', tensor([0.8300]))]))



Time to build a training and testing loop.

First we'll need a loss function and an optimizer.

Let's use the same functions we used earlier, `nn.L1Loss()` and `torch.optim.SGD()`.

We'll have to pass the new model's parameters (`model.parameters()`) to the optimizer for it to adjust them during training.

The learning rate of `0.01` worked well before too so let's use that again.




```python
# Check model device
next(model_1.parameters()).device
```




    device(type='cpu')




```python
# Set model to GPU if it's availalble, otherwise it'll default to CPU
model_1.to(device) # the device variable was set above to be "cuda" if available or "cpu" if not
next(model_1.parameters()).device
```




    device(type='cpu')



## 6.3 Training


```python
# Create loss function
loss_fn = nn.L1Loss()

# Create optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=0.01)
```


```python
# seed
torch.manual_seed(42)

# number of epochs
epochs = 1000

X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

for epoch in range(epochs):
  ### Training
  model_1.train()

  # 1. forward pass
  y_pred = model_1(X_train)

  # 2. calculate loss
  loss = loss_fn(y_pred, y_train)

  # 3. zero grad optimizer
  optimizer.zero_grad()

  # 4. loss backward
  loss.backward()

  # 5. step the optimizer
  optimizer.step()

  ### Testing
  model_1.eval()

  # 1. forward pass
  with torch.inference_mode():
    test_pred = model_1(X_test)

    # 2. calculate the loss
    test_loss = loss_fn(test_pred, y_test)

  if epoch % 10 == 0:
    print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")
```

    Epoch: 0 | Train loss: 0.5551779866218567 | Test loss: 0.5739762187004089
    Epoch: 10 | Train loss: 0.4399680495262146 | Test loss: 0.4392663538455963
    Epoch: 20 | Train loss: 0.3247582018375397 | Test loss: 0.30455657839775085
    Epoch: 30 | Train loss: 0.20954827964305878 | Test loss: 0.16984674334526062
    Epoch: 40 | Train loss: 0.09433844685554504 | Test loss: 0.03513689711689949
    Epoch: 50 | Train loss: 0.023886386305093765 | Test loss: 0.04784906655550003
    Epoch: 60 | Train loss: 0.0199567973613739 | Test loss: 0.04580312222242355
    Epoch: 70 | Train loss: 0.016517987474799156 | Test loss: 0.0375305712223053
    Epoch: 80 | Train loss: 0.013089170679450035 | Test loss: 0.029944902285933495
    Epoch: 90 | Train loss: 0.009653178043663502 | Test loss: 0.02167237363755703
    Epoch: 100 | Train loss: 0.006215679459273815 | Test loss: 0.014086711220443249
    Epoch: 110 | Train loss: 0.002787243574857712 | Test loss: 0.005814164876937866
    Epoch: 120 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 130 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 140 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 150 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 160 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 170 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 180 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 190 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 200 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 210 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 220 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 230 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 240 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 250 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 260 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 270 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 280 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 290 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 300 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 310 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 320 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 330 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 340 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 350 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 360 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 370 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 380 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 390 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 400 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 410 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 420 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 430 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 440 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 450 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 460 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 470 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 480 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 490 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 500 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 510 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 520 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 530 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 540 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 550 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 560 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 570 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 580 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 590 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 600 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 610 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 620 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 630 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 640 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 650 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 660 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 670 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 680 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 690 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 700 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 710 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 720 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 730 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 740 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 750 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 760 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 770 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 780 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 790 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 800 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 810 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 820 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 830 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 840 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 850 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 860 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 870 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 880 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 890 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 900 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 910 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 920 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 930 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 940 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 950 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 960 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 970 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 980 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904
    Epoch: 990 | Train loss: 0.0012645035749301314 | Test loss: 0.013801807537674904


Let's check the parameters our model has learned and compare them to the original parameters we hard-coded.




```python
# find our model's learning parameters
from pprint import pprint   # pprint = pretty print, see: https://docs.python.org/3/library/pprint.html

print("The model's weights and bias:")
pprint(model_1.state_dict())
print("\nAnd the original values of weights and bias are: ")
print(f"weights: {weight}, bias: {bias}")
```

    The model's weights and bias:
    OrderedDict([('linear_layer.weight', tensor([[0.6968]])),
                 ('linear_layer.bias', tensor([0.3025]))])
    
    And the original values of weights and bias are: 
    weights: 0.7, bias: 0.3


## 6.4 Making predictions


```python
# turn model into evaluation mode
model_1.eval()

# make prediction on test data
with torch.inference_mode():
  y_preds = model_1(X_test)
y_preds
```




    tensor([[0.8600],
            [0.8739],
            [0.8878],
            [0.9018],
            [0.9157],
            [0.9296],
            [0.9436],
            [0.9575],
            [0.9714],
            [0.9854]])




```python
plot_predictions(predictions=y_preds)
plt.savefig('predictionV2.png')
```


    
![png](predictionV2.png)
    


## 6.5 Saving and loading the model


```python
from pathlib import Path

# 1. create model directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. create model save path
MODEL_NAME = "01_pytorch_workflow_model_1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_1.state_dict(),
           f=MODEL_SAVE_PATH)
```

    Saving model to: models/01_pytorch_workflow_model_1.pth


Loading the model


```python
# Instantiate an instance of model
loaded_model_1 = LinearRegressionModelV2()

# load moidel
loaded_model_1.load_state_dict(torch.load(MODEL_SAVE_PATH))

# put model to target device
loaded_model_1.to(device)

print(f"Loaded model:\n {loaded_model_1}")
print(f"Model on device:\n {next(loaded_model_1.parameters()).device}")
```

    Loaded model:
     LinearRegressionModelV2(
      (linear_layer): Linear(in_features=1, out_features=1, bias=True)
    )
    Model on device:
     cpu


Now we can evaluate the loaded model to see if its predictions line up with the predictions made prior to saving.




```python
# Evaluate loadedd model
loaded_model_1.eval()
with torch.inference_mode():
  loaded_model_1_preds = loaded_model_1(X_test)
y_preds == loaded_model_1_preds
```




    tensor([[True],
            [True],
            [True],
            [True],
            [True],
            [True],
            [True],
            [True],
            [True],
            [True]])



# Exercises

You should be able to complete them by referencing their specific section.

Note: For all exercises, your code should be device agnostic (meaning it could run on CPU or GPU if it's available).

1. Create a straight line dataset using the linear regression formula (`weight * X + bias`).
- Set weight=0.3 and bias=0.9 there should be at least 100 datapoints total.
- Split the data into 80% training, 20% testing.
- Plot the training and testing data so it becomes visual.

2. Build a PyTorch model by subclassing `nn.Module`.
- Inside should be a randomly initialized `nn.Parameter()` with `requires_grad=True`, one for weights and one for bias.
- Implement the `forward()` method to compute the linear regression function you used to create the dataset in 1.
- Once you've constructed the model, make an instance of it and check its `state_dict()`.
- **Note:** If you'd like to use `nn.Linear()` instead of `nn.Parameter()`you can.
3. Create a loss function and optimizer using `nn.L1Loss()` and torch.`optim.SGD(params, lr)` respectively.
- Set the learning rate of the optimizer to be 0.01 and the parameters to optimize should be the model parameters from the model you created in 2.
- Write a training loop to perform the appropriate training steps for 300 epochs.
- The training loop should test the model on the test dataset every 20 epochs.
4. Make predictions with the trained model on the test data.
- Visualize these predictions against the original training and testing data (note: you may need to make sure the predictions are not on the GPU if you want to use non-CUDA-enabled libraries such as matplotlib to plot).
5. Save your trained model's `state_dict()` to file.
- Create a new instance of your model class you made in 2. and load in the `state_dict()` you just saved to it.
- Perform predictions on your test data with the loaded model and confirm they match the original model predictions from 4.
>Resource: See the [exercises notebooks templates](https://github.com/mrdbourke/pytorch-deep-learning/tree/main/extras/exercises) and [solutions](https://github.com/mrdbourke/pytorch-deep-learning/tree/main/extras/solutions) on the course GitHub.



# Extra-curriculum

- Listen to The [Unofficial PyTorch Optimization Loop Song](https://youtu.be/Nutpusq_AFw) (to help remember the steps in a PyTorch training/testing loop).
- Read [What is `torch.nn`, really?](https://pytorch.org/tutorials/beginner/nn_tutorial.html) by Jeremy Howard for a deeper understanding of how one of the most important modules in PyTorch works.
- Spend 10-minutes scrolling through and checking out the [PyTorch documentation cheatsheet](https://pytorch.org/tutorials/beginner/ptcheat.html) for all of the different PyTorch modules you might come across.
- Spend 10-minutes reading the [loading and saving documentation on the PyTorch website](https://pytorch.org/tutorials/beginner/saving_loading_models.html) to become more familiar with the different saving and loading options in PyTorch.
- Spend 1-2 hours read/watching the following for an overview of the internals of gradient descent and backpropagation, the two main algorithms that have been working in the background to help our model learn.
- [Wikipedia page for gradient descent](https://en.wikipedia.org/wiki/Gradient_descent)
- [Gradient Descent Algorithm — a deep dive](https://towardsdatascience.com/gradient-descent-algorithm-a-deep-dive-cf04e8115f21) by Robert Kwiatkowski
- [Gradient descent, how neural networks learn video](https://youtu.be/IHZwWFHWa-w) by 3Blue1Brown
- [What is backpropagation really doing?](https://youtu.be/Ilg3gGewQ5U) video by 3Blue1Brown
- [Backpropagation Wikipedia Page](https://en.wikipedia.org/wiki/Backpropagation)
