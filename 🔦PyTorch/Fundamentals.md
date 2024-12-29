PyTorch is an open source machine learning and deep learning framework.

PyTorch allows you to manipulate and process data and write machine learning algorithms using Python code.

Topics to cover:

- **Introduction to tensors**: Tensors are the basic building block of all of machine learning and deep learning.

- **Creating tensors**:	Tensors can represent almost any kind of data (images, words, tables of numbers).

- **Getting information from tensors**: If you can put information into a tensor, you'll want to get it out too.

- **Manipulating tensors**: Machine learning algorithms (like neural networks) involve manipulating tensors in many different ways such as adding, multiplying, combining.

- **Dealing with tensor shapes**: One of the most common issues in machine learning is dealing with shape mismatches (trying to mixed wrong shaped tensors with other tensors).

- **Indexing on tensors**: If you've indexed on a Python list or NumPy array, it's very similar with tensors, except they can have far more dimensions.

- **Mixing PyTorch tensors with NumPy**: PyTorch plays with tensors (`torch.Tensor`), NumPy likes arrays (`np.ndarray`) sometimes you'll want to mix and match these.

- **Reproducibility**: Machine learning is very experimental and since it uses a lot of randomness to work, sometimes you'll want that randomness to not be so random.

- **Running tensors on GPU**:GPUs (Graphics Processing Units) make your code faster, PyTorch makes it easy to run your code on GPUs.

## Importing PyTorch


```python
import torch
torch.__version__
```




    '2.2.1+cu121'



## Introduction to tensors


Tensors are the fundamental building block of machine learning.

Their job is to represent data in a numerical way.

 you could represent an image as a tensor with shape `[3, 224, 224]` which would mean `[colour_channels, height, width]`, as in the image has `3` colour channels (red, green, blue), a height of `224` pixels and a width of `224` pixels.

![Tensor_Shape](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/00-tensor-shape-example-of-image.png)

## Creating Tensors

A scalar is a single number and in tensor-speak it's a zero dimension tensor.




```python
# Scalar
scalar = torch.tensor(7)
scalar
```




    tensor(7)




```python
scalar.ndim
```




    0




```python
# to retrieve the number from tensor
scalar.item()
```




    7




```python
# Vector
vector = torch.tensor([7,7])
vector
```




    tensor([7, 7])




```python
vector.ndim
```




    1




```python
vector.shape
```




    torch.Size([2])




```python
# Matrix
matrix = torch.tensor([[7,8],
                      [8,7]])
matrix
```




    tensor([[7, 8],
            [8, 7]])




```python
matrix.ndim
```




    2




```python
matrix.shape
```




    torch.Size([2, 2])




```python
# Tensor
tensor = torch.tensor([[[1,2,3],
                        [4,5,6],
                        [7,8,9]]])
tensor
```




    tensor([[[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]])




```python
tensor.ndim
```




    3




```python
tensor.shape
```




    torch.Size([1, 3, 3])



![Tensor_Dimension](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/00-pytorch-different-tensor-dimensions.png)

![](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/00-scalar-vector-matrix-tensor.png)

## Random tensors

In Machine Learning,

`Start with random numbers -> look at data -> update random numbers -> look at data -> update random numbers...`

As a data scientist, you can define how the machine learning model starts (initialization), looks at data (representation) and updates (optimization) its random numbers.

`torch.rand()`



```python
# Create a random tensor of size (3,4)
random_tensor = torch.rand(3,4)
random_tensor
```




    tensor([[0.3205, 0.6955, 0.1629, 0.4989],
            [0.1220, 0.0860, 0.7333, 0.1381],
            [0.6968, 0.3819, 0.2719, 0.8880]])




```python
# with torch.rand() we can adjust the size flexibily
num = torch.rand(size=(224, 224, 3))
num.shape, num.ndim
```




    (torch.Size([224, 224, 3]), 3)



## Zeros and Ones


```python
# tensor of all zeros
zeros = torch.zeros(size=(3,4))
zeros
```




    tensor([[0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]])




```python
# tensor of all ones
ones = torch.ones(size=(3,4))
ones
```




    tensor([[1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]])



## Creating a range and tensors like

`torch.arange(start, end, step)`

- `start` = start of range
- `end` = end of range
- `step` = how many steps in between each value


```python
# Use torch.arange(), torch.range() is deprecated
zero_to_ten_deprecated = torch.range(0, 10) # Note: this may return an error in the future

# Create a range of values 0 to 10
zero_to_ten = torch.arange(start=0, end=10, step=1)
zero_to_ten
```

    <ipython-input-19-a404776195c1>:2: UserWarning: torch.range is deprecated and will be removed in a future release because its behavior is inconsistent with Python's range builtin. Instead, use torch.arange, which produces values in [start, end).
      zero_to_ten_deprecated = torch.range(0, 10) # Note: this may return an error in the future





    tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
# Can also create a tensor of zeros similar to another tensor
ten_zeros = torch.zeros_like(input=zero_to_ten) # will have same shape
ten_zeros
```




    tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])



## Matrix Multiplication

[matrix multiplication](https://www.mathsisfun.com/algebra/matrix-multiplying.html)

PyTorch implements matrix multiplication functionality in the [`torch.matmul()`](https://pytorch.org/docs/stable/generated/torch.matmul.html) method.

The main two rules for matrix multiplication to remember are:

1. The inner dimensions must match:
  - (3, 2) @ (3, 2) won't work
  - (2, 3) @ (3, 2) will work
  - (3, 2) @ (2, 3) will work

2. The resulting matrix has the shape of the outer dimensions:
  - (2, 3) @ (3, 2) -> (2, 2)
  - (3, 2) @ (2, 3) -> (3, 3)

Note: "@" in Python is the symbol for matrix multiplication.

Resource: You can see all of the rules for matrix multiplication using torch.matmul() in the [PyTorch documentation](https://pytorch.org/docs/stable/generated/torch.matmul.html).


```python
import torch
tensor = torch.tensor([1,2,3])
tensor.shape
```




    torch.Size([3])



| Operation                   | Calculation                   | Code                  |
| --------------------------- | ----------------------------- | --------------------- |
| Element-wise multiplication | `[1*1, 2*2, 3*3] = [1, 4, 9]` | tensor * tensor       |
| Matrix multiplication       | `[1*1 + 2*2 + 3*3] = [14]`    | tensor.matmul(tensor) |
|                             |                               |                       |



```python
# Element-wise
tensor * tensor
```




    tensor([1, 4, 9])




```python
# Matrix multiplication
torch.matmul(tensor, tensor)
```




    tensor(14)




```python
# Shapes need to be in the right way
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
                         [8, 11],
                         [9, 12]], dtype=torch.float32)

torch.matmul(tensor_A, tensor_B) # (this will error)
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-37-5893e600ebdf> in <cell line: 10>()
          8                          [9, 12]], dtype=torch.float32)
          9 
    ---> 10 torch.matmul(tensor_A, tensor_B) # (this will error)
    

    RuntimeError: mat1 and mat2 shapes cannot be multiplied (3x2 and 3x2)



```python
tensor_A.shape, tensor_B.shape
```




    (torch.Size([3, 2]), torch.Size([3, 2]))




```python
tensor_A.shape, tensor_B.T.shape
```




    (torch.Size([3, 2]), torch.Size([2, 3]))




```python
mat = torch.matmul(tensor_A, tensor_B.T)
mat
```




    tensor([[ 27.,  30.,  33.],
            [ 61.,  68.,  75.],
            [ 95., 106., 117.]])



You can also use `torch.mm()` which is a short for `torch.matmul()`.




```python
# torch.mm is a shortcut for matmul
torch.mm(tensor_A, tensor_B.T)
```




    tensor([[ 27.,  30.,  33.],
            [ 61.,  68.,  75.],
            [ 95., 106., 117.]])




Neural networks are full of matrix multiplications and dot products.

The `torch.nn.Linear()` module (we'll see this in action later on), also known as a feed-forward layer or fully connected layer, implements a matrix multiplication between an input x and a weights matrix A.

$$ y=x⋅A^T+b $$

Where:
- `x` is the input to the layer (deep learning is a stack of layers like `torch.nn.Linear()` and others on top of each other).
- `A` is the weights matrix created by the layer, this starts out as random numbers that get adjusted as a neural network learns to better represent patterns in the data (notice the "`T`", that's because the weights matrix gets transposed).
  - Note: You might also often see W or another letter like X used to showcase the weights matrix.
- `b` is the bias term used to slightly offset the weights and inputs.
- `y` is the output (a manipulation of the input in the hopes to discover patterns in it).



```python
# Since the linear layer starts with a random weights matrix
torch.manual_seed(42)
# this uses mm
linear = torch.nn.Linear(in_features = 2,   # in_features = matches inner dimension of input
                        out_features=6) # out_features = describes outer value

x = tensor_A
output = linear(x)
print(f"Input shape: {x.shape}\n")
print(f"Output:\n{output}\n\nOutput shape: {output.shape}")
```

    Input shape: torch.Size([3, 2])
    
    Output:
    tensor([[2.2368, 1.2292, 0.4714, 0.3864, 0.1309, 0.9838],
            [4.4919, 2.1970, 0.4469, 0.5285, 0.3401, 2.4777],
            [6.7469, 3.1648, 0.4224, 0.6705, 0.5493, 3.9716]],
           grad_fn=<AddmmBackward0>)
    
    Output shape: torch.Size([3, 6])


![](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/00_matrix_multiplication_is_all_you_need.jpeg)

## Positional min/max

`torch.argmax()`
`torch.argmin()`


```python
# Create a tensor
tensor = torch.arange(10, 100, 10)
print(f"Tensor: {tensor}")

# return index of max and min  values
print(f"Index where max value occurs: {tensor.argmax()}")
print(f"Index where min value occurs: {tensor.argmin()}")
```

    Tensor: tensor([10, 20, 30, 40, 50, 60, 70, 80, 90])
    Index where max value occurs: 8
    Index where min value occurs: 0


## Change tensor datatype

`torch.float64`
`torch.float32`

`torch.Tensor.type(dtype=None)`


```python
# Create a tensor and check its datatype
tensor = torch.arange(10., 100., 10.)
tensor.dtype
```




    torch.float32




```python
# Create a float16 tensor
tensor_float16 = tensor.type(torch.float16)
tensor_float16
```




    tensor([10., 20., 30., 40., 50., 60., 70., 80., 90.], dtype=torch.float16)




```python
# Create a int8 tensor
tensor_int8 = tensor.type(torch.int8)
tensor_int8
```




    tensor([10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=torch.int8)



**Note:** lower the number (8,16,32) less precise the tensor.

## Reshaping, stacking, squeezing and unsqueezing



| Method                     | One-line description                                                                     |
|----------------------------|------------------------------------------------------------------------------------------|
| `torch.reshape(input, shape)`| Reshapes input to shape (if compatible), can also use torch.Tensor.reshape().            |
| `Tensor.view(shape)`         | Returns a view of the original tensor in a different shape but shares the same data as the original tensor. |
| `torch.stack(tensors, dim=0)`| Concatenates a sequence of tensors along a new dimension (dim), all tensors must be same size. |
| `torch.squeeze(input)`       | Squeezes input to remove all the dimensions with value 1.                                  |
| `torch.unsqueeze(input, dim)`| Returns input with a dimension value of 1 added at dim.                                    |
| `torch.permute(input, dims)`| Returns a view of the original input with its dimensions permuted (rearranged) to dims.     |



Why do any of these?

Because deep learning models (neural networks) are all about manipulating tensors in some way. And because of the rules of matrix multiplication, if you've got shape mismatches, you'll run into errors. These methods help you make sure the right elements of your tensors are mixing with the right elements of other tensors.


```python
# Create a tensor
import torch
x = torch.arange(1.,8.)
x, x.shape
```




    (tensor([1., 2., 3., 4., 5., 6., 7.]), torch.Size([7]))



Add extra dimension with `torch.reshape()`


```python
x_reshape = x.reshape(1,7)
x_reshape, x_reshape.shape
```




    (tensor([[1., 2., 3., 4., 5., 6., 7.]]), torch.Size([1, 7]))



Remember though, changing the view of a tensor with `torch.view()` really only creates a new view of the same tensor.




```python
# Change view (keeps same data as original but changes view)
# See more: https://stackoverflow.com/a/54507446/7900723
z = x.view(1, 7)
z, z.shape
```




    (tensor([[1., 2., 3., 4., 5., 6., 7.]]), torch.Size([1, 7]))




```python
# Changing z changes x
z[:, 0] = 5
z, x
```




    (tensor([[5., 2., 3., 4., 5., 6., 7.]]), tensor([5., 2., 3., 4., 5., 6., 7.]))



If we wanted to stack our new tensor on top of itself five times, we could do so with `torch.stack()`.




```python
x_stacked = torch.stack([x, x, x, x], dim=0)
x_stacked
```




    tensor([[5., 2., 3., 4., 5., 6., 7.],
            [5., 2., 3., 4., 5., 6., 7.],
            [5., 2., 3., 4., 5., 6., 7.],
            [5., 2., 3., 4., 5., 6., 7.]])




```python
x_stacked = torch.stack([x, x, x, x], dim=1)
x_stacked
```




    tensor([[5., 5., 5., 5.],
            [2., 2., 2., 2.],
            [3., 3., 3., 3.],
            [4., 4., 4., 4.],
            [5., 5., 5., 5.],
            [6., 6., 6., 6.],
            [7., 7., 7., 7.]])



How about removing all single dimensions from a tensor?

To do so you can use `torch.squeeze()` (I remember this as squeezing the tensor to only have dimensions over 1).




```python
print(f"Previous tensor: {x_reshape}")
print(f"Previous shape: {x_reshape.shape}")

# Remove extra dimension from x_reshaped
x_squeezed = x_reshape.squeeze()
print(f"\nNew tensor: {x_squeezed}")
print(f"New shape: {x_squeezed.shape}")
```

    Previous tensor: tensor([[5., 2., 3., 4., 5., 6., 7.]])
    Previous shape: torch.Size([1, 7])
    
    New tensor: tensor([5., 2., 3., 4., 5., 6., 7.])
    New shape: torch.Size([7])


And to do the reverse of `torch.squeeze()` you can use `torch.unsqueeze()` to add a dimension value of 1 at a specific index.




```python
print(f"Previous tensor: {x_squeezed}")
print(f"Previous shape: {x_squeezed.shape}")

## Add an extra dimension with unsqueeze
x_unsqueezed = x_squeezed.unsqueeze(dim=0)
print(f"\nNew tensor: {x_unsqueezed}")
print(f"New shape: {x_unsqueezed.shape}")
```

    Previous tensor: tensor([5., 2., 3., 4., 5., 6., 7.])
    Previous shape: torch.Size([7])
    
    New tensor: tensor([[5., 2., 3., 4., 5., 6., 7.]])
    New shape: torch.Size([1, 7])


You can also rearrange the order of axes values with `torch.permute(input, dims)`, where the `input` gets turned into a view with new `dims`.




```python
# Create tensor with specific shape
x_original = torch.rand(size=(224, 200, 3))

# Permute the original tensor to rearrange the axis order
x_permuted = x_original.permute(2, 0, 1) # shifts axis 0->1, 1->2, 2->0

print(f"Previous shape: {x_original.shape}")
print(f"New shape: {x_permuted.shape}")
```

    Previous shape: torch.Size([224, 200, 3])
    New shape: torch.Size([3, 224, 200])


## Indexing (selecting data from tensors)


```python
# Create a tensor
import torch
x = torch.arange(1, 10).reshape(1, 3, 3)
x, x.shape
```




    (tensor([[[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]]]),
     torch.Size([1, 3, 3]))




```python
# Let's index bracket by bracket
print(f"First square bracket:\n{x[0]}")
print(f"Second square bracket: {x[0][0]}")
print(f"Third square bracket: {x[0][0][0]}")
```

    First square bracket:
    tensor([[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]])
    Second square bracket: tensor([1, 2, 3])
    Third square bracket: 1


## Pytorch tensors and NumPy

The two main methods you'll want to use for NumPy to PyTorch (and back again) are:

- `torch.from_numpy(ndarray)` - NumPy array -> PyTorch tensor.
- `torch.Tensor.numpy()` - PyTorch tensor -> NumPy array.



```python
# NumPy array to tensor
import torch
import numpy as np

array = np.arange(1.0,8.0)
tensor = torch.from_numpy(array)
array, tensor
```




    (array([1., 2., 3., 4., 5., 6., 7.]),
     tensor([1., 2., 3., 4., 5., 6., 7.], dtype=torch.float64))




```python
# Tensor to NumPy array
tensor = torch.ones(7) # create a tensor of ones with dtype=float32
numpy_tensor = tensor.numpy() # will be dtype=float32 unless changed
tensor, numpy_tensor
```




    (tensor([1., 1., 1., 1., 1., 1., 1.]),
     array([1., 1., 1., 1., 1., 1., 1.], dtype=float32))



## Reproducibility


```python
import torch

# Create two random tensors
random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)

print(f"Tensor A:\n{random_tensor_A}\n")
print(f"Tensor B:\n{random_tensor_B}\n")
print(f"Does Tensor A equal Tensor B? (anywhere)")
random_tensor_A == random_tensor_B
```

    Tensor A:
    tensor([[0.4681, 0.5993, 0.2674, 0.4412],
            [0.1681, 0.3956, 0.1113, 0.0671],
            [0.3361, 0.1435, 0.3796, 0.3324]])
    
    Tensor B:
    tensor([[0.0817, 0.7385, 0.2578, 0.6926],
            [0.5032, 0.1346, 0.8236, 0.2403],
            [0.5515, 0.2301, 0.4249, 0.2892]])
    
    Does Tensor A equal Tensor B? (anywhere)


    tensor([[False, False, False, False],
            [False, False, False, False],
            [False, False, False, False]])



`torch.manual_seed()`


```python
import torch
import random

# Set the random seed
RANDOM_SEED=42
torch.manual_seed(seed=RANDOM_SEED)
random_tensor_C = torch.rand(3, 4)

# Have to reset the seed every time a new rand() is called
# Without this, tensor_D would be different to tensor_C
torch.random.manual_seed(seed=RANDOM_SEED) # try commenting this line out and seeing what happens
random_tensor_D = torch.rand(3, 4)

print(f"Tensor C:\n{random_tensor_C}\n")
print(f"Tensor D:\n{random_tensor_D}\n")
print(f"Does Tensor C equal Tensor D? (anywhere)")
random_tensor_C == random_tensor_D
```

    Tensor C:
    tensor([[0.8823, 0.9150, 0.3829, 0.9593],
            [0.3904, 0.6009, 0.2566, 0.7936],
            [0.9408, 0.1332, 0.9346, 0.5936]])
    
    Tensor D:
    tensor([[0.8823, 0.9150, 0.3829, 0.9593],
            [0.3904, 0.6009, 0.2566, 0.7936],
            [0.9408, 0.1332, 0.9346, 0.5936]])
    
    Does Tensor C equal Tensor D? (anywhere)

    tensor([[True, True, True, True],
            [True, True, True, True],
            [True, True, True, True]])



## Exercises

All of the exercises are focused on practicing the code above.

You should be able to complete them by referencing each section or by following the resource(s) linked.

Resources:

- [Exercise template notebook for 00](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/exercises/00_pytorch_fundamentals_exercises.ipynb)
- [Example solutions notebook for 00](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/solutions/00_pytorch_fundamentals_exercise_solutions.ipynb) (try the exercises before looking at this).

1. Documentation reading - A big part of deep learning (and learning to code in general) is getting familiar with the documentation of a certain framework you're using. We'll be using the PyTorch documentation a lot throughout the rest of this course. So I'd recommend spending 10-minutes reading the following (it's okay if you don't get some things for now, the focus is not yet full understanding, it's awareness). See the documentation on `torch.Tensor` and for `torch.cuda`.
2. Create a random tensor with shape `(7, 7)`.
3. Perform a matrix multiplication on the tensor from 2 with another random tensor with shape `(1, 7)` (hint: you may have to transpose the second tensor).
4. Set the random seed to `0` and do exercises 2 & 3 over again.
5. Speaking of random seeds, we saw how to set it with `torch.manual_seed()` but is there a GPU equivalent? (hint: you'll need to look into the documentation for `torch.cuda` for this one). If there is, set the GPU random seed to `1234`.
6. Create two random tensors of shape `(2, 3)` and send them both to the GPU (you'll need access to a GPU for this). Set `torch.manual_seed(1234)` when creating the tensors (this doesn't have to be the GPU random seed).
7. Perform a matrix multiplication on the tensors you created in 6 (again, you may have to adjust the shapes of one of the tensors).
8. Find the maximum and minimum values of the output of 7.
9. Find the maximum and minimum index values of the output of 7.
10. Make a random tensor with shape `(1, 1, 1, 10)` and then create a new tensor with all the `1` dimensions removed to be left with a tensor of shape `(10)`. Set the seed to `7` when you create it and print out the first tensor and it's shape as well as the second tensor and it's shape.


## Extra-curriculum

- Spend 1-hour going through the [PyTorch basics tutorial](https://pytorch.org/tutorials/beginner/basics/intro.html) (I'd recommend the [Quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) and [Tensors](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) sections).
- To learn more on how a tensor can represent data, see this video: [What's a tensor?](https://youtu.be/f5liqUk0ZTw)
