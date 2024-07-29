# WYEF AI NEURAL NETWORK FROM SCRATCH

## Project Overview

This project is focused on training a neural network to classify images from the MNIST dataset. The MNIST dataset consists of handwritten digit images and is commonly used for training various image processing systems.

本项目专注于训练一个神经网络来分类MNIST数据集中的图像。MNIST数据集由手写数字图像组成，常用于训练各种图像处理系统。

## Project Steps

### 1. Imports
The project starts by importing necessary libraries, including NumPy for numerical operations, Matplotlib for plotting, and the random library for shuffling data.

项目开始时导入了必要的库，包括用于数值运算的NumPy，用于绘图的Matplotlib以及用于数据混洗的随机库。

### 2. Function Definitions
Key functions are defined:
- `show_image`: Displays an image from the dataset.
- `sigmoid`: Implements the sigmoid activation function.
- `sigmoid_deriv`: Computes the derivative of the sigmoid function.
- `cost`: Calculates the mean squared error between the actual and predicted outputs.

定义了关键函数：
- `show_image`: 显示数据集中的图像。
- `sigmoid`: 实现sigmoid激活函数。
- `sigmoid_deriv`: 计算sigmoid函数的导数。
- `cost`: 计算实际输出和预测输出之间的均方误差。

### 2.5 Dataset 

The dataset can be found and downloaded here [https://github.com/fgnt/mnist](https://github.com/fgnt/mnist)

### 3. Reading the Train Set
The training images and labels are read from the MNIST dataset files. The images are normalized, and the labels are converted into one-hot vectors.

从MNIST数据集文件中读取训练图像和标签。图像被标准化，标签被转换为one-hot向量。

### 4. Reading the Test Set
Similarly, the test images and labels are read and processed as described above for the training set.

同样，测试图像和标签也按照上面描述的方式进行读取和处理。

### 5. Weights and Biases Initialization
Weights and biases for three layers are initialized. The first layer has 784 input neurons and 16 output neurons, the second layer has 16 input and output neurons, and the third layer has 16 input neurons and 10 output neurons.

对三层的权重和偏置进行初始化。第一层有784个输入神经元和16个输出神经元，第二层有16个输入和输出神经元，第三层有16个输入神经元和10个输出神经元。

### 6. Hyperparameters
The hyperparameters such as the number of samples, batch size, number of epochs, and learning rate (alpha) are set.

设置超参数，例如样本数量、批处理大小、训练轮数和学习率（alpha）。

### 7. Training the Neural Network
The neural network is trained using the backpropagation algorithm. The process includes:
- Shuffling the training set.
- Performing forward and backward propagation to update weights and biases.
- Recording the average cost after each epoch.

使用反向传播算法训练神经网络。过程包括：
- 混洗训练集。
- 执行前向传播和后向传播以更新权重和偏置。
- 在每个训练轮结束后记录平均成本。

### 8. Average Cost Over Time
The average cost over time is plotted to visualize the learning process of the neural network.

绘制随时间变化的平均成本，以可视化神经网络的学习过程。

### 9. Evaluating the Network on the Train Set
The success rate of the neural network on the training set is evaluated by comparing the predicted labels with the actual labels.

通过比较预测标签与实际标签来评估神经网络在训练集上的成功率。

### 10. Evaluating the Network on the Test Set
Similarly, the success rate on the test set is calculated to check the network's performance on unseen data.

同样，计算测试集上的成功率，以检查网络在未见数据上的表现。

## Results
The trained neural network achieved a high success rate on both the training and test sets, demonstrating its effectiveness in classifying handwritten digits.

训练好的神经网络在训练集和测试集上都达到了很高的成功率，证明了它在手写数字分类上的有效性。

Handwritten digit recognition (MNIST) using a Neural Network implemented from scratch!

Resources:
- [3Blue1Brown - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Stochastic Gradient Descent, Clearly Explained!](https://www.youtube.com/watch?v=vMh0zPT0tLI)
- [The MNIST Database of Handwritten Digits](http://yann.lecun.com/exdb/mnist)
