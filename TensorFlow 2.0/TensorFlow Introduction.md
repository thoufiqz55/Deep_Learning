# TENSORFLOW 2.0

## What is TensorFlow?

### TensorFlow is an open-source library used to make end-to-end Machine Learning apps.<br>

### TensorFlow library is used in Numerical computation and Large-scale machine learning.<br>

### TensorFlow was developed by Google Brain Team in 2015.<br>

### TensorFlow 2.0 is launched in the year 2019.<br>

### TensorFlow can also be called as a mathematical library used in training of deep learning neural networks.<br>

### For eg .like Google AI uses Tensorflow to optimize search engine user experience so that it automatically shows next word to come as we try to type something inside the search tab.<br>

### Tensorflow can be used in wide variety of programming languages like Python, JavaScript, C++ & Java.<br>

### It was designed to work on multiple CPUs or GPUs, as well as mobile operating systems in some circumstances, and it includes wrappers in Python, C++, and Java.<br>

### The most important thing to realize about TensorFlow is that, for the most part, the core is not written in Python: It's written in a combination of highly-optimized C++ and CUDA (Nvidia's language for programming GPUs).<br>  
 
## Why to use Tensorflow?

### It is open source library that is free to use and develope software without any constraint.<br>

### It has a comprehensive, flexible ecosystem of tools, libraries and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML powered applications.<br>

### Easy model building because of availibility of high-end API like Keras for debugging and iteration of model.<br>

### Robust model production lets you deploy model on any platform with no matter what language you use.<br>
 
 
 
## How TensorFlow Works?
 
 ![image](https://user-images.githubusercontent.com/100178747/172631362-beccfaec-060f-4a03-b3eb-fdf4f9506a54.png)

 

### TensorFlow allows developers to create graph or structure to show how data flows through graphs or processing nodes.<br>

### So each node in this graph represents mathematical operation and the connection or edges between these shows tensor that is called Tensors(Multi Dimensional Array).<br>
 
### TensorFlow applications can be run on most any target that’s convenient: a local machine, a cluster in the cloud, iOS and Android devices, CPUs or GPUs.<br>

### A trained model can be used to deliver predictions as a service via a Docker container using REST or gRPC APIs.<br>

## TensorFlow using Python

### Nodes and tensors in TensorFlow are Python objects, and TensorFlow applications are themselves Python applications.<br>

### The actual math operations, however, are not performed in Python. The libraries of transformations that are available through TensorFlow are written as high-performance C++ binaries.<br>

### Python just directs traffic between the pieces and provides high-level programming abstractions to hook them together.<br>

### High-level work in TensorFlow—creating nodes and layers and linking them together—uses the Keras library.<br>

### The Keras API is outwardly simple; a basic model with three layers can be defined in less than 10 lines of code, and the training code for the same takes just a few more lines of code. But if you want to "lift the hood" and do more fine-grained work, such as writing your own training loop, you can do that.<br>
 
 
## Applications of TensorFlow  

### 1) Face Recognition

### 2) Self Driving Cars

### 3) Alexa

### 4) Human Robotics

### & Many More

## Pre-requisites / Dependencies for TensorFlow 
 
### 1) Python

### 2) NumPy, SciPy, Pandas, Matplotlib & few more

### 3) Matrix - Vector, Statistics, Algebra, Calculus, Differentials

## Installation of TensorFlow -

### 1) Create virtual enviorment

### python3 -m venv env ---To create virtual env.

### source env/bin/activate ---To activate virtual env.

### 2) pip install tensorflow (CPU installation only)

-------------------------OR----------------------------------

### 3) pip install tensorflow-gpu (N-vidia GPU with CUDA)

### For Anaconda Nav users -

### CPU Only TensorFlow :

### 1) conda create -n env_name tensorflow

### 2) conda activate env_name

### GPU only TensorFlow :

### 1) conda create -n env_name-gpu tensorflow-gpu

### 2) conda activate env_name-gpu

### For CUDA version -

### GPU Tensorflow uses CUDA

### 1) conda create -n env_name-gpu-cuda8 tensorflow-gpu cudatoolkit

### 2) conda activate env_name-gpu-cuda8
 
## Importing TensorFlow

### Note - first activate the created env otherwise it will throw error.

### import tensorflow as tf

### tf is usually used notation for tensorflow as standard while importing the library.

## A Quick Example

### Lets see, quick example of TensorFlow using code snippet.<br>

### 1) Import & get installed version<br>

### Importing tensorflow as tf<br>

### import tensorflow as tf<br>

### To get curent version of tensorflow installed

### print("TensorFlow version:", tf.__version__)<br>

### 2) Load a dataset<br>

### Lets use MNIST dataset which is default and provided with tensorflow.<br>

### mnist = tf.keras.datasets.mnist<br>

### 3) Convert the sample data from integers to floating-point numbers:<br>

### (x_train, y_train), (x_test, y_test) = mnist.load_data() x_train, x_test = x_train / 255.0, x_test / 255.0<br>

### 4) Build a machine learning model

### Build a tf.keras.Sequential model by stacking layers.<br>

### model = tf.keras.models.Sequential([   tf.keras.layers.Flatten(input_shape=(28, 28)),   tf.keras.layers.Dense(128, activation='relu'),   tf.keras.layers.Dropout(0.2),   tf.keras.layers.Dense(10) ])<br>

### 5) Lets Predict<br>

### For each example, the model returns a vector of logits or log-odds scores, one for each class.<br>

### predictions = model(x_train[:1]).numpy() predictions<br>

### array([[ 0.2760778 , -0.39324787, -0.17098302,  1.2016621 , -0.03416392, 0.5461229 , -0.7203061 , -0.41886678, -0.59480035, -0.7580608 ]], dtype=float32)<br>

### 6) Lets Convert these Logits into class Probabilities<br>

### tf.nn.softmax(predictions).numpy()<br>

### array([[0.11960829, 0.06124588, 0.0764901 , 0.30181262, 0.08770514, 0.15668967, 0.04416083, 0.05969675, 0.05006609, 0.04252464]], dtype=float32)<br>
### 7) Lets Define Loss Function<br>

### loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)<br>

### This loss is equal to the negative log probability of the true class: The loss is zero if the model is sure of the correct class.<br>

### This untrained model gives probabilities close to random (1/10 for each class), so the initial loss should be close to -tf.math.log(1/10) ~= 2.3.

### loss_fn(y_train[:1], predictions).numpy()<br>

### 1.8534881<br>

### Before you start training, configure and compile the model using Keras Model.compile. Set the optimizer class to adam, set the loss to the<br> loss_fn function you defined earlier, and specify a metric to be evaluated for the model by setting the metrics parameter to accuracy.<br>

### model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])<br>

### 8) Lets train & evaluate model<br>

### Use the Model.fit method to adjust your model parameters and minimize the loss:<br>

### model.fit(x_train, y_train, epochs=5)<br>

### Epoch 1/5 1875/1875 [==============================] - 4s 2ms/step - loss: 0.2950 - accuracy: 0.9143<br>
### Epoch 2/5 1875/1875 [==============================] - 3s 2ms/step - loss: 0.1451 - accuracy: 0.9567<br>
### Epoch 3/5 1875/1875 [==============================] - 4s 2ms/step - loss: 0.1080 - accuracy: 0.9668<br>
### Epoch 4/5 1875/1875 [==============================] - 4s 2ms/step - loss: 0.0906 - accuracy: 0.9717<br>
### Epoch 5/5 1875/1875 [==============================] - 4s 2ms/step - loss: 0.0749 - accuracy: 0.9761<br>
### <keras.callbacks.History at 0x7f062c606850><br>
 
### The Model.evaluate method checks the models performance, usually on a "Validation-set" or "Test-set".<br>

### model.evaluate(x_test,  y_test, verbose=2)<br>

### 313/313 - 1s - loss: 0.0783 - accuracy: 0.9755 - 588ms/epoch - 2ms/step<br>
### [0.07825208455324173, 0.9754999876022339]<br>
### The image classifier is now trained to ~98% accuracy on this dataset.<br>

### If you want your model to return a probability, you can wrap the trained model, and attach the softmax to it:<br>

### probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])<br>

### probability_model(x_test[:5])<br>

### <tf.Tensor: shape=(5, 10), dtype=float32, numpy= array([[2.72807270e-08, 2.42517650e-08, 7.75602894e-06, 1.28684027e-04, 7.66215633e-11, 3.54162950e-07, 3.04894151e-14, 9.99857187e-01, 2.32766553e-08, 5.97762892e-06], [7.37396704e-08, 4.73638036e-04, 9.99523997e-01, 7.20633352e-07, 4.54133671e-17, 1.42298268e-06, 5.96959016e-09, 1.23534145e-13, 7.77225608e-08, 6.98619169e-16], [1.95462448e-07, 9.99295831e-01, 1.02249986e-04, 1.86699708e-05, 5.65737491e-06, 1.12115902e-06, 5.32719559e-06, 5.22767776e-04, 4.79981136e-05, 1.76624681e-07], [9.99649286e-01, 1.80224735e-09, 3.73612856e-05, 1.52324446e-07, 1.30824594e-06, 2.82781020e-05, 6.99703523e-05, 3.30940424e-07, 2.13184350e-07, 2.13106396e-04], [1.53770895e-06, 1.72272063e-08, 1.98980865e-06, 3.97882580e-08, 9.97192323e-01, 1.10544443e-05, 1.54713348e-06, 2.81727880e-05, 3.48721733e-06, 2.75991508e-03]], dtype=float32)><br>
 
# Congratulations!
## You have done your job of training model using Keras API.
 
 

 
