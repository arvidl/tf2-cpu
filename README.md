# tf2-cpu
TF2-CPU   (should also work on the Mac)


## Make a conda environment

> conda create --name tf2-cpu python=3.7

## Activate conda environment tf2-cpu
> conda activate tf2-cpu


## Install Preview TF 2.0 Beta build for CPU-only (unstable)
> pip install tensorflow==2.0.0-beta1


## Install Jupyter kernel
> pip install ipykernel <br>
> python -m ipykernel install --user --name tf2-cpu --display-name "TF2-CPU"

## Install Jupyter notebook
> pip install jupyter notebook

## Get started
```python
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
```
