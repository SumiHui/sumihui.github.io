---
title:	TENSORFLOW INPUT PIPELINE EXAMPLE
date: 2017-10-17 19:44:00
categories: [tensorflow,cnn]
tags: [tensorflow,python,cnn]
---
*From [TENSORFLOW INPUT PIPELINE EXAMPLE](http://ischlag.github.io/2016/06/19/tensorflow-input-pipeline-example/)*

A brief example of using tensorflows input pipeline to load your own custom data structures into tensorflows computational graph. This includes the partitioning of the data into a test and train set and batching together a set of images. You can find all the code at the bottom of this page. Note: This will not work for large datasets as ops.convert_to_tensor will create constants of your data in your graph! Have a look at this next post in order to use TensorFlow input pipelines with very large datasets.

#### Context

The official tensorflow documentation on reading in data can be found here but also have a look at the API. With this script you can create a jpg based dataset from the compact mnist dataset format. With this dataset, we are going to build a simple and efficient input pipeline. This example was done using tensorflow 0.8.0.

#### Load Data in Tensorflow

<!-- more -->

Ok, let’s take one step back. There are two ways on how you can load your data into your tensorflow graph (excluding a third way of loading the data as a constant). In an earlier post, we have used the feeding method where we provide a feed_dict object with our data and labels at every step. This can be problematic if our dataset is too big to be stored in our working memory and so the authors of tensorflow introduced input pipelines. The next steps are going to describe how that pipeline should look like. Only when we start our queue runners right before our session operations the pipeline will be active and loading data.

The input pipeline deals with reading csv files, decode file formats, restructure the data, shuffle the data, data augmentation or other preprocessing, and load the data in batches using threads. However, we do have to write some code to get this to work.

#### Load the Label Data

Make sure you are using the correct paths to the dataset created by the script file linked above.
```python
dataset_path      = "/path/to/out/dataset/mnist/"
test_labels_file  = "test-labels.csv"
train_labels_file = "train-labels.csv"
The first thing we do is loading the label and image path information from the text files we have generated. The encode_label function assigns an int value to our string text. For this example, we are going to simply convert the string label into an integer since they are all numbers. We are not going to train a model in this example so if you want to train e.g. a neural network you probably want to do some one-hot encoding here. But be aware that the format of our label needs to be congruent with the tensor dtype later on!

def encode_label(label):
  return int(label)

def read_label_file(file):
  f = open(file, "r")
  filepaths = []
  labels = []
  for line in f:
    filepath, label = line.split(",")
    filepaths.append(filepath)
    labels.append(encode_label(label))
  return filepaths, labels

# reading labels and file path
train_filepaths, train_labels = read_label_file(dataset_path + train_labels_file)
test_filepaths, test_labels = read_label_file(dataset_path + test_labels_file)
```
#### Do Some Optional Processing on Our String Lists

Next, we will transform the relative image path into a full image path and for the sake of this example we are also going to concat the given train and test set. We then shuffle the data and create our own train and test set later on. Be aware that you should not do this if you want to compare your trained models with other mnist models. We only do this to show the capabilities of the tensorflow input pipeline!

To make the output of the script clearer we will also shrink the full dataset to only 20 samples (did I tell you yet that this is an example?).
```python
# transform relative path into full path
train_filepaths = [ dataset_path + fp for fp in train_filepaths]
test_filepaths = [ dataset_path + fp for fp in test_filepaths]

# for this example we will create or own test partition
all_filepaths = train_filepaths + test_filepaths
all_labels = train_labels + test_labels

# we limit the number of files to 20 to make the output more clear!
all_filepaths = all_filepaths[:20]
all_labels = all_labels[:20]
```
#### Start Building the Pipeline

Now we will start to define our pipeline. Make sure that the dtype of our tensor is matching the type of data that we have in our lists. For the next part, we will need the following imports.
```python
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
With those we can create our tensorflow objects. If you decide to use the train and test data set of mnist as it is given you would just do the next steps for both sets simultaneously.

# convert string into tensors
all_images = ops.convert_to_tensor(all_filepaths, dtype=dtypes.string)
all_labels = ops.convert_to_tensor(all_labels, dtype=dtypes.int32)
```
#### Lets Partition the Data

This step is optional. Since we have all our 20 samples in one (big) set, we want to perform some partitioning to build a test and train set. Tensorflow comes with the ability to do this on-the-fly with tensors so we don’t have to do this beforehand. If the partition function is confusing you can read up the official explanation here. For the running example, we define the test_set_size to be 5 samples.

![local_response_normalization](https://sumihui.github.io/source/images/DynamicPartition.png)

A visualization of the dynamic partition function in tensorflow.
```python
# create a partition vector
partitions = [0] * len(all_filepaths)
partitions[:test_set_size] = [1] * test_set_size
random.shuffle(partitions)

# partition our data into a test and train set according to our partition vector
train_images, test_images = tf.dynamic_partition(all_images, partitions, 2)
train_labels, test_labels = tf.dynamic_partition(all_labels, partitions, 2)
```
#### Build the Input Queues and Define How to Load Images

The slice_input_producer will slice our tensors into single instances and queue them up using threads. There are further parameters to define the number of threads used and the capacity of the queue (look at the API documentation). We then use the path information to read the file into our pipeline and decode it using the jpg decoder (other decoders can be found in the API documentation).
```python
# create input queues
train_input_queue = tf.train.slice_input_producer(
                                    [train_images, train_labels],
                                    shuffle=False)
test_input_queue = tf.train.slice_input_producer(
                                    [test_images, test_labels],
                                    shuffle=False)

# process path and string tensor into an image and a label
file_content = tf.read_file(train_input_queue[0])
train_image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
train_label = train_input_queue[1]

file_content = tf.read_file(test_input_queue[0])
test_image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
test_label = test_input_queue[1]
```
#### Group Samples into Batches

If you run train_image in a session you would get a single image i.e. (28, 28, 1) since according to the dimensions of our mnist images. Training a model on single images can be inefficient which is why we would like to queue up images into a batch and perform our operations on a whole batch of images instead of a single one. We have yet to start our runners to load images. So far we have only described how the pipeline would look like and tensorflow doesn’t know the shape of our images. To use tf.train_batch we need to define the shape of our image tensors before they can be combined into batches. For this example, we will use a batch size of 5 samples.
```python
# define tensor shape
train_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
test_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])


# collect batches of images before processing
train_image_batch, train_label_batch = tf.train.batch(
                                    [train_image, train_label],
                                    batch_size=BATCH_SIZE
                                    #,num_threads=1
                                    )
test_image_batch, test_label_batch = tf.train.batch(
                                    [test_image, test_label],
                                    batch_size=BATCH_SIZE
                                    #,num_threads=1
                                    )
```
#### Run the Queue Runners and Start a Session

We have finished building our input pipeline. However, if we would now try to access e.g. test_image_batch, we would not get any data as we have not started the threads who will load the queues and push data into our tensorflow objects. After doing that, we will have two loops one going over the training data and one going over the test data. You will probably notice that the loops are bigger than the number of samples in each data set. What is the behaviour you expect?
```python
with tf.Session() as sess:
  
  # initialize the variables
  sess.run(tf.initialize_all_variables())
  
  # initialize the queue threads to start to shovel data
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  print "from the train set:"
  for i in range(20):
    print sess.run(train_label_batch)

  print "from the test set:"
  for i in range(10):
    print sess.run(test_label_batch)

  # stop our queue threads and properly close the session
  coord.request_stop()
  coord.join(threads)
  sess.close()
```
As you can see in the following output, tensorflow doesn’t care about epochs. We are not shuffling the data (see the parameter of the input slicer) and so our input pipeline will just cycle over the training data as often as it has to. It is your own responsibility to make sure that you correctly count the number of epochs. Play around with the batch size and shuffle and try to predict how this will change the output produced. Can you predict what will happen if you change the batch_size to 4 instead of 5?
```python
(tf-env)worker1:~$ python mnist_feed.py 
I tensorflow/stream_executor/dso_loader.cc:105] successfully opened CUDA library libcublas.so locally
I tensorflow/stream_executor/dso_loader.cc:105] successfully opened CUDA library libcudnn.so locally
I tensorflow/stream_executor/dso_loader.cc:105] successfully opened CUDA library libcufft.so locally
I tensorflow/stream_executor/dso_loader.cc:105] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:105] successfully opened CUDA library libcurand.so locally
input pipeline ready
I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:900] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
I tensorflow/core/common_runtime/gpu/gpu_init.cc:102] Found device 0 with properties: 
name: GeForce GTX 960
major: 5 minor: 2 memoryClockRate (GHz) 1.253
pciBusID 0000:01:00.0
Total memory: 2.00GiB
Free memory: 1.77GiB
I tensorflow/core/common_runtime/gpu/gpu_init.cc:126] DMA: 0 
I tensorflow/core/common_runtime/gpu/gpu_init.cc:136] 0:   Y 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:755] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 960, pci bus id: 0000:01:00.0)
from the train set:
[5 4 1 9 2]
[1 3 1 3 6]
[1 7 2 6 9]
[5 4 1 9 2]
[1 3 1 3 6]
[1 7 2 6 9]
[5 4 1 9 2]
[1 3 1 3 6]
[1 7 2 6 9]
[5 4 1 9 2]
[1 3 1 3 6]
[1 7 2 6 9]
[5 4 1 9 2]
[1 3 1 3 6]
[1 7 2 6 9]
[5 4 1 9 2]
[1 3 1 3 6]
[1 7 2 6 9]
[5 4 1 9 2]
[1 3 1 3 6]
from the test set:
[0 4 5 3 8]
[0 4 5 3 8]
[0 4 5 3 8]
[0 4 5 3 8]
[0 4 5 3 8]
[0 4 5 3 8]
[0 4 5 3 8]
[0 4 5 3 8]
[0 4 5 3 8]
[0 4 5 3 8]
```
Since we shuffled our partition vector you will obviously get different labels but what is important here is that you understand how the loading mechanism works. Since our batch size is as big as our test set every batch is the same.

#### Complete Code for this example

You need the script here (also linked above) to create the dataset and fix the path variables to the dataset at the beginning. This is the complete script.

```python
# Example on how to use the tensorflow input pipelines. The explanation can be found here ischlag.github.io.
import tensorflow as tf
import random
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

dataset_path      = "/path/to/your/dataset/mnist/"
test_labels_file  = "test-labels.csv"
train_labels_file = "train-labels.csv"

test_set_size = 5

IMAGE_HEIGHT  = 28
IMAGE_WIDTH   = 28
NUM_CHANNELS  = 3
BATCH_SIZE    = 5

def encode_label(label):
  return int(label)

def read_label_file(file):
  f = open(file, "r")
  filepaths = []
  labels = []
  for line in f:
    filepath, label = line.split(",")
    filepaths.append(filepath)
    labels.append(encode_label(label))
  return filepaths, labels

# reading labels and file path
train_filepaths, train_labels = read_label_file(dataset_path + train_labels_file)
test_filepaths, test_labels = read_label_file(dataset_path + test_labels_file)

# transform relative path into full path
train_filepaths = [ dataset_path + fp for fp in train_filepaths]
test_filepaths = [ dataset_path + fp for fp in test_filepaths]

# for this example we will create or own test partition
all_filepaths = train_filepaths + test_filepaths
all_labels = train_labels + test_labels

all_filepaths = all_filepaths[:20]
all_labels = all_labels[:20]

# convert string into tensors
all_images = ops.convert_to_tensor(all_filepaths, dtype=dtypes.string)
all_labels = ops.convert_to_tensor(all_labels, dtype=dtypes.int32)

# create a partition vector
partitions = [0] * len(all_filepaths)
partitions[:test_set_size] = [1] * test_set_size
random.shuffle(partitions)

# partition our data into a test and train set according to our partition vector
train_images, test_images = tf.dynamic_partition(all_images, partitions, 2)
train_labels, test_labels = tf.dynamic_partition(all_labels, partitions, 2)

# create input queues
train_input_queue = tf.train.slice_input_producer(
                                    [train_images, train_labels],
                                    shuffle=False)
test_input_queue = tf.train.slice_input_producer(
                                    [test_images, test_labels],
                                    shuffle=False)

# process path and string tensor into an image and a label
file_content = tf.read_file(train_input_queue[0])
train_image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
train_label = train_input_queue[1]

file_content = tf.read_file(test_input_queue[0])
test_image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
test_label = test_input_queue[1]

# define tensor shape
train_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
test_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])


# collect batches of images before processing
train_image_batch, train_label_batch = tf.train.batch(
                                    [train_image, train_label],
                                    batch_size=BATCH_SIZE
                                    #,num_threads=1
                                    )
test_image_batch, test_label_batch = tf.train.batch(
                                    [test_image, test_label],
                                    batch_size=BATCH_SIZE
                                    #,num_threads=1
                                    )

print "input pipeline ready"

with tf.Session() as sess:
  
  # initialize the variables
  sess.run(tf.initialize_all_variables())
  
  # initialize the queue threads to start to shovel data
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  print "from the train set:"
  for i in range(20):
    print sess.run(train_label_batch)

  print "from the test set:"
  for i in range(10):
    print sess.run(test_label_batch)

  # stop our queue threads and properly close the session
  coord.request_stop()
  coord.join(threads)
  sess.close()
  ```
