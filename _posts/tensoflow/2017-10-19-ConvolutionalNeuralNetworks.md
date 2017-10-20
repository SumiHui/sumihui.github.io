---
title:	google官方实现的cifar10卷积神经网络源码解析学习
date: 2017-10-19 19:44:00
categories: [tensorflow,cnn]
tags: [tensorflow,python,cnn]
---

### 1. 任务表 [通过该源码的阅读将掌握什么？]
- [x] TF多GPU运行计算图的方式
- [x] TF底层方法、高级特性实现神经网络的方式深入理解掌握
- [x] InceptionV3源码阅读，注意修改其bash文件（剔除下载imagenet的代码）

Use TensorFlow to train and evaluate a convolutional neural network (CNN) on both CPU and GPU

<!-- more -->

### Overview
CIFAR-10 classification is a common benchmark problem in machine learning. The problem is to classify RGB 32x32 pixel images across 10 categories:

> airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

### src code
build a relatively small convolutional neural network (CNN) for recognizing images. 

使用到的API：
tf.nn.conv2d

```python
conv2d(
    input,
    filter,
    strides,
    padding,
    use_cudnn_on_gpu=None,
    data_format=None,
    name=None
)
```
Must have strides[0] = strides[3] = 1. For the most common case of the same horizontal and vertices strides, strides = [1, stride, stride, 1].

**Args:**
> * input: A Tensor. Must be one of the following types: half, float32. A 4-D tensor. The dimension order is interpreted according to the value of data_format, see below for details.
> * filter: A Tensor. Must have the same type as input. A 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels]
> * strides: A list of ints. 1-D tensor of length 4. The stride of the sliding window for each dimension of input. The dimension order is determined by the value of data_format, see below for details.
> * padding: A string from: "SAME", "VALID". The type of padding algorithm to use.
use_cudnn_on_gpu: An optional bool. Defaults to True.
> * data_format: An optional string from: "NHWC", "NCHW". Defaults to "NHWC". Specify the data format of the input and output data. With the default format "NHWC", the data is stored in the order of: [batch, height, width, channels]. Alternatively, the format could be "NCHW", the data storage order of: [batch, channels, height, width].
> * name: A name for the operation (optional).

**Returns:**
> A Tensor. Has the same type as input. A 4-D tensor. The dimension order is determined by the value of data_format, see below for details.


google 官方提供的cifar-10训练网络实现过程大致如下：

> * Model inputs
> * Model prediction
> * Model training

下面来一一分析`cifar10.py`源码：
[cifar10.py link](https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py)

---
#### 源码段 - 导入模块：
```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import cifar10_input
```
导入必要的模块，其中，cifar10_input模块是google自己实现的读取cifar10-binary数据的模块

-----

#### 源码段 - 命令行参数解析设定：
```python
parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--batch_size', type=int, default=128,
                    help='Number of images to process in a batch.')

parser.add_argument('--data_dir', type=str, default='/tmp/cifar10_data',
                    help='Path to the CIFAR-10 data directory.')

parser.add_argument('--use_fp16', type=bool, default=False,
                    help='Train the model using fp16.')

FLAGS = parser.parse_args()
```
`argparse`：Command-line parsing library.
这个是python标准库推荐使用的命令行参数解析模块！
上面这段代码先后设置了batch、data_dir、use_fp16,并从命令行解析命令赋值给了`FLAGS`变量.`fp16`是半精度浮点数`Half-precision floating-point`的简称

-----

**关于argparse的一些简短介绍：**
* 代码1：
```python
import argparse
parser=argparse.ArgumentParser()
parser.parse_args()
```
这是一个没有任何自定义的文件，运行该文件无结果。

* 代码2：
```python
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('batch_size', type=int) # 定义了位置参数batch_size,则运行时命令行中一定要有参数；不加type=int的话，argparse会将输入当作字符串处理
args=parser.parse_args()
print(args.batch_size)
```
运行该代码：
```python
[sumihui@DLIG cifar10]$ python argparseTest.py 128
128
```
* 代码3：
```python
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int) # 定义了可选参数batch_size,则运行时命令行中参数可有可无
args=parser.parse_args()
print(args.batch_size)
```
运行该代码：
```python
[sumihui@DLIG cifar10]$ python argparseTest.py
None
[sumihui@DLIG cifar10]$ python argparseTest.py --batch_size 128
128
```
大致了解了这些过后我们再继续往下看

-----

#### 源码段 - 全局变量设定：
```python
# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
```
这个提一下`cifar10_input.py`，在`cifar10_input.py`模块中定义了如下的全局变量：
```python
# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24
# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
```
------

#### 源码段 - _activation_summary：
```python
def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))
```
这段代码相对于我们的网络模型来说不是重点，仅仅用作记录和统计模型训练过程中的传入信息x。也就是说代码中出现这个方法的地方都可以删除之。

-----

#### 源码段 - _variable_on_cpu：
```python
def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var
```
看完注释是明白作了什么了，但--为何要这么做呢？
这个就涉及到Tensorflow的多GPU训练了，将初始化、控制、更新参数等行为交由CPU负责，充分利用好GPU在高精度浮点型计算和密集型计算上的优势。
关于这个会专门再**【辟文讲解，先mark了】**

-----

#### 源码段 - _variable_with_weight_decay：
```python
def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var
```
**L2 Loss**:Computes half the L2 norm of a tensor without the sqrt.
```python
output=sum(t**2)/2
```
对于`add_to_collection(name,value)`，官网API的解释为：
> Stores `value` in the collection with the given `name`.
> Note that collections are not sets, so it is possible to add a value to a collection several times.
这个操作是将执行权重衰减操作后（如果有的话），也就是更新后的权重值加入到计算图中吗？《=== 这句话有待考证，先往后看
这个函数实际上返回的就是**巻积核**！！

-----

#### 源码段 - distorted_inputs：
```python
def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                  batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels
```
google官方在实现cifar10的巻积时，用的是从数据集网站下载的`cifar-10-binary.tar.gz`，网站实际上还专门提供了python版本的数据集。如果自己已有数据，则可以将google默认的`/tmp/cifar10_data`改为自己的数据存放路径，但是使用google的代码时，数据就必须用的是二进制的数据集。
这段代码用于将输入图片转为4D张量，标签转为1D张量。

------

#### 源码段 - inputs：
```python
def inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.
  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels
```
同上

-----

#### 源码段 - inference：
{% highlight python linenos %}
def inference(images):
  """Build the CIFAR-10 model.
  Args:
    images: Images returned from distorted_inputs() or inputs().
  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 3, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear
{% endhighlight %}

前面已经说过了，用于记录和统计信息的函数`_activation_summary(x)`先不予理会（比如line 23），
重点关注下`tf.nn.lrn`:
```python
local_response_normalization(
    input,
    depth_radius=None,
    bias=None,
    alpha=None,
    beta=None,
    name=None
)
```
google真坑，在APIr1.3里搜索不到这个，反倒是可以通过搜`tf.nn.local_response_normalization`搜到:bowtie:，整个方法返回值和输入值类型一致。
**Local Response Normalization.**

> The 4-D input tensor is treated as a 3-D array of 1-D vectors (along the last dimension), and each vector is normalized independently. Within a given vector, each component is divided by the weighted, squared sum of inputs within depth_radius. In detail,

> sqr_sum[a, b, c, d] =
>     sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
> output = input / (bias + alpha * sqr_sum) ** beta

目前对这个的翻译都译为“局部响应归一化”，最早是由Krizhevsky和Hinton在关于ImageNet的论文里面使用的一种数据标准化方法，即使现在，也依然会有不少CNN网络会使用到这种正则手段，

![local_response_normalization](https://sumihui.github.io/source/images/201710200804lrn.png)

以上是这种归一手段的公式，其中`a`的上标指该层的第几个`feature map`，`a`的下标`x`，`y`表示`feature map的像素位置`，`N`指`feature map的总数量`，公式里的其它参数都是超参（在机器学习的上下文中，超参数是在开始学习过程之前设置值的参数，而不是通过训练得到的参数数据），需要自己指定的。
这种方法是受到神经科学的启发，激活的神经元会抑制其邻近神经元的活动（侧抑制现象），至于为什么使用这种正则手段，以及它为什么有效，查阅了很多文献似乎也没有详细的解释，可能是由于后来提出的batch normalization手段太过火热，渐渐的就把local response normalization掩盖了吧。
> `depth_radius`：这个值需要自己指定，就是上述公式中的n/2
> `bias`：上述公式中的 k , An offset (usually positive to avoid dividing by 0).
> `alpha`：上述公式中的 α
> `beta`：上述公式中的 β

-----

#### 源码段 - loss：
```python
def loss(logits, labels):
  """Add L2Loss to all the trainable variables.
  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')
```
------

#### 源码段 - _add_loss_summaries：
```python
def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.
  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op
```
------

#### 源码段 - train：
```python

def train(total_loss, global_step):
  """Train CIFAR-10 model.
  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.
  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op
```
------

#### 源码段 - maybe_download_and_extract:
```python

def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = FLAGS.data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
  if not os.path.exists(extracted_dir_path):
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

```


------



