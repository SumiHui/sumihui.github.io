---
title:	google tutorials cifar10/cifar10_train.py
date: 2017-10-23 08:44:00
categories: [tensorflow,cnn]
tags: [tensorflow,python,cnn,cifar10]
---

### 1. Task [通过该源码的阅读将掌握什么？]
- [O] TF单GPU运行计算图的方式.To train CIFAR-10 using a single GPU
- [O] TF底层方法、高级特性实现神经网络的方式深入理解掌握

<!-- more -->

### Overview
CIFAR-10 classification is a common benchmark problem in machine learning. The problem is to classify RGB 32x32 pixel images across 10 categories:

> airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

A binary to train CIFAR-10 using a single GPU.

**Accuracy:**
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of data) as judged by cifar10_eval.py.

**Speed:** With batch_size 128.

|	System        | 	Step Time (sec/batch)  |     Accuracy	|
| --- | --- | --- | --- |
| 1 Tesla K20m  |  0.35-0.60       | ~86% at 60K steps  (5 hours)|
| 1 Tesla K40m  |  0.25-0.35       | ~86% at 100K steps (4 hours)|

**Usage:**
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/

### src code

这部分代码属于模型训练部分，模型定义部分请移步搜站内文章 [**TutorialsCifar10Cifar10**]


| mark	|	File	|	Purpose	|
| ---   | --- | --- |
| [x]   | cifar10_input.py	| Reads the native CIFAR-10 binary file format.	|
| [x]   |cifar10.py	| Builds the CIFAR-10 model.	|
| [O]   |cifar10_train.py	| Trains a CIFAR-10 model on a CPU or GPU.	|
| [x]   |cifar10_multi_gpu_train.py	| Trains a CIFAR-10 model on multiple GPUs.	|
| [x]   |cifar10_eval.py	| Evaluates the predictive performance of a CIFAR-10 model.	|


下面来分析`cifar10_train.py`源码：[cifar10_train.py link](https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_train.py)

-----

#### src_0 - 命令行参数解析设定：
```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf

import cifar10

parser = cifar10.parser

parser.add_argument('--train_dir', type=str, default='/tmp/cifar10_train',
                    help='Directory where to write event logs and checkpoint.')

parser.add_argument('--max_steps', type=int, default=1000000,
                    help='Number of batches to run.')

parser.add_argument('--log_device_placement', type=bool, default=False,
                    help='Whether to log device placement.')

parser.add_argument('--log_frequency', type=int, default=10,
                    help='How often to log results to the console.')
```
导入必要的模块，其中，`cifar10.py`模块是上篇文章讲过的模块（在其中定义了网络模型），此处与`cifar10.py`不同的是，直接利用了cifar10中的parser = argparse.ArgumentParser()传递命令行参数。命令行参数解析模块`argparse`上次已经提过了

-----

#### src_1 - train()：
```python
def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      images, labels = cifar10.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images)

    # Calculate loss.
    loss = cifar10.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)
```
> `tf.Graph().as_default()`:
> * Returns a context manager that makes this Graph the default graph.

==本例中，Hook的内容先不展开讲，MonitoredTrainingSession参数填写等留坑==

> `tf.train.SessionRunHook`:
> > class SessionRunHook
> > * Hook to extend calls to MonitoredSession.run()

在Hook中一般会编写类似于` if x: then do something `这样功能的代码。用来监控、分析、增加额外的功能。
`SessionRunHook`类中有如下5个方法：
> * `after_create_session(session,coord)`

> * `after_run(run_context,run_values)`
>  Called after each call to run().
>>The run_values argument contains results of requested ops/tensors by before_run().
The run_context argument is the same one send to before_run call. run_context.request_stop() can be called to stop the iteration.
If session.run() raises any exceptions then after_run() is not called.

> * `before_run(run_context)`
>  Called before each call to run().
>>You can return from this call a SessionRunArgs object indicating ops or tensors to add to the upcoming run() call. These ops/tensors will be run together with the ops/tensors originally passed to the original run() call. The run args you return can also contain feeds to be added to the run() call.
The run_context argument is a SessionRunContext that provides information about the upcoming run() call: the originally requested op/tensors, the TensorFlow Session.
At this point graph is finalized and you can not add ops.

> * `begin()`
>  Called once before using the session.
>>When called, the default graph is the one that will be launched in the session. The hook can modify the graph by adding new operations to it. After the begin() call the graph will be finalized and the other callbacks can not modify the graph anymore. Second call of begin() on the same graph, should not change the graph.
> * `end(session)`

以上只列出本例中出现的方法的解释。


> `tf.train.MonitoredTrainingSession`:  Creates a MonitoredSession for training.
该方法参数多，仅介绍本例中该方法中出现的几个参数
> * `checkpoint_dir`: A string. Optional path to a directory where to restore variables.
> * `hooks`: Optional list of SessionRunHook objects.
> * `config`: an instance of tf.ConfigProto proto used to configure the session. It's the config argument of constructor of tf.Session.
> * `log_step_count_steps`: The frequency, in number of global steps, that the global step/sec is logged.

本例中，设置了训练完成后的网络节点保存保存路径在FLAGS.train_dir中，

----

#### src_2 - main(argv=None)：

```python
def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()
```
主调函数内，第一行判断指定的数据集地址内是否有所需数据，如没有则下载并解压输出，否则调用训练函数开始训练，当然了，本例中会先判定用于存取事件日志和节点保存的路径是否存在，若有，则递归删除路径下所有内容，若没有，直接创建新的存取文件路径。

----

#### src\_3 - \_\_name\_\_ == '\_\_main\_\_'：
```python
if __name__ == '__main__':
  FLAGS = parser.parse_args()
  tf.app.run()
```

FLAGS已经是导入的模块`cifar10.py`中定义过的变量，通过其接收解析的命令行参数。

> ```python
# tf.app
run(main=None,argv=None)
# Runs the program with an optional 'main' function and 'argv' list.
```

`tf.app`是通用入口点脚本，通过`run()`方法执行程序
`tf.app.flags` module: Implementation of the flags interface.
TensorFlow项目例子中经常出现`tf.app.flags`，它支持应用从命令行接受参数，可以用来指定集群配置等.
分布式通过`tf.app.run()`运行 ， `main()`调用的时候必须填写一个参数，请看下面例子：

举一个简单的`tf.app.flags`例子，可以指定多个参数和不同默认值：
```python
# tf.app.flags example
import tensorflow as tf

flags=tf.app.flags
flags.DEFINE_string(flag_name="data_path",default_value='/home/sumihui/dataset',docstring='get directory')
flags.DEFINE_string("distribute_model",False,'run in distribute model or not')

FLAGS=flags.FLAGS

def main(x):	# 此处的 x 不可获缺，必须有接收参数的变量，可以使用任意其他符合变量命名规则的符号替代 x（如下划线等）
	FLAGS.distribute_model = True
    print FLAGS.data_path
    print FLAGS.distribute_model

if __name__=='__main__':
    tf.app.run()
```
输出结果：
```python
/home/sumihui/dataset
True
```

执行`main`函数之前首先进行flags的解析，也就是说TensorFlow通过设置flags来传递`tf.app.run()`所需要的参数，我们可以直接在程序运行前初始化flags，也可以在运行程序的时候设置命令行参数来达到传参的目的。



------



