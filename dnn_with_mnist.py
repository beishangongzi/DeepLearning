import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

# 这是对日志的描述
# namespace tensorflow {
# const int INFO = 0;            // base_logging::INFO;
# const int WARNING = 1;         // base_logging::WARNING;
# const int ERROR = 2;           // base_logging::ERROR;
# const int FATAL = 3;           // base_logging::FATAL;
# const int NUM_SEVERITIES = 4;  // base_logging::NUM_SEVERITIES;
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}


def mnist_dataset():
    """
    在keras的datasets中有boston_housing cifar10 cifar100 fashion_mnist imdb mnist retuers
    """
  (x, y), _ = datasets.mnist.load_data()
  ds = tf.data.Dataset.from_tensor_slices((x, y))
  ds = ds.map(prepare_mnist_features_and_labels)
# take 最多有多少数据 shuffle 需要大于数据量的大小 batch 每个batch有100个，有参数可以选择放弃不足100的数据
  ds = ds.take(20000).shuffle(20000).batch(100)
  return ds

# function 在构建计算图中使用，具体还不了解，但是应该可以增加优先级
@tf.function
def prepare_mnist_features_and_labels(x, y):
  x = tf.cast(x, tf.float32) / 255.0
  y = tf.cast(y, tf.int64)
  return x, y


model = keras.Sequential([
    layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(10)])

optimizer = optimizers.Adam()


@tf.function
def compute_loss(logits, labels):

# tf.nn.sparse_softmax_cross_entropy_with_logits 是用来计算交叉熵的，因为是softmax函数，所以y是一维列表，而logits是概率数组，是二维的
  return tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=labels))

@tf.function
def compute_accuracy(logits, labels):
    # argmax 可以找到哪个可能性最大
  predictions = tf.argmax(logits, axis=1)
  return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))

@tf.function
def train_one_step(model, optimizer, x, y):
	# 用来计算导数的时候会用到
  with tf.GradientTape() as tape:
    logits = model(x)
    loss = compute_loss(logits, y)

  # compute gradient
  grads = tape.gradient(loss, model.trainable_variables)
  # update to weights
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  accuracy = compute_accuracy(logits, y)

  # loss and accuracy is scalar tensor
  return loss, accuracy


def train(epoch, model, optimizer):

  train_ds = mnist_dataset()
  loss = 0.0
  accuracy = 0.0
  for step, (x, y) in enumerate(train_ds):
    loss, accuracy = train_one_step(model, optimizer, x, y)

    if step % 500 == 0:
      print('epoch', epoch, ': loss', loss.numpy(), '; accuracy', accuracy.numpy())

  return loss, accuracy


for epoch in range(20):
  loss, accuracy = train(epoch, model, optimizer)

print('Final epoch', epoch, ': loss', loss.numpy(), '; accuracy', accuracy.numpy())
