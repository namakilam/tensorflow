from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnsit = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)

def init_weights(shape):
    weights = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def init_bias(shape):
    bias = tf.constant(0.1, shape=shape)
    return tf.Variable(bias)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_l1 = init_weights([5, 5, 1, 32])
B_l1 = init_bias([32])
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])
o_c1 = tf.nn.relu(conv2d(x_image, W_l1) + B_l1)
o_p1 = max_pool_2x2(o_c1)

W_l2 = init_weights([5, 5, 32, 64])
B_l2 = init_bias([64])

o_c2 = tf.nn.relu(conv2d(o_p1, W_l2) + B_l2)
o_p2 = max_pool_2x2(o_c2)

W_l3 = init_weights([7 * 7 * 64, 1024])
B_l3 = init_bias([1024])

o_p2_flat = tf.reshape(o_p2, [-1, 7*7*64])
o_l3 = tf.nn.relu(tf.matmul(o_p2_flat, W_l3) + B_l3)

W_l4 = init_weights([1024, 10])
B_l4 = init_bias([10])

y_conv = tf.matmul(o_l3, W_l4) + B_l4

cross_entropy_cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_cost_function)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(10000):
    batch = mnsit.train.next_batch(50)
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1]})
      print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

  print('test accuracy %g' % accuracy.eval(feed_dict={
      x: mnsit.test.images, y_: mnsit.test.labels}))
