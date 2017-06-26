from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


mnsit = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b


y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy_cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.7 ).minimize(cross_entropy_cost_function)


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1000):
    X, Y = mnsit.train.next_batch(100)
    sess.run(train_step, feed_dict={x: X, y_: Y})

correct_prediction = tf.equal(tf.arg_max(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


print(sess.run(accuracy, feed_dict={x: mnsit.test.images, y_: mnsit.test.labels}))
