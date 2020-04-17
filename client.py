#分布式TensorFlow 调用程序
import tensorflow as tf

if __name__ == "__main__":
    with tf.device("/job:ps/task:0"):
        x = tf.Variable(tf.ones([2, 2]))
        y = tf.Variable(tf.ones([2, 2]))

    with tf.device("/job:worker/task:0"):
        z = tf.matmul(x, y) + x

    with tf.device("/job:worker/task:1"):
        z = tf.matmul(z, x) + x

    with tf.Session("grpc://localhost:2223") as sess:
        sess.run(tf.global_variables_initializer())
        val = sess.run(z)
        print(val)
