import tensorflow as tf


class MedicineNet:
    def __init__(self, input_dim, out_dim=2, lr=2e-4, drop_out=1):
        self.out_dim = out_dim
        self.inputs = tf.placeholder(tf.float32, [None, *input_dim], name='inputs')
        self.labels = tf.placeholder(tf.float32, [None, self.out_dim], name='outputs')
        self.lr = lr
        self.drop = drop_out
        self.net_out = self.create_net()
        self.loss, self.optimizer, self.accuracy, self.out, self.or_out = self.create_loss()

    def create_net(self):
        with tf.variable_scope('Convolution'):
            print(self.inputs)
            x1 = tf.layers.conv2d(self.inputs, 64, kernel_size=[5, 5], strides=[2, 2], padding='same',
                                  )
            print(x1)

            x2 = tf.layers.conv2d(x1, 128, kernel_size=[5, 5], strides=[2, 2], padding='same',
                                  activation=tf.nn.leaky_relu)
            print(x2)

            x3 = tf.layers.conv2d(x2, 256, kernel_size=[5, 5], strides=[2, 2], padding='same',
                                  activation=tf.nn.leaky_relu)
            print(x3)

            x4 = tf.layers.dropout(tf.layers.dense(tf.layers.flatten(x3), 512, activation=tf.nn.relu),
                                   rate=1.1 * self.drop)
            print(x4)

            x5 = tf.layers.dropout(tf.layers.dense(x4, 256, activation=tf.nn.relu), rate=0.7*self.drop)
            print(x5)

            x6 = tf.layers.dropout(tf.layers.dense(x5, 128, activation=tf.nn.relu), rate=0.6*self.drop)
            print(x6)

            x7 = tf.layers.dense(x6, self.out_dim)
            print(x7)

            return x7

    def create_loss(self):

        logits = self.net_out

        cost = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels)

        loss = tf.reduce_mean(cost)

        out = tf.nn.softmax(logits)
        print(out)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

        correct_predicts = tf.equal(tf.argmax(out, axis=1), tf.argmax(self.labels, axis=1))

        accuracy = tf.reduce_mean(tf.cast(correct_predicts, tf.float32))

        return loss, optimizer, accuracy, tf.argmax(out, axis=1), out


if __name__ == '__main__':
    model = MedicineNet((64, 64, 3))
