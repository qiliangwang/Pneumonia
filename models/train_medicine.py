import tensorflow as tf
from models.medicine_data import MedicineData
from models.medicine_model import MedicineNet
import numpy as np


def main():
    model = MedicineNet((64, 64, 3))
    data = MedicineData()

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    epochs = 100
    saver = tf.train.Saver()
    # saver.restore(sess, "/tmp/model.ckpt")
    print("Model restored.")
    for epoch in range(epochs):
        loss_lst = []
        for batch_x, batch_y in data.batch_iter(data_type='train', batch_size=200):
            train_dict = {model.inputs: batch_x, model.labels: batch_y}
            _, loss = sess.run([model.optimizer, model.loss], feed_dict=train_dict)
            loss_lst.append(loss)
            print('Iterator Loss:{}/{}'.format(epoch, epochs),
                  'Train losses: {:.4f}'.format(loss))
        losses = np.mean(loss_lst)
        print('Epoch Loss:{}/{}'.format(epoch, epochs),
              'Train losses: {:.4f}'.format(losses))

        if epoch % 5 == 0:
            acc_lst = []
            for test_x, test_y in data.batch_iter(data_type='test', batch_size=200):
                test_dict = {model.inputs: test_x, model.labels: test_y}
                acc = sess.run([model.accuracy], feed_dict=test_dict)
                acc_lst.append(acc[0])
            acc = np.mean(acc_lst)
            print('Epoch:{}/{}'.format(epoch, epochs),
                  'Test acc: {:.4f}'.format(acc))
            save_path = saver.save(sess, "./tmp/model.ckpt")
            print("Model saved in path: %s" % save_path)
    sess.close()


if __name__ == '__main__':
    main()
