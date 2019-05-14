import os
import cv2 as cv
import numpy as np
import random


class MedicineData:
    def __init__(self, data_dir=os.path.join(os.getcwd(), 'data')):

        self.train_ratio = 0.8

        self.data_dir = data_dir

        self.normal_folder_list = os.path.join(data_dir, 'pos')

        self.pneumonia_folder_list = os.path.join(data_dir, 'neg')

        self.normal_data = self.merge_folder_list(self.normal_folder_list)

        self.pneumonia_data = self.merge_folder_list(self.pneumonia_folder_list)

        self.train_normal, self.train_pneumonia, self.test_normal, self.test_pneumonia = self.build_data()

        self.train_normal_labels, self.train_pneumonia_labels, self.test_normal_labels, self.test_pneumonia_labels = self.build_labels()

    @staticmethod
    def merge_folder_list(folder):

        simple_data_list = os.listdir(folder)

        data_list = [os.path.join(folder, data) for data in simple_data_list]

        remove_str = '.DS_Store'

        return [data for data in data_list if not data.endswith(remove_str)]

    def build_data(self):

        random.shuffle(self.normal_data)
        random.shuffle(self.pneumonia_data)

        normal_size = int(len(self.normal_data) * self.train_ratio)
        pneumonia_size = int(len(self.pneumonia_data) * self.train_ratio)

        train_normal = self.normal_data[:normal_size]
        train_pneumonia = self.pneumonia_data[:pneumonia_size]

        test_normal = self.normal_data[normal_size:]
        test_pneumonia = self.pneumonia_data[pneumonia_size:]

        return train_normal, train_pneumonia, test_normal, test_pneumonia

    def build_labels(self):
        # [1, 0] for normal, [0, 1] for pneumonia
        train_normal_labels = [[1, 0] for _ in self.train_normal]

        train_pneumonia_labels = [[0, 1] for _ in self.train_pneumonia]

        test_normal_labels = [[1, 0] for _ in self.test_normal]

        test_pneumonia_labels = [[0, 1] for _ in self.test_pneumonia]

        return train_normal_labels, train_pneumonia_labels, test_normal_labels, test_pneumonia_labels

    @staticmethod
    def convert_img(img_dirs):

        def parse_img(img_dir):
            img = cv.imread(img_dir) / 255.0
            img = cv.resize(img, (64, 64))
            img = img.reshape(-1, 64, 64, 3)
            return img

        images = [parse_img(img_dir) for img_dir in img_dirs]
        return np.concatenate(images)

    def batch_iter(self, batch_size, data_type='train', shuffle=True):

        data = np.array(self.train_normal + self.train_pneumonia)
        labels = np.array(self.train_normal_labels + self.train_pneumonia_labels)

        if data_type == 'train':
            pass

        if data_type == 'test':
            data = np.array(self.test_normal + self.test_pneumonia)
            labels = np.array(self.test_normal_labels + self.test_pneumonia_labels)

        data_size = len(data)
        num_batches = ((data_size-1) // batch_size) + 1

        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffle_x = data[shuffle_indices]
            shuffle_y = labels[shuffle_indices]
        else:
            shuffle_x = data
            shuffle_y = labels

        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            batch_x = self.convert_img(shuffle_x[start_index: end_index])
            batch_y = shuffle_y[start_index: end_index]

            yield batch_x, batch_y
        pass


def main():
    data_path = os.path.join(os.getcwd(), 'data')
    data = MedicineData(data_path)
    for batch_x, batch_y in data.batch_iter(50):
        print(batch_x.shape, batch_y.shape)


if __name__ == '__main__':
    main()
