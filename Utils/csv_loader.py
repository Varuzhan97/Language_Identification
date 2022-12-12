import numpy as np
import csv

import imageio

from tensorflow.keras.utils import to_categorical

class CSVLoader(object):

    def __init__(self, data_dir, batch_size, num_classes, input_shape):

        self.batch_size = batch_size
        self.num_classes = num_classes
        self.input_shape = input_shape

        self.images_label_pairs = []

        with open(data_dir, "r") as csvfile:
            for (file_path, label)in list(csv.reader(csvfile)):
                self.images_label_pairs.append((file_path, int(label)))

    def get_data(self, should_shuffle=True, is_prediction=False):

        start = 0

        while True:

            data_batch = np.zeros((self.batch_size, ) + self.input_shape)  # (batch_size, cols, rows, channels)
            label_batch = np.zeros((self.batch_size, self.num_classes))  # (batch_size,  num_classes)

            for i, (file_path, label) in enumerate(self.images_label_pairs[start:start + self.batch_size]):

                data = self.process_file(file_path)
                height, width, channels = data.shape
                data_batch[i, : height, :width, :] = data
                label_batch[i, :] = to_categorical([label], num_classes=self.num_classes) # one-hot encoding

            start += self.batch_size

            # Reset generator
            if start + self.batch_size > self.get_num_files():
                start = 0
                if should_shuffle:
                    np.random.shuffle(self.images_label_pairs)

            # For predicitions only return the data
            if is_prediction:
                yield data_batch
            else:
                yield data_batch, label_batch

    def get_input_shape(self):

        return self.input_shape

    def get_num_files(self):

        # Minimum number of data points without overlapping batches
        return (len(self.images_label_pairs) // self.batch_size) * self.batch_size


    def get_labels(self):

        return [label for (file_path, label) in self.images_label_pairs]


    def process_file(self, file_path):
        image = imageio.imread(file_path, pilmode='L')

        # Image shape should be (cols, rows, channels)
        if len(image.shape) == 2:
            image = np.expand_dims(image, -1)

        assert len(image.shape) == 3

        return np.divide(image, 255.0)  # Normalize images to 0-1.0
