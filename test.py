import argparse
import numpy as np
import os
import sys
from tensorflow.keras.models import load_model
import spectrogram

class_labels = ["BN", "CN"]

def predict(cli_args):

    config = {"pixel_per_second": 50, "input_shape": [129, 500, 1], "num_classes": 2}

    image = spectrogram.audio_to_image_file(cli_args.input_file, 16000, 500, 129)
    #if len(image.shape) == 2:
    #    image = np.expand_dims(image, -1)
    image = np.expand_dims(image, -1)
    image = np.expand_dims(image, -1)
    data = np.divide(image, 255.0)
    #data_generator = SpectrogramGenerator(cli_args.input_file, config, shuffle=False, run_only_once=True).get_generator()
    #data = [np.divide(image, 255.0) for image in data_generator]
    data = np.stack(data)

    # Model Generation
    model = load_model(cli_args.model_dir)

    probabilities = model.predict(data)

    classes = np.argmax(probabilities, axis=1)
    average_prob = np.mean(probabilities, axis=0)
    average_class = np.argmax(average_prob)

    print(classes, class_labels[average_class], average_prob)
    return probabilities

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model_dir', required=True)
    parser.add_argument('--input', dest='input_file', required=True)
    cli_args = parser.parse_args()

    if not os.path.isfile(cli_args.input_file):
        sys.exit("Input is not a file.")


    predict(cli_args)
