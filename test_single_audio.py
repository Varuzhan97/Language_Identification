import argparse
import numpy as np
import os
import sys
from tensorflow.keras.models import load_model
from Utils.spectrogram import audio_to_spectrogram


def predict(cli_args):

    class_labels = ["HY", "AS"]
    pixel_per_second = 50
    image_width = 500
    image_height = 129
    num_classes = 2
    sample_rate = 16000

    images = audio_to_spectrogram(cli_args.input_file, sample_rate, pixel_per_second, image_width, image_height, True)
    data = [np.divide(image, 255.0) for image in images]
    data = np.stack(data)

    # Model Generation
    model = load_model(cli_args.model_dir)

    probabilities = model.predict(data, verbose = 0)

    #classes = np.argmax(probabilities, axis=1)
    average_prob = np.mean(probabilities, axis=0)
    average_class = np.argmax(average_prob)

    print(class_labels[average_class])
    return probabilities

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model_dir', required=True)
    parser.add_argument('--input', dest='input_file', required=True)
    cli_args = parser.parse_args()

    if not os.path.isfile(cli_args.input_file):
        sys.exit("Input is not a file.")


    predict(cli_args)
