import argparse
import numpy as np
import os
import sys
from tensorflow.keras.models import load_model
from Utils.spectrogram import audio_to_spectrogram
from Utils.record import listen_audio

def predict(cli_args):

    class_labels = ["HY", "BN", 'RU']
    pixel_per_second = 50
    image_width = 500
    image_height = 129
    num_classes = 2
    sample_rate = 16000

    # Model Generation
    model = load_model(cli_args.model_dir)

    for speech in listen_audio():
        images = audio_to_spectrogram(speech, sample_rate, pixel_per_second, image_width, image_height, True)
        data = [np.divide(image, 255.0) for image in images]
        data = np.stack(data)

        probabilities = model.predict(data, verbose=0)

        #classes = np.argmax(probabilities, axis=1)
        average_prob = np.mean(probabilities, axis=0)
        average_class = np.argmax(average_prob)

        print(class_labels[average_class])
        os.remove(speech)
    return probabilities

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model_dir', required=True)
    cli_args = parser.parse_args()

    predict(cli_args)
