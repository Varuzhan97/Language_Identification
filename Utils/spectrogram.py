import numpy as np
import librosa as lr
import imageio
import os
import soundfile as sf
import random
from subprocess import Popen, PIPE, STDOUT

#Function to repeat the audio segment to a length of 10 seconds
def fix_audio_segment_to_10_seconds(file_path, sample_rate):
    audio_segment, _ = lr.load(file_path, sr=sample_rate)
    target_len = 10 * sample_rate
    audio_segment = np.concatenate([audio_segment]*3, axis=0)
    audio_segment = audio_segment[0:target_len]

    #Delete old file
    os.remove(file_path)
    #Save new file(with length of 10 seconds)
    sf.write(file_path, audio_segment, sample_rate)

def audio_to_spectrogram(audio_file, sample_rate, pixel_per_sec, image_width, image_height):
    '''
    V0 - Verbosity level: ignore everything
    c 1 - channel 1 / mono
    n - apply filter/effect
    rate 10k - limit sampling rate to 10k --> max frequency 5kHz (Shenon Nquist Theorem)
    y - small y: defines height
    X capital X: defines pixels per second
    m - monochrom
    r - no legend
    o - output to stdout (-)
    '''
    try:
        fix_audio_segment_to_10_seconds(audio_file, sample_rate)
        #Generate temporary spectrogram
        temp_image_file = audio_file + '_temp.png'
        command = "sox -V0 '{}' -n remix 1 rate 10k spectrogram -y {} -X {} -m -r -o {}".format(audio_file, image_height, pixel_per_sec, temp_image_file)
        p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)

        output, errors = p.communicate()
        p.wait()
        if errors:
            print(errors)

        image = imageio.imread(temp_image_file)
        #Remove temporary spectrogram
        os.remove(temp_image_file)

        #Add dimension for mono channel
        image = np.expand_dims(image, -1)

        height, width, channels = image.shape

        assert image_height == height, "Heigh mismatch {} vs {}".format(image_height, height)

        num_segments = width // image_width

        j = 0;
        for i in range(0, num_segments):
            image_file_path = audio_file.rsplit( ".", 1)[0] + "_" + str(j) + ".png"

            slice_start = i * image_width
            slice_end = slice_start + image_width

            slice = image[:, slice_start:slice_end]

            # Ignore black images
            if slice.max() == 0 and slice.min() == 0:
                continue

            imageio.imwrite(image_file_path, slice)
            yield image_file_path
            j+=1
    except Exception as e:
                print("SpectrogramGenerator Exception: ", e, audio_file)
                pass
