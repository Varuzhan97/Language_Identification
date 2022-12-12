import csv
import os
import subprocess
from multiprocessing import Pool
from collections import Counter
import progressbar
import sox
import yaml
from random import shuffle
from shutil import copy2
import pandas as pd

from Utils.augment import augment_audio_file_with_noise
from Utils.spectrogram import audio_to_spectrogram
from global_parameters import import_parameters


SIMPLE_BAR = [
    "Progress ",
    progressbar.Bar(),
    " ",
    progressbar.Percentage(),
    " completed",
]

def secs_to_hours(secs):
    hours, remainder = divmod(secs, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "%d:%02d:%02d" % (hours, minutes, seconds)

def get_counter():
    return Counter(
        {
            "all": 0,
            "failed": 0,
            "too_short": 0,
            "too_long": 0,
            "imported_time": 0,
            "total_time": 0,
        }
    )

def get_imported_samples(counter):
    return (
        counter["all"]
        - counter["failed"]
        - counter["too_short"]
        - counter["too_long"]
    )

def print_import_report(counter):
    print("Imported %d samples." % (get_imported_samples(counter)))
    if counter["failed"] > 0:
        print("Skipped %d samples that failed upon conversion." % counter["failed"])
    if counter["too_short"] > 0:
        print("Skipped %d samples that were too short to match the transcript." % counter["too_short"])
    if counter["too_long"] > 0:
        print("Skipped %d samples that were longer than %d seconds." % (counter["too_long"], import_parameters['max_secs']))
    print("Final amount of imported audio: %s from %s." % (secs_to_hours(counter["imported_time"] / import_parameters['sample_rate']), secs_to_hours(counter["total_time"] / import_parameters['sample_rate'])))

def one_sample(sample):
    """Take an audio file, and optionally convert it to 16kHz WAV"""
    mp3_filename = sample
    if not os.path.splitext(mp3_filename.lower())[1] == ".mp3":
        mp3_filename += ".mp3"
    #Storing wav files next to the mp3 ones - just with a different suffix
    wav_filename = os.path.splitext(mp3_filename)[0] + ".wav"
    _maybe_convert_wav(mp3_filename, wav_filename)
    #Remove mp3 file
    os.remove(mp3_filename)
    frames = 0
    if os.path.exists(wav_filename):
        frames = int(subprocess.check_output(["soxi", "-s", wav_filename], stderr=subprocess.STDOUT))

    rows = []
    counter = get_counter()

    if frames / import_parameters['sample_rate'] < import_parameters['min_secs']:
        #Excluding very short samples to keep a reasonable batch-size
        counter["too_short"] += 1
        #Remove invalid wav file
        os.remove(wav_filename)
    elif frames / import_parameters['sample_rate'] > import_parameters['max_secs']:
        #Excluding very long samples to keep a reasonable batch-size
        counter["too_long"] += 1
        #Remove invalid wav file
        os.remove(wav_filename)
    else:
        #This one is good - keep it for the target CSV
        rows = [os.path.split(wav_filename)[-1]]
        counter["imported_time"] += frames
    counter["all"] += 1
    counter["total_time"] += frames
    return (counter, rows)


def _maybe_convert_set(tsv_dir, audio_dir, language_label):
    rows = []
    input_tsv = os.path.join(os.path.abspath(tsv_dir), "validated.tsv")
    if not os.path.isfile(input_tsv):
        return rows
    print("Loading TSV file: ", input_tsv)
    #Get audiofile path and transcript for each sentence in tsv
    samples = []
    with open(input_tsv, encoding="utf-8") as input_tsv_file:
        reader = csv.DictReader(input_tsv_file, delimiter="\t")
        for row in reader:
            samples.append(os.path.join(audio_dir, row["path"]))

    counter = get_counter()
    num_samples = len(samples)

    print("Importing mp3 files...")
    pool = Pool()
    bar = progressbar.ProgressBar(max_value=num_samples, widgets=SIMPLE_BAR)
    for i, processed in enumerate(pool.imap_unordered(one_sample, samples), start=1):
        counter += processed[0]
        rows += processed[1]
        bar.update(i)
    bar.update(num_samples)
    pool.close()
    pool.join()

    imported_samples = get_imported_samples(counter)
    assert counter["all"] == num_samples

    assert len(rows) == imported_samples
    print_import_report(counter)

    output_csv = os.path.join(os.path.abspath(audio_dir), "validated.csv")
    print("Saving new file to: ", output_csv)
    with open(output_csv, "w", encoding="utf-8", newline="") as output_csv_file:
        writer = csv.writer(output_csv_file)
        bar = progressbar.ProgressBar(max_value=len(rows), widgets=SIMPLE_BAR)
        for filename in bar(rows):
            writer.writerow([filename, language_label])

    return rows, counter["imported_time"]

def _maybe_convert_wav(mp3_filename, wav_filename):
    if not os.path.exists(wav_filename):
        transformer = sox.Transformer()
        transformer.convert(samplerate=import_parameters['sample_rate'], n_channels=import_parameters['channels'])
        try:
            transformer.build(mp3_filename, wav_filename)
        except sox.core.SoxError:
            pass

def copy_audio_files_for_language(rows, imported_time, language, label, original_dataset_paths, target_root_path):

    print("Copying files for language ", language)

    shuffle(rows)

    #Get language name (not label) from original_dataset_paths
    train_rows = []
    validation_rows = []
    test_rows = []

    validation_seconds = 0
    train_seconds = 0
    test_seconds = 0

    #Make target dirs
    target_dir = os.path.join(target_root_path, language)
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    total_imported_seconds = imported_time / import_parameters['sample_rate']
    print("heeeeeeeeeeeeeeey:", total_imported_seconds, secs_to_hours(total_imported_seconds))

    #Process files
    for filename in rows:
        file = os.path.join(original_dataset_paths, language, 'clips', filename)
        new_file = copy2(file, target_dir)

        if (total_imported_seconds*0.6) > (train_seconds):
            train_rows.append([new_file, label])
            #Get duration of file and append to train seconds
            frames = int(subprocess.check_output(["soxi", "-s", new_file], stderr=subprocess.STDOUT))
            train_seconds += (frames/import_parameters['sample_rate'])
            continue
        if (total_imported_seconds*0.2) > (validation_seconds):
            validation_rows.append([new_file, label])
            #Get duration of file and append to train seconds
            frames = int(subprocess.check_output(["soxi", "-s", new_file], stderr=subprocess.STDOUT))
            validation_seconds += (frames/import_parameters['sample_rate'])
            continue
        if (total_imported_seconds*0.2) > (test_seconds):
            test_rows.append([new_file, label])
            #Get duration of file and append to train seconds
            frames = int(subprocess.check_output(["soxi", "-s", new_file], stderr=subprocess.STDOUT))
            test_seconds += (frames/import_parameters['sample_rate'])
            continue


    print("Samples for training: ", len(train_rows))
    print("Samples for validating: ", len(validation_rows))
    print("Samples for testing: ", len(test_rows))

    print("Final amount of tarining audio: %s (%s seconds)." % (secs_to_hours(train_seconds), train_seconds))
    print("Final amount of validating audio: %s (%s seconds)." % (secs_to_hours(validation_seconds), validation_seconds))
    print("Final amount of testing audio: %s (%s seconds)." % (secs_to_hours(test_seconds), test_seconds))

    return train_rows, validation_rows, test_rows

if __name__ == "__main__":
    languages = import_parameters['languages']
    original_dataset_paths = import_parameters['cv_path']
    target_root_path = import_parameters['target_root_path']

    for language, id in languages.items():
        tsv_dir = os.path.join(original_dataset_paths, language)
        audio_dir = os.path.join(tsv_dir, "clips")

        #Import Common Voice mp3 files
        #Use validated clips sets of Common Voice dataset
        imported_rows, imported_time = _maybe_convert_set(tsv_dir, audio_dir, id)
        #Copy converted wav files into target direction
        train_rows, validation_rows, test_rows = copy_audio_files_for_language(imported_rows, imported_time, language, id, original_dataset_paths, target_root_path)

        #Augment data (only training and validating sets)
        print("Augmenting training and validating files for language: ", language)
        num_files = int(len(train_rows)*import_parameters['augment_data_factor'])
        print("Number of files to augment: ", num_files)
        for i in range(num_files):
            if i % 100 == 0:
                print('Still processing ' + language + ' ' + str(i) + '/' + str(num_files))
            augmented_file_path = augment_audio_file_with_noise(train_rows[i][0])
            train_rows.append([augmented_file_path, id])

        num_files = int(len(validation_rows)*import_parameters['augment_data_factor'])
        print("Number of files to augment: ", num_files)
        for i in range(num_files):
            if i % 100 == 0:
                print('Still processing ' + language + ' ' + str(i) + '/' + str(num_files))
            augmented_file_path = augment_audio_file_with_noise(validation_rows[i][0])
            validation_rows.append([augmented_file_path, id])

        print("Generating spectrograms for language: ", language)

        image_train = []
        image_test = []
        image_validate = []
        #First for training set
        for i in range(len(train_rows)):
            if i % 100 == 0:
                print('Still processing ' + language + ' ' + "train set" + ' ' + str(i) + '/' + str(len(train_rows)))
            for j in audio_to_spectrogram(train_rows[i][0], import_parameters['sample_rate'], import_parameters['pixel_per_sec'], import_parameters['image_width'], import_parameters['image_height']):
                image_train.append(j)

        #Second for validating set
        for i in range(len(validation_rows)):
            if i % 100 == 0:
                print('Still processing ' + language + ' ' + "validating set" + ' ' + str(i) + '/' + str(len(validation_rows)))
            for j in audio_to_spectrogram(validation_rows[i][0], import_parameters['sample_rate'], import_parameters['pixel_per_sec'], import_parameters['image_width'], import_parameters['image_height']):
                image_validate.append(j)

        #Third for testing set
        for i in range(len(test_rows)):
            if i % 100 == 0:
                print('Still processing ' + language + ' ' + "test set" + ' ' + str(i) + '/' + str(len(test_rows)))
            for j in audio_to_spectrogram(test_rows[i][0], import_parameters['sample_rate'], import_parameters['pixel_per_sec'], import_parameters['image_width'], import_parameters['image_height']):
                image_test.append(j)

        train_csv = os.path.join(target_root_path, 'train.csv')
        print("Saving Training CSV files to: ", train_csv)

        with open(train_csv, "a", encoding="utf-8", newline="") as output_csv_file:
            writer = csv.writer(output_csv_file)
            bar = progressbar.ProgressBar(max_value=len(image_train), widgets=SIMPLE_BAR)
            for filename in bar(image_train):
                writer.writerow([filename, id])

        validate_csv = os.path.join(target_root_path, 'validate.csv')
        print("Saving Training CSV files to: ", validate_csv)

        with open(validate_csv, "a", encoding="utf-8", newline="") as output_csv_file:
            writer = csv.writer(output_csv_file)
            bar = progressbar.ProgressBar(max_value=len(image_validate), widgets=SIMPLE_BAR)
            for filename in bar(image_validate):
                writer.writerow([filename, id])

        test_csv = os.path.join(target_root_path, 'test.csv')
        print("Saving Testing CSV files to: ", test_csv)

        with open(test_csv, "a", encoding="utf-8", newline="") as output_csv_file:
            writer = csv.writer(output_csv_file)
            bar = progressbar.ProgressBar(max_value=len(image_test), widgets=SIMPLE_BAR)
            for filename in bar(image_test):
                writer.writerow([filename, id])

    #Open, read and random shuffle CSV files contents
    df1 = pd.read_csv(train_csv)
    x1 = df1.sample(frac=1)
    os.remove(train_csv)
    x1.to_csv(train_csv, index=False)

    df2 = pd.read_csv(validate_csv)
    x2 = df2.sample(frac=1)
    os.remove(validate_csv)
    x2.to_csv(validate_csv, index=False)

    df3 = pd.read_csv(test_csv)
    x3 = df3.sample(frac=1)
    os.remove(test_csv)
    x3.to_csv(test_csv, index=False)
