download_parameters = {
    'languages': ('hy', 'as'),
    'version': 'Common Voice Corpus 11.0', # Common Voice Delta Segment 11.0 | Common Voice Delta Segment 10.0
                                           # Common Voice Corpus 10.0 | Common Voice Corpus 9.0
                                           # Common Voice Corpus 8.0 | Common Voice Corpus 7.0
    'info_url': "https://commonvoice.mozilla.org/api/v1/datasets/languages/",
    'download_url': "https://commonvoice.mozilla.org/api/v1/bucket/dataset/"
}

import_parameters = {
    'languages': {'hy': 0, 'as': 1},
    'cv_path': '/home/varuzhan/Desktop/Language_Identification/Data',
    'target_root_path': '/home/varuzhan/Desktop/Language_Identification/Data/LID_DB',
    "image_height": 129,
    "image_width": 500,
    "augment_data_factor": 1.0,
    "sample_rate": 16000,
    'max_secs': 10,
    'min_secs': 1,
    'channels': 1,
    'pixel_per_sec': 50
}

training_parameters = {
    'dataset_root_path': '/home/varuzhan/Desktop/Language_Identification/Data/LID_DB',
    'labels': ["HY", "AS"],
    'batch_size': 1,
    'input_shape': (129, 500, 1),
    'learning_rate': 0.0001, #0.001
    'model': 'inceptionv3_crnn',
    'epochs': 100,
    'drop': 0.94,
    'epochs_drop': 2.0,
    "augment_data_factor": 1.0,
    "sample_rate": 16000,
    'pixel_per_sec': 50
}
