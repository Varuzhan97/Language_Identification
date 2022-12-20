download_parameters = {
    'languages': ('hy-AM', 'bn', 'ru'),
    'version': 'Common Voice Corpus 12.0', # Common Voice Delta Segment 12.0 | Common Voice Delta Segment 11.0 | Common Voice Delta Segment 10.0
                                           # Common Voice Corpus 10.0 | Common Voice Corpus 9.0
                                           # Common Voice Corpus 8.0 | Common Voice Corpus 7.0
    'info_url': "https://commonvoice.mozilla.org/api/v1/datasets/languages/",
    'download_url': "https://commonvoice.mozilla.org/api/v1/bucket/dataset/",
    'download_path': 'Data' #Folder in current direction
}

import_parameters = {
    'languages': {'hy': 0, 'bn': 1, 'ru': 2},
    'cv_path': '/home/varuzhan/Desktop/Language_Identification/Data', #Each language folder must be the name
                                                                      #of language specified in 'languages'
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
    'labels': ["HY", "BN", "RU"],
    'batch_size': 1,
    'input_shape': (129, 500, 1),
    'learning_rate': 0.001, #0.001
    'model': 'topcoder_5s_finetune', # inceptionv3_crnn | inceptionv3 | topcoder_5s_finetune | topcoder_crnn_finetune
    'epochs': 50,
    'drop': 0.94,
    'epochs_drop': 2.0,
    "augment_data_factor": 1.0,
    "sample_rate": 16000,
    'pixel_per_sec': 50,
    'decay': 'lr_time_based_decay'    # lr_time_based_decay | lr_step_decay | lr_exp_decay
}
