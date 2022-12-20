import argparse
import numpy as np
from yaml import load
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from Utils.csv_loader import CSVLoader

def equal_error_rate(y_true, probabilities):

    y_one_hot = to_categorical(y_true)
    fpr, tpr, thresholds = roc_curve(y_one_hot.ravel(), probabilities.ravel())
    eer = brentq(lambda x : 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)

    return eer


def metrics_report(y_true, y_pred, probabilities, label_names=None):

    available_labels = range(0, len(label_names))

    print("Accuracy %s" % accuracy_score(y_true, y_pred))
    print("Equal Error Rate (avg) %s" % equal_error_rate(y_true, probabilities))
    print(classification_report(y_true, y_pred, labels=available_labels, target_names=label_names))
    #print(confusion_matrix(y_true, y_pred, labels=available_labels))


def evaluate(dataset_dir, model_file_name, batch_size, label_names, input_shape, num_classes):
    # Load Data + Labels
    data_generator = CSVLoader(dataset_dir, batch_size, num_classes, input_shape)

    # Model Generation
    model = load_model(model_file_name)
    #print(model.summary())

    probabilities = model.predict(
        data_generator.get_data(should_shuffle=False, is_prediction=True),
        steps=(data_generator.get_num_files()/batch_size),
        workers=1, # parallelization messes up data order. careful!
        max_queue_size=batch_size
    )

    y_pred = np.argmax(probabilities, axis=1)
    y_true = data_generator.get_labels()[:len(y_pred)]
    metrics_report(y_true, y_pred, probabilities, label_names=label_names)
