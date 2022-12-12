import os
import math
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras import metrics
from Utils.csv_loader import CSVLoader
from Models import inceptionv3, inceptionv3_crnn
from datetime import datetime
from Utils.evaluate import evaluate
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from global_parameters import training_parameters

def step_decay(epoch, lr):
    lrate = lr * math.pow(training_parameters['drop'], math.floor((1+epoch)/training_parameters['epochs_drop']))
    return lrate

if __name__ == "__main__":
    main_dir = os.getcwd()
    print(training_parameters['labels'], len(training_parameters['labels']))
    #time.sleep(500)
    #image_width = main_config["Image Width"]
    #image_height = main_config["Image Height"]

    #Generate training,validating and testing CSV files directions
    train_data_dir = os.path.join(training_parameters['dataset_root_path'], "train.csv")
    validation_data_dir = os.path.join(training_parameters['dataset_root_path'], "validate.csv")
    test_data_dir = os.path.join(training_parameters['dataset_root_path'], "test.csv")

    #Load CSV data with generators
    #(self, data_dir, batch_size, num_classes, input_shape):

    train_data_generator = CSVLoader(train_data_dir, training_parameters['batch_size'], len(training_parameters['labels']), training_parameters['input_shape'])
    validation_data_generator = CSVLoader(validation_data_dir,  training_parameters['batch_size'], len(training_parameters['labels']), training_parameters['input_shape'])

    #Generate log direction
    log_dir = os.path.join(main_dir, "logs", datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

    checkpoint_filename = os.path.join(log_dir, "weights.{epoch:02d}.model")
    model_checkpoint_callback = ModelCheckpoint(checkpoint_filename, save_best_only=True, verbose=1, monitor="val_accuracy")

    tensorboard_callback = TensorBoard(log_dir=log_dir, write_images=True)
    csv_logger_callback = CSVLogger(os.path.join(log_dir, "log.csv"))
    early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode="min")
    #learning_rate_decay = LearningRateScheduler(step_decay, verbose=1)

    validation_steps = int((validation_data_generator.get_num_files()) / training_parameters['batch_size'])
    steps_per_epoch = int(train_data_generator.get_num_files() / training_parameters['batch_size'])

    print('Steps per Epoch: ' + str(steps_per_epoch))
    print('Validation steps: ' + str(validation_steps))

    model = None
    if training_parameters['model'] == "inceptionv3":
        model = inceptionv3.create_model(train_data_generator.get_input_shape(), len(training_parameters['labels']))
    if training_parameters['model'] == "inceptionv3_crnn":
        model = inceptionv3_crnn.create_model(train_data_generator.get_input_shape(), len(training_parameters['labels']))
    #print(model.summary())

    optimizer = Adam(learning_rate=training_parameters['learning_rate'])
    # optimizer = RMSprop(lr=config["learning_rate"], rho=0.9, epsilon=1e-08, decay=0.95)
    # optimizer = SGD(lr=config["learning_rate"], decay=1e-6, momentum=0.9, clipnorm=1, clipvalue=10)

    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=['accuracy', metrics.Recall(), metrics.Precision()])

    #if cli_args.weights:
    #    model.load_weights(cli_args.weights)

    # Training
    history = model.fit(
        train_data_generator.get_data(),
        steps_per_epoch=steps_per_epoch,
        epochs=training_parameters['epochs'],
        callbacks=[model_checkpoint_callback, tensorboard_callback, csv_logger_callback, early_stopping_callback], #, learning_rate_decay],
        verbose=1,
        validation_data=validation_data_generator.get_data(should_shuffle=False),
        validation_steps=validation_steps,
        workers=1,
        max_queue_size=training_parameters['batch_size']
    )

    #Do evaluation on model with best validation accuracy
    best_epoch = np.argmax(history.history["val_accuracy"])
    print("Log files: ", log_dir)
    print("Best epoch: ", best_epoch + 1, checkpoint_filename)
    model_file_name = checkpoint_filename.replace("{epoch:02d}", "{:02d}".format(best_epoch+1))

    evaluate(test_data_dir, model_file_name, training_parameters['batch_size'], training_parameters['labels'], training_parameters['input_shape'], len(training_parameters['labels']))

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    config_file.close()
