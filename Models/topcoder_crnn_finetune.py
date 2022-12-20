from tensorflow.keras.layers import Permute
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM
#from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import L2

NAME = "Topcoder_CRNN_Finetune"

def create_model(input_shape, num_classes):

    weight_decay = 0.001

    model = Sequential()

    model.add(Conv2D(16, 7, kernel_regularizer=L2(weight_decay), activation="relu", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(32, 5, kernel_regularizer=L2(weight_decay), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(64, 3, kernel_regularizer=L2(weight_decay), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, 3, kernel_regularizer=L2(weight_decay), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 1)))

    model.add(Conv2D(256, 3, kernel_regularizer=L2(weight_decay), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 1)))

    #model.load_weights("logs/2017-04-08-13-03-44/weights.08.model", by_name=True)
    # for ref_layer in ref_model.layers:
    #     layer = model.get_layer(ref_layer.name)
    #     if layer:
    #         layer.set_weights(ref_layer.get_weights())

    #for layer in model.layers:
    #    layer.trainable = False

    # (bs, y, x, c) --> (bs, x, y, c)
    model.add(Permute((2, 1, 3)))

    # (bs, x, y, c) --> (bs, x, y * c)
    bs, x, y, c = model.layers[-1].output_shape
    model.add(Reshape((x, y*c)))

    model.add(Bidirectional(LSTM(512, return_sequences=False), merge_mode="concat"))
    model.add(Dense(num_classes, activation="softmax"))

    return model
