from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import L2

#from tensorflow.keras import backend
#backend.set_image_data_format('channels_first')

NAME = "topcoder_5s_finetune"

def create_model(input_shape, num_classes, is_training=True):

    weight_decay = 0.001

    model = Sequential()

    model.add(Conv2D(16, 5, kernel_regularizer=L2(weight_decay), activation="relu", input_shape=input_shape))
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
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(256, 3, kernel_regularizer=L2(weight_decay), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=L2(weight_decay), activation="relu"))

    model.add(Dense(num_classes, activation="softmax"))

    #ref_model = load_model("logs/2016-12-08-15-14-06/weights.20.model")
    #for ref_layer in ref_model.layers[:-2]:
    #    layer = model.get_layer(ref_layer.name)
    #    if layer:
    #        print ref_layer.name
    #        layer.set_weights(ref_layer.get_weights())
    #        layer.trainable = False

    return model
