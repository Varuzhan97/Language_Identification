from tensorflow.keras.layers import Dense, Permute, Reshape, Input
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.inception_v3 import InceptionV3

NAME = "Inceptionv3 CRNN"

def create_model(input_shape, num_classes):


    input_tensor = Input(shape=input_shape)  # this assumes K.image_dim_ordering() == 'tf'
    inception_model = InceptionV3(include_top=False, weights=None, input_tensor=input_tensor)
    # inception_model.load_weights("logs/2016-12-18-13-56-44/weights.21.model", by_name=True)

    for layer in inception_model.layers:
        layer.trainable = False

    x = inception_model.output
    #x = GlobalAveragePooling2D()(x)

    # (bs, y, x, c) --> (bs, x, y, c)
    x = Permute((2, 1, 3))(x)

    # (bs, x, y, c) --> (bs, x, y * c)
    _x, _y, _c = [int(s) for s in x._shape[1:]]
    x = Reshape((_x, _y*_c))(x)
    x = Bidirectional(LSTM(512, return_sequences=False), merge_mode="concat")(x)

    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inception_model.input, outputs=predictions)
    model.load_weights("logs/2017-01-02-13-39-41/weights.06.model")

    return model
