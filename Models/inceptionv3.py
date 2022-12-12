from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import GlobalAveragePooling2D

NAME = "InceptionV3"

def create_model(input_shape, num_classes):

    input_tensor = Input(shape=input_shape)  # this assumes K.image_dim_ordering() == 'tf'
    inception_model = InceptionV3(include_top=False, weights=None, input_tensor=input_tensor)
    #print(inception_model.summary())

    x = inception_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=inception_model.input, outputs=predictions)
