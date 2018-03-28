from __future__ import print_function
from keras import backend as K
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
import os


def music_tagger_cnn(weights = 'msd', input_tensor=None,
                   include_top=True):

    # if K.image_dim_ordering() == 'th':
    #     input_shape = (1, 96, 1366)
    # else:
    #     input_shape = (96, 1366, 1)
    #
    # if input_tensor is None:
    #     melgram_input = Input(shape=input_shape)
    # else:
    #     if not K.is_keras_tensor(input_tensor):
    #         melgram_input = Input(tensor=input_tensor, shape=input_shape)
    #     else:
    #         melgram_input = input_tensor

    # if K.image_dim_ordering() == 'th':
    # channel_axis = 1
    # freq_axis = 2
    # time_axis = 3
    # else:
    channel_axis = 3
    freq_axis = 1
    time_axis = 2


    input_shape = (1, 96, 1366)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

    return model
