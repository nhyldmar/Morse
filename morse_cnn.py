"""This file attempts to build a convolutional neural network to decrypt morse letters."""

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils.np_utils import to_categorical

# Morse translator; 0 = dot, 1 = dash
morse = {'a': (0, 1),
         'b': (1, 0, 0, 0),
         'c': (1, 0, 1, 0),
         'd': (1, 0, 0),
         'e': (0,),
         'f': (0, 0, 1, 0),
         'g': (1, 1, 0),
         'h': (0, 0, 0, 0),
         'i': (0, 0),
         'j': (0, 1, 1, 1),
         'k': (1, 0, 1),
         'l': (0, 1, 0, 0),
         'm': (1, 1),
         'n': (1, 0),
         'o': (1, 1, 1),
         'p': (0, 1, 1, 0),
         'q': (1, 1, 0, 1),
         'r': (0, 1, 0),
         's': (0, 0, 0),
         't': (1,),
         'u': (0, 0, 1),
         'v': (0, 0, 0, 1),
         'w': (0, 1, 1),
         'x': (1, 0, 0, 1),
         'y': (1, 0, 1, 1),
         'z': (1, 1, 0, 0),
         '1': (0, 1, 1, 1, 1),
         '2': (0, 0, 1, 1, 1),
         '3': (0, 0, 0, 1, 1),
         '4': (0, 0, 0, 0, 1),
         '5': (0, 0, 0, 0, 0),
         '6': (1, 0, 0, 0, 0),
         '7': (1, 1, 0, 0, 0),
         '8': (1, 1, 1, 0, 0),
         '9': (1, 1, 1, 1, 0),
         '0': (1, 1, 1, 1, 1)}


def get_signal(message):
    """Generates the signal for a given word."""
    signal = []

    for letter in message:
        if letter == ' ':
            signal.extend((0, 0, 0, 0, 0))  # Seven units between words (two from below)
            continue

        try:
            letter_triggers = morse[letter.lower()]
        except KeyError:
            raise ValueError("{} is not an approved character".format(letter))

        for dash in letter_triggers:
            if dash:
                signal.extend((1, 1, 1))  # Dash is three units
            else:
                signal.append(1)  # Dot is one unit
            signal.append(0)  # One unit between triggers
        signal.extend((0, 0))  # Three units between letters (one from after trigger)

    return signal[:-3]


morse_keys = np.array([*morse.keys()])

output_size = 5
train_size = 6000
test_size = 1000

# Generating training and test data
output = np.random.randint(0, len(morse), (train_size + test_size, output_size))
classes = morse_keys[output]
vectors = [get_signal(message) for message in classes]

pad_length = len(get_signal('0' * output_size))
vectors = [np.pad(vector, (0, pad_length - len(vector)), 'constant') for vector in vectors]

train_vectors = [vectors[:train_size]]
test_vectors = [vectors[test_size:]]

output = to_categorical(output, len(morse))
output = output.reshape(train_size + test_size, len(morse) * output_size)
train_classes = output[:train_size]
test_classes = output[test_size:]


def build_model():
    """Defining the neural network"""
    model = Sequential()
    model.add(Dense(512, input_shape=(pad_length,)))  # First hidden layer
    model.add(Activation('relu'))  # Activation function
    model.add(Dropout(0.2))  # Protects from overfitting
    model.add(Dense(512))  # Second hidden layer
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(len(morse) * output_size))  # Output layer
    model.add(Activation('softmax'))
    return model


# Building the model
model = build_model()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_vectors, train_classes, batch_size=128, epochs=4, validation_data=(test_vectors, test_classes))

# Testing the accuracy
score = model.evaluate(test_vectors, test_classes, batch_size=32)
print('Test score:', score[0])
print('Test accuracy:', score[1])
