"""Generates a .wav file of a message in morse."""

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import wave

message = "Hello World"
time_unit = 0.05  # Dot = 0.1 seconds
audio_rate = 44100
tone_freq = 1000  # Frequency of tone when triggered

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

chunk = time_unit * audio_rate  # Samples in one time unit

tone_t_array = np.arange(0, time_unit, 1 / audio_rate)
tone = np.sin(2 * np.pi * tone_freq * tone_t_array)

t_array = np.arange(0, len(signal) * time_unit, 1 / audio_rate)
audio = []

for trigger in signal:
    audio.extend(tone * trigger)

# Save audio file as .wav
audio = np.asarray(audio)
audio *= 1000
audio = np.asarray(audio, dtype=np.int16)
wavfile.write('morse_out.wav', audio_rate, audio)

# Show waveform
plt.plot(t_array, audio)
plt.show()
