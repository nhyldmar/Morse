"""Generates a .wav file of a message in morse."""

import argparse
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

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


def get_waveform(signal):
    """Generates the waveform for a given signal"""
    signal = np.asarray(signal, dtype=np.int16)

    chunk = time_unit * audio_rate  # Samples in one time unit

    tone_t_array = np.arange(0, time_unit, 1 / audio_rate)
    tone = np.sin(2 * np.pi * tone_freq * tone_t_array)

    audio = []

    for trigger in signal:
        audio.extend(tone * trigger)

    return audio


def write_wav(audio, filename):
    """Save audio file as .wav"""
    audio = np.asarray(audio)
    audio *= 1000
    audio = np.asarray(audio, dtype=np.int16)
    wavfile.write(filename, audio_rate, audio)


def show_waveform(waveform):
    """Plots the waveform"""
    t_array = np.arange(0, len(waveform), 1)
    plt.plot(t_array, waveform)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # Setup parser
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-m', '--message', nargs='+', type=str, default='E', help='Message to be processed')
    parser.add_argument('-n', '--filename', type=str, default='morse_out', help='Name of file to audio save to')
    parser.add_argument('-f', '--frequency', type=int, default=1000, help='Frequency of tone in Hz')
    parser.add_argument('-t', '--time_unit', type=float, default=0.05, help='Length of a dot in seconds')
    parser.add_argument('-s', '--show', nargs='?', const=True, default=False, help='Show the waveform')

    # Process arguments
    args = parser.parse_args()
    message = ' '.join(args.message)
    filename = args.filename + '.wav'
    time_unit = args.time_unit
    tone_freq = args.frequency
    show = args.show
    audio_rate = 44100

    # Generate audio
    signal = get_signal(message)
    waveform = get_waveform(signal)
    write_wav(waveform, filename)

    # Show waveform
    if show:
        show_waveform(waveform)
