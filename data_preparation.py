from music21 import converter, note, chord
from pathlib import Path
import numpy as np
import os
import shutil

import torch
import torch.nn.functional as F

def extract_sub_folders(input_path, output_path):
    """
    Copies MIDI files from a sub folder and save them to a new folder. Useful for dataset https://www.kaggle.com/datasets/soumikrakshit/classical-music-midi
    :param input_path: path of the folder containing the MIDI files
    :param output_path: chosen path to save the extracted MIDI files
    """
    os.makedirs(output_path, exist_ok=True)
    for sub_folder in os.listdir(input_path):
        source_path = os.path.join(input_path, sub_folder)
        midi_files = os.listdir(source_path)
        for midi_file in midi_files:
            destination_path = os.path.join(output_path, midi_file)
            shutil.copyfile(os.path.join(source_path, midi_file), destination_path)
            #print(f"Fichier {midi_file} -> {destination_path}")


def get_notes(midi_folder_path, unique=False):
    """
    Extract notes and chords from all MIDI files in a given folder.
    :param midi_folder_path: path of the folder containing the MIDI files. !! all dataset must be from 1 instrument (no duo, etc., same instrument. example : piano.) !!
    :param  unique: to return a set of the notes (each note appearing at most once)
            or return all of them (each note appearing at least once, may and will be multiple but necessary to recognize the music patterns).
    :return: all the notes found in the MIDI files
    """
    notes = []
    midi_folder_path = Path(midi_folder_path).resolve()

    for file in midi_folder_path.glob("*.mid"):
        midi = converter.parse(str(file))  # music21 expects a str for the path
        notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            duration = element.duration.quarterLength
            if duration < 0.75:
                duration_class = 'short'
            elif duration < 1.5:
                duration_class = 'medium'
            else:
                duration_class = 'long'

            if isinstance(element, note.Note): # if single note
                note_str = f"{element.pitch}_{duration_class}"
                notes.append(note_str)
            elif isinstance(element, chord.Chord): # if chord
                chord_str = f"{'.'.join(str(n) for n in element.normalOrder)}_{duration_class}"
                notes.append(chord_str)
    if unique: # useless mdr
        return set(notes) #
    else:
        return notes


def prepare_sequences(notes, n_vocab, sequence_length=100):
    """
    Prepare the sequences used by the Neural Network
    PyTorch version

    :param notes: output from get_notes, ALL notes from the dataset, NOT a set(). Needed to learn the patterns in time
    :param n_vocab:size of the vocabulary (len(set(notes)) ; output onehot vector size
    :param sequence_length: May be interisting to change it to see what it changes in the output of the model
    :return:
    """

    # get all pitch names
    pitchnames = sorted(set(notes))

    # create a dictionary to map pitches to integers
    note_to_int = {note: number for number, note in enumerate(pitchnames)}
    int_to_note = {number: note for note, number in note_to_int.items()}

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    # Convert to PyTorch tensors with correct types
    network_input = torch.tensor(network_input, dtype=torch.long)
    network_output = torch.tensor(network_output, dtype=torch.long)

    return network_input, network_output, note_to_int, int_to_note


if __name__ == '__main__':
    source = ".\\raw_datasets\\"            # https://www.kaggle.com/datasets/soumikrakshit/classical-music-midi
    path = ".\\datasets\\classical_music"
    extract_sub_folders(source, path)

    #path = "datasets/midi_songs_FF_skuldur" # https://github.com/Skuldur/Classical-Piano-Composer/tree/master/midi_songs
    notes = get_notes(path)
    n_vocab = len(set(notes))
    network_input, network_output = prepare_sequences(notes, n_vocab) # X, y

    print(f"network_input.shape = {network_input.shape} -> network_output.shape = {network_output.shape}")
    #print(f"{network_input}")
    #print(f"{network_output}")


