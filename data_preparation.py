from music21 import converter, note, chord
from pathlib import Path
import numpy as np
import os
import shutil

import torch
import torch.nn.functional as F
from tqdm import tqdm
import pickle

def extract_sub_folders(input_path, output_path):
    """
    Copies MIDI files from subfolders and saves them to a flat folder.
    """
    os.makedirs(output_path, exist_ok=True)
    for sub_folder in os.listdir(input_path):
        source_path = os.path.join(input_path, sub_folder)
        midi_files = os.listdir(source_path)
        for midi_file in midi_files:
            destination_path = os.path.join(output_path, midi_file)
            shutil.copyfile(os.path.join(source_path, midi_file), destination_path)

def get_notes(midi_folder_path, unique=False, cache_file="notes.pkl"):
    """
    Extract single notes from all MIDI files in a folder (ignores chords).
    If notes.pkl exists, loads from it. Else, parses MIDI files and saves notes.
    """
    # Si le cache existe déjà
    if os.path.exists(cache_file):
        print(f"Loading cached notes from {cache_file} 🎵")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    #sinon on le crée
    print("Parsing MIDI files")
    notes = []
    midi_folder_path = Path(midi_folder_path).resolve()

    for file in tqdm(midi_folder_path.glob("*.mid"), desc="Parsing MIDI"):
        try:
            midi = converter.parse(str(file))
            notes_to_parse = midi.flatten().notes 

            for element in notes_to_parse:
                duration = element.duration.quarterLength
                if duration < 0.75:
                    duration_class = 'short'
                elif duration < 1.5:
                    duration_class = 'medium'
                else:
                    duration_class = 'long'

                if isinstance(element, note.Note):
                    note_str = f"{element.pitch}_{duration_class}"
                    notes.append(note_str)
                elif isinstance(element, chord.Chord):
                    chord_str = f"{'.'.join(str(n) for n in element.normalOrder)}_{duration_class}"
                    notes.append(chord_str)

        except Exception as e:
            print(f"Error processing {file.name}: {e}")

    # Sauvegarde dans le cache
    with open(cache_file, "wb") as f:
        pickle.dump(notes, f)
        print(f"Notes cached in {cache_file}")

    return set(notes) if unique else notes



def prepare_sequences(notes, n_vocab, sequence_length=100):
    """
    Original version — used for LSTM that predicts the next note.
    Returns (X, y) where y is the next note.
    """
    pitchnames = sorted(set(notes))
    note_to_int = {note: number for number, note in enumerate(pitchnames)}
    int_to_note = {number: note for note, number in note_to_int.items()}

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    network_input = torch.tensor(network_input, dtype=torch.long)
    network_output = torch.tensor(network_output, dtype=torch.long)

    return network_input, network_output, note_to_int, int_to_note

def prepare_sequences_for_vae(notes, sequence_length=100):
    """
    Version modifiée pour VAE — reconstruit toute la séquence.
    Returns (X, y) where y = X (full sequence reconstruction).
    """
    pitchnames = sorted(set(notes))
    note_to_int = {note: number for number, note in enumerate(pitchnames)}
    int_to_note = {number: note for note, number in note_to_int.items()}

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence = notes[i:i + sequence_length]
        encoded = [note_to_int[note] for note in sequence]
        network_input.append(encoded)
        network_output.append(encoded)

    network_input = torch.tensor(network_input, dtype=torch.long)
    network_output = torch.tensor(network_output, dtype=torch.long)

    return network_input, network_output, note_to_int, int_to_note

if __name__ == '__main__':
    source = ".\\raw_datasets\\"
    path = ".\\datasets\\classical_music"
    extract_sub_folders(source, path)

    notes = get_notes(path)
    n_vocab = len(set(notes))

    sequence_length = 100

    # For LSTM:
    # network_input, network_output, note_to_int, int_to_note = prepare_sequences(notes, n_vocab, sequence_length)

    # For VAE:
    network_input, network_output, note_to_int, int_to_note = prepare_sequences_for_vae(notes, sequence_length)

    print(f"network_input.shape = {network_input.shape} -> network_output.shape = {network_output.shape}")
