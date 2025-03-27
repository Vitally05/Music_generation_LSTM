import numpy as np
import os
import torch
import torch.nn as nn
from music21 import converter, instrument, note, chord
from torch.utils.data import Dataset, DataLoader
from data_preparation import extract_sub_folders, get_notes, prepare_sequences

SOURCE = ".\\raw_datasets\\"            # https://www.kaggle.com/datasets/soumikrakshit/classical-music-midi
PATH = ".\\datasets\\classical_music"

class MusicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        super(MusicLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, hidden_size)

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):

        x = self.embedding(x)  # [batch_size, sequence_length, hidden_size]

        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))

        out = out[:, -1, :]  # Get output from the last time step
        out = self.fc(out)  # Apply the fully connected layer
        return out  # Return predictions and hidden states for the next iteration

def train_music_model(model, train_loader, device, epochs):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"models\\epoch_{epoch + 1}.pt")
            print(f"model saved at epoch {epoch + 1}")

        print(f'Epoch {epoch+1}, Average Loss: {total_loss/len(train_loader):.4f}')

    torch.save(model.state_dict(), f"models\\epoch_{epoch + 1}.pt")
    print(f"model saved at epoch {epoch + 1}")


def generate_music(model, device, notes, note_to_int, int_to_note, sequence_length=100, generate_length=100):
    # Mettre le modèle en mode évaluation
    model.eval()

    # Sélectionner une séquence de départ aléatoire
    start_index = np.random.randint(0, len(notes) - sequence_length)
    pattern = notes[start_index:start_index + sequence_length]

    # Convertir la séquence de départ en indices
    generated_notes = list(pattern)

    # Convertir en tenseur
    with torch.no_grad():
        for _ in range(generate_length):
            # Convertir le pattern en tenseur
            pattern_indices = torch.tensor([note_to_int[note] for note in pattern],
                                           dtype=torch.long).unsqueeze(0).to(device)

            # Obtenir la prédiction
            prediction = model(pattern_indices)

            # Obtenir l'index de la note prédite
            predicted_index = torch.argmax(prediction, dim=1).item()

            # Convertir l'index en note
            predicted_note = int_to_note[predicted_index]

            # Ajouter la note générée
            generated_notes.append(predicted_note)

            # Mettre à jour le pattern (faire glisser la fenêtre)
            pattern = pattern[1:] + [predicted_note]

    return generated_notes

def save_generated_music_to_midi(generated_notes, output_file='generated_music.mid'):
    # Create a music21 stream
    stream = converter.stream.Stream()

    for note_str in generated_notes:
        try:
            if '.' in note_str:  # It's a chord
                chord_notes = [note.Note(int(p)) for p in note_str.split('.') if p.isdigit()]
                c = chord.Chord(chord_notes)

                # Set duration based on suffix
                if '_short' in note_str:
                    c.duration.quarterLength = 0.5
                elif '_medium' in note_str:
                    c.duration.quarterLength = 1.0
                else:
                    c.duration.quarterLength = 2.0

                stream.append(c)
            else:  # It's a single note
                pitch = note_str.split('_')[0]
                if pitch.isdigit():
                    n = note.Note(int(pitch))  # Convert pitch to MIDI note
                else:
                    n = note.Note(pitch)

                # Set duration based on suffix
                if '_short' in note_str:
                    n.duration.quarterLength = 0.5
                elif '_medium' in note_str:
                    n.duration.quarterLength = 1.0
                else:
                    n.duration.quarterLength = 2.0

                stream.append(n)
        except Exception as e:
            print(f"Error converting {note_str}: {e}")

    # Save as MIDI
    stream.write('midi', fp=output_file)

def get_output_name(folder_name):
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    os.listdir(folder_name)
    i = 1
    for file in os.listdir(folder_name):
        if file.endswith('.mid'):
            i += 1
    final_name = f"{folder_name}\\music{i}.mid"
    return final_name

if __name__ == '__main__':
    extract_sub_folders(SOURCE, PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    path = ".\\datasets\\classical_music"
    notes = get_notes(path)
    n_vocab = len(set(notes))

    sequence_length = 100
    network_input, network_output, note_to_int, int_to_note = prepare_sequences(notes, n_vocab, sequence_length)

    # Create dataset and dataloader
    dataset = torch.utils.data.TensorDataset(network_input, network_output)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Model parameters
    input_size = n_vocab
    hidden_size = 256
    output_size = n_vocab

    # Initialize model
    model = MusicLSTM(input_size, hidden_size, output_size)

    # Train model
    train_music_model(model, dataloader, epochs=50, device=device)

    generated_music = generate_music(model, device, notes, note_to_int, int_to_note)
    output_name = get_output_name("generated_music")
    save_generated_music_to_midi(generated_music, output_name)
