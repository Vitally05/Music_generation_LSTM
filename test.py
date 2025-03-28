import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data_preparation import extract_sub_folders, get_notes, prepare_sequences_for_vae
from music21 import converter, instrument, note, chord
from tqdm import tqdm
import pickle as pkl

def save_generated_music_to_midi(generated_notes, output_file='generated_music.mid'):
    # Crée un music21 stream
    stream = converter.stream.Stream()

    for note_str in generated_notes:
        try:
            if '.' in note_str:  # c'est un accord
                chord_notes = [note.Note(int(p)) for p in note_str.split('.') if p.isdigit()]
                c = chord.Chord(chord_notes)

               
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

class MusicVAE(nn.Module):
    """"
    Modèle VAE pour la génération de musique.
    """
    def __init__(self, input_size, hidden_size, latent_dim, num_layers=2, dropout=0.2):
        """"
        Constructeur du modèle VAE
        """
        super(MusicVAE, self).__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, hidden_size)

        # Encodeur
        self.encoder_lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)

        # Décodeur
        self.decoder_input = nn.Linear(latent_dim, hidden_size)
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc_output = nn.Linear(hidden_size, input_size)

    def encode(self, x):
        """"
        Encode la séquence d'entrée en un vecteur latent.
        """
        embedded = self.embedding(x)
        _, (h_n, _) = self.encoder_lstm(embedded)
        h_n = h_n[-1]  # Prendre la dernière couche
        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparamétrisation pour échantillonner à partir de la distribution latente.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len):
        """"
        "Décode le vecteur latent en une séquence de sortie.
        """
        # Utilise la couche linéaire pour transformer le vecteur latent en entrée du LSTM
        hidden = self.decoder_input(z).unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.decoder_lstm(hidden)
        return self.fc_output(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, x.size(1))
        return x_recon, mu, logvar

def train_music_vae(model, train_loader, device, epochs, int_to_note, note_to_int):
    """"
    "Entraîne le modèle VAE sur les données d'entrée.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    history = {'total': [], 'recon': [], 'kl': []}


    for epoch in range(epochs):
        model.train()
        total_loss = 0


        print(f"\nEpoch {epoch + 1}/{epochs}")
        loop = tqdm(train_loader, desc="Training", leave=False)

        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs, mu, logvar = model(inputs)

            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)

            recon_loss = criterion(outputs, targets)

            # KL Divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss /= inputs.size(0)

            kl_weight = min(1.0, epoch / 10)  # passe de 0 à 1 sur 10 epochs
            loss = recon_loss + kl_weight * kl_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            loop.set_postfix(
                total=loss.item(),
                recon=recon_loss.item(),
                kl=kl_loss.item(),
                kl_weight=kl_weight,)


        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")
        print(f"  (Breakdown: recon = {recon_loss.item():.4f} | KL = {kl_loss.item():.4f})")

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            # Sauvegarde du modèle
            torch.save(model.state_dict(), f"models\\vae_epoch_{epoch + 1}.pt")
            print(f"VAE model saved at epoch {epoch + 1}")

            # Génération d'un échantillon
            print("Generating sample...")
            sample = generate_music_from_vae_autoregressive(model, device, int_to_note, note_to_int, sequence_length=100)

            sample_name = get_output_name("generated_music")
            save_generated_music_to_midi(sample, sample_name)
            print(f"Sample saved to {sample_name}")

        history['total'].append(avg_loss)
        history['recon'].append(recon_loss.item())
        history['kl'].append(kl_loss.item())
    
    with open("training_history.pkl", "wb") as f:
        pkl.dump(history, f)
    return model

def generate_variation_from_sequence(model, device, input_sequence, int_to_note, sequence_length=100):
    """
    Génère une variation d'une séquence d'entrée.
    """
    model.eval()
    with torch.no_grad():
        input_sequence = input_sequence.unsqueeze(0).to(device)  # [1, seq_len]
        mu, logvar = model.encode(input_sequence)
        z = model.reparameterize(mu, logvar)

        # Démarre avec la première note de l'extrait
        current_input = input_sequence[:, 0].unsqueeze(1)  # [1, 1]

        generated_indices = [current_input.item()]
        hidden = None

        # Boucle de génération
        for _ in range(sequence_length - 1):
            # Prend la note courante et l'encode
            embedded = model.embedding(current_input)
            z_context = model.decoder_input(z).unsqueeze(1)
            decoder_input = embedded + z_context

            # Passer par le LSTM décodeur
            output, hidden = model.decoder_lstm(decoder_input, hidden)
            logits = model.fc_output(output.squeeze(1))
            probs = torch.softmax(logits, dim=1)
            next_idx = torch.multinomial(probs, num_samples=1).item()

            # Ajoute la note générée à la liste
            generated_indices.append(next_idx)
            current_input = torch.tensor([[next_idx]], dtype=torch.long).to(device)

        generated_notes = [int_to_note[idx] for idx in generated_indices]
        return generated_notes

def generate_music_from_vae_autoregressive(model, device, int_to_note, note_to_int, sequence_length=100):
    model.eval()
    with torch.no_grad():
        # Échantillonner un vecteur latent z ~ N(0, 1)
        z = torch.randn(1, model.latent_dim).to(device)

        # Initialiser une note de départ aléatoire
        start_note_idx = np.random.randint(0, len(int_to_note))
        current_input = torch.tensor([[start_note_idx]], dtype=torch.long).to(device)

        generated_indices = [start_note_idx]

        hidden = None
        for _ in range(sequence_length - 1):
            embedded = model.embedding(current_input)  # [1, 1, hidden]
            z_context = model.decoder_input(z).unsqueeze(1)  # [1, 1, hidden]
            decoder_input = embedded + z_context  # Combine contexte latent + note

            output, hidden = model.decoder_lstm(decoder_input, hidden)  # hidden is carried over
            logits = model.fc_output(output.squeeze(1))  # [1, vocab]
            probs = torch.softmax(logits, dim=1)
            next_idx = torch.multinomial(probs, num_samples=1).item()

            generated_indices.append(next_idx)
            current_input = torch.tensor([[next_idx]], dtype=torch.long).to(device)

        generated_notes = [int_to_note[idx] for idx in generated_indices]
        return generated_notes



if __name__ == '__main__':
    # Chemins
    SOURCE = ".\\raw_datasets\\"
    PATH = ".\\datasets\\classical_music"

    # Préparation des données
    extract_sub_folders(SOURCE, PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    notes = get_notes(PATH, cache_file="notes.pkl")
    n_vocab = len(set(notes))
    sequence_length = 100

    # Préparation pour VAE (séquence -> même séquence)
    network_input, network_output, note_to_int, int_to_note = prepare_sequences_for_vae(notes, sequence_length)
    print(f"network_input.shape = {network_input.shape} -> network_output.shape = {network_output.shape}")

    # Dataset & DataLoader
    dataset = torch.utils.data.TensorDataset(network_input, network_output)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Paramètres du modèle VAE
    input_size = n_vocab
    hidden_size = 128
    latent_dim = 32


    # Instanciation du modèle
    model = MusicVAE(input_size, hidden_size, latent_dim)

    # Entraînement
    model = train_music_vae(model, dataloader, device=device, epochs=500, int_to_note=int_to_note, note_to_int=note_to_int)
    print("Training completed!")

    # Sauvegarder le modèle complet
    torch.save(model.state_dict(), "models/vae_final.pt")
    print("VAE final model saved")

    # Génération de musique depuis l’espace latent
    generated_notes = generate_music_from_vae_autoregressive(model, device, int_to_note, note_to_int, sequence_length=100)

    output_name = get_output_name("generated_music")
    save_generated_music_to_midi(generated_notes, output_name)
    print(f"Generated music saved to {output_name}")

    # Générer une variation d’un vrai extrait musical
    print("Generating variation from real sequence...")

    example_sequence = network_input[0]  # Prends le premier extrait comme base
    variation_notes = generate_variation_from_sequence(model, device, example_sequence, int_to_note, sequence_length=100)

    output_name = get_output_name("generated_music")
    save_generated_music_to_midi(variation_notes, output_name)
    print(f"Variation saved to {output_name}")

