# Music_generation_LSTM

Accéder au dataset GIGA MIDI

S'enregistrer avec notre institution au lien suivant : https://huggingface.co/datasets/Metacreation/GigaMIDI

Dans ses paramètres, créer un "Access Tokens" avec permission "Read" (au minimum)

Copier cette clé dans le fichier token_access.py


<hr>
Contraintes actuelles des données en entrée (sinon le modèle a moins de chances de faire une sortie cohérente):  

- Un seul instrument de musique par dataset (ne gère pas les orchestres etc.)
- Dataset avec un style homogène


Pour faire tourner Pytorch sur le GPU : pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
