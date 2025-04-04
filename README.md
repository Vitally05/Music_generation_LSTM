# Music_generation_LSTM

<hr>
Contraintes actuelles des données en entrée (sinon le modèle a moins de chances de faire une sortie cohérente):  

- Un seul instrument de musique par dataset (ne gère pas les orchestres etc.)
- Dataset avec un style homogène


Pour faire tourner Pytorch sur le GPU : pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


Pour faire tourner le modèle : Fichier LSTM_Pytorch.py

Indiquer le chemin d'accès vers les fichiers MIDI dans la variable PATH. Attention, il faut que les fichiers soient au format .mid et qu'il n'y ait pas de sous-dossiers.

Si vous avez des sous-dossiers, il faut renseigner le chemin dans SOURCE qui va ensuite les copier dans PATH

Il y aura ensuite un entrainement du modèle pour chaque compositeur dans COMPOSER. Si vous voulez un modèle qui peut générer de la musique de tous les compositeurs, mettre "all" dans COMPOSER.

