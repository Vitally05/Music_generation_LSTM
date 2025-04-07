from flask import Flask, render_template, request, send_from_directory, redirect, url_for
from LSTM_Pytorch import generate_music, save_generated_music_to_midi
import subprocess
import os
from LSTM_Pytorch import generate_mp3_music

app = Flask(__name__)

OUTPUT_FOLDER = "static/generated"
OUTPUT_FILE = "result.mp3"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    artist = request.form['artist']
    duration = request.form['duration']

    artist = str(artist)
    duration = int(duration)

    print(f"Artist: {artist}, Duration: {duration}")

    # Appelle la fonction de génération de musique
    generate_mp3_music(composer = artist, generate_length=duration)

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
