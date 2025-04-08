from flask import Flask, render_template, request, send_from_directory, redirect, url_for
import subprocess
import os
from inference import generate_mp3_music

app = Flask(__name__)

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
