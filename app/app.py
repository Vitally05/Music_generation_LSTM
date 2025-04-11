from flask import Flask, render_template, request, send_from_directory, redirect, url_for
import subprocess
import os
from inference import generate_mp3_music
from datetime import datetime


app = Flask(__name__)

@app.route('/')
def index():
    just_generated = request.args.get('generated') == '1'
    cache_buster = int(datetime.now().timestamp())
    return render_template('index.html', just_generated=just_generated, cache_buster=cache_buster)



@app.route('/generate', methods=['POST'])
def generate():
    artist = request.form['artist']
    duration = request.form['duration']

    artist = str(artist)
    duration = int(duration)

    print(f"Artist: {artist}, Duration: {duration}")

    # Appelle la fonction de génération de musique
    generate_mp3_music(composer = artist, generate_length=duration)

    return redirect(url_for('index', generated=1))

if __name__ == '__main__':
    app.run(debug=True)
