from flask import Flask, render_template, request, send_from_directory
from LSTM_Pytorch import generate_music, save_generated_music_to_midi
import subprocess
import os

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

    # Appelle inference.py avec les arguments
    subprocess.run(['python', 'inference.py', artist, duration])

    return send_from_directory(OUTPUT_FOLDER, OUTPUT_FILE, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
