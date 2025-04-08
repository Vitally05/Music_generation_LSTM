import os

from midi2audio import FluidSynth
from pydub import AudioSegment


def convert_midi_to_mp3(midi_file_path,
                        soundfont_path,
                        absolute_fluidsynth_path,
                        absolute_ffmpeg_path,
                        mp3_file_path=None):
    """
    converts a MIDI file to MP3

    parameters:
    midi_file_path (str): path to the MIDI file to convert
    soundfont_path (str): path to a SoundFont file (.sf2) for synthesis.
    absolute_fluidsynth_path(str): path to the fluidsynth bin folder (because PATH may not work idk why)
    absolute_ffmpeg_path(str): path to the ffmpeg folder (because PATH may not work idk why too)
    mp3_file_path (str): output path for the MP3 file. if not specified,
                                   uses the same name as the MIDI file with .mp3 extension

    returns:
    str: path to the generated MP3 file

    note: !!!!
    this function requires the installation of:
    - FluidSynth: to be installed separately (https://github.com/FluidSynth/fluidsynth/releases) + add to PATH
                (may require SDL3.dll : https://www.dllme.com/dll/files/sdl3, put it in the bin directory of fluidsynth)
    - ffmpeg: to be installed separately (https://www.ffmpeg.org/) + add to PATH
    requires IDE restart after adding to PATH. same for cmd. paste access path for absolute_fluidsynth_path and absolute_ffmpeg_path
        to check if installed correctly, in command prompt type:
        fluidsynth --version
        ffmpeg -version
    - sf2 sound fount such as https://member.keymusician.com/Member/FluidR3_GM/index.html

    """

    AudioSegment.converter = os.path.join(absolute_ffmpeg_path,
                                          "ffmpeg.exe")  # force path of ffmpeg (the program fails to find it
    # otherwise, even if properly added to PATH)
    AudioSegment.ffprobe = os.path.join(absolute_ffmpeg_path, "ffprobe.exe")  # same for ffprobe

    os.environ["PATH"] += os.pathsep + absolute_fluidsynth_path  # path for fluidsynth bin folder
    # print("PATH:", os.environ["PATH"])

    if not os.path.isfile(midi_file_path):  # check if the MIDI file exists
        raise FileNotFoundError(f"the MIDI file '{midi_file_path}' does not exist")

    if mp3_file_path is None:  # MP3 output path if not specified
        base_name = os.path.splitext(midi_file_path)[0]
        mp3_file_path = f"{base_name}.mp3"

    wav_temp_path = f"{os.path.splitext(mp3_file_path)[0]}_temp.wav"  # temp Wav file

    try:

        fs = FluidSynth(
            sound_font=soundfont_path) if soundfont_path else FluidSynth()  # initialize FluidSynth with the
                                            # specified SoundFont or the default one (not working properly, specify one)

        fs.midi_to_audio(midi_file_path, wav_temp_path)  # convert MIDI to wav

        audio = AudioSegment.from_wav(wav_temp_path)  # convert wav to MP3
        audio.export(mp3_file_path, format="mp3")

        print(f"conversion successful: '{midi_file_path}' -> '{mp3_file_path}'")
        return mp3_file_path

    except Exception as e:
        print(f"error during conversion: {str(e)}")
        raise

    finally:
        if os.path.exists(wav_temp_path):
            os.remove(wav_temp_path)


if __name__ == "__main__":
    # ====================================================
    #                       DEMO
    # ====================================================

    midi_file = "generated_music/music7.mid"  # CHANGE

    soundfont_path = "app/static/sound_fonts/FluidR3_GM.sf2"  # piano
    absolute_fluidsynth_path = r"C:\Users\vigou\Documents\GitHub\Music_generation_LSTM\app\static\FluidSynth\fluidsynth-2.4.4-win10-x64\bin"  # CHANGE
    absolute_ffmpeg_path = r"C:\ffmpeg" 

    convert_midi_to_mp3(midi_file,
                        soundfont_path=soundfont_path,
                        absolute_fluidsynth_path=absolute_fluidsynth_path,
                        absolute_ffmpeg_path=absolute_ffmpeg_path,
                        mp3_file_path=None)
