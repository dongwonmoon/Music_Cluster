import os
import pretty_midi
import soundfile as sf


def convert_midi_to_wav(source_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for composer in os.listdir(source_dir):
        composer_path = os.path.join(source_dir, composer)
        composer_output_dir = os.path.join(output_dir, composer)
        os.makedirs(composer_output_dir, exist_ok=True)

        for midi_file in os.listdir(composer_path):
            if not midi_file.endswith(".mid"):
                continue
            midi_path = os.path.join(composer_path, midi_file)
            output_wav = os.path.join(
                composer_output_dir, midi_file.replace(".mid", ".wav")
            )

            midi_data = pretty_midi.PrettyMIDI(midi_path)
            audio_data = midi_data.fluidsynth()
            sf.write(output_wav, audio_data, 16000)

            print(f"Converted {midi_file} to WAV")


source_dir = "./data/archive"
output_dir = "./data/archive_wav"
convert_midi_to_wav(source_dir, output_dir)
