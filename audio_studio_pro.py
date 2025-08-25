import os
import sys
import math
import subprocess
import tempfile
from pathlib import Path
import time
import shutil
import random
import hashlib
import string
import json
from urllib.parse import quote
import ctypes
from ctypes.util import find_library
import platform
import requests
import zipfile
import io

os.environ["COQUI_TOS_AGREED"] = "1"

def run_command(command):
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
            text=True,
            encoding='utf-8'
        )
        for line in process.stdout:
            print(line, end='')
        process.wait()
        return process.returncode == 0
    except Exception as e:
        print(f"An exception occurred while running command: {command}\n{e}")
        return False

def download_and_unzip(url, extract_to):
    try:
        print(f"Downloading from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            print(f"Extracting to {extract_to}...")
            z.extractall(extract_to)
        print("Download and extraction successful.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
    except zipfile.BadZipFile:
        print("Error: Downloaded file is not a valid zip file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return False

def add_to_path_windows(folder_path):
    print(f"Adding {folder_path} to user PATH...")
    command = f'setx PATH "{folder_path};%PATH%"'
    result = run_command(command)
    if result == 0:
        print(f"Successfully added {folder_path} to PATH. Please restart your terminal for changes to take effect.")
    else:
        print(f"Failed to add {folder_path} to PATH.")

def install_dependencies():
    os_name = platform.system()
    
    if os_name == "Linux":
        print("Detected Linux. Installing system dependencies with apt-get...")
        dependencies_apt = ["rubberband-cli", "fluidsynth", "fluid-soundfont-gm"]
        run_command("apt-get update -y")
        run_command(f"apt-get install -y {' '.join(dependencies_apt)}")
    
    elif os_name == "Windows":
        print("Detected Windows. Automating dependency installation...")
        install_dir = os.path.join(os.path.expanduser("~"), "app_dependencies")
        os.makedirs(install_dir, exist_ok=True)
        print(f"Dependencies will be installed in: {install_dir}")

        rubberband_url = "https://breakfastquay.com/files/releases/rubberband-3.3.0-gpl-executable-windows.zip"
        fluidsynth_url = "https://github.com/FluidSynth/fluidsynth/releases/download/v2.3.5/fluidsynth-2.3.5-win64.zip"

        rubberband_extract_path = os.path.join(install_dir, "rubberband")
        if download_and_unzip(rubberband_url, rubberband_extract_path):
            rubberband_bin_path = os.path.join(rubberband_extract_path, os.listdir(rubberband_extract_path)[0])
            add_to_path_windows(rubberband_bin_path)

        fluidsynth_extract_path = os.path.join(install_dir, "fluidsynth")
        if download_and_unzip(fluidsynth_url, fluidsynth_extract_path):
            fluidsynth_bin_path = os.path.join(fluidsynth_extract_path, "bin")
            add_to_path_windows(fluidsynth_bin_path)

    else:
        print(f"Unsupported OS: {os_name}. Manual installation of system dependencies may be required.")

    print("\nInstalling Python packages with pip...")
    dependencies_1 = ["cython"]
    dependencies_2 = [
        "requests", "accelerate", "numpy", "httpx", "gradio", "compressed-tensors", "sentencepiece",
        "spaces", "matchering", "librosa", "pydub", "googledrivedownloader", "torch", 
        "torchvision", "torchaudio", "basic-pitch", "midi2audio", "imageio", "moviepy", 
        "pillow", "demucs", "matplotlib", "transformers", "scipy", "soundfile", "madmom",
        "chatterbox-tts"
    ]
    
    pip_executable = f'"{sys.executable}" -m pip'
    run_command(f"{pip_executable} install --upgrade pip")
    run_command(f"{pip_executable} install --force-reinstall {' '.join(dependencies_1)}")
    run_command(f"{pip_executable} install --force-reinstall {' '.join(dependencies_2)}")
    
    print("\nDependency installation process finished.")

install_dependencies()

import torch
import numpy as np
import spaces
import gradio as gr
import matplotlib.pyplot as plt
from scipy.io.wavfile import write as write_wav
import matchering as mg
import pydub
from googledrivedownloader import download_file_from_google_drive
from transformers import AutoProcessor, MusicgenForConditionalGeneration, pipeline, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from moviepy import ImageClip, AudioFileClip, CompositeVideoClip, VideoFileClip, TextClip, ColorClip, vfx
from moviepy.video.VideoClip import DataVideoClip
from PIL import Image, ImageFilter
from basic_pitch.inference import predict as predict_midi
from midi2audio import FluidSynth
import soundfile as sf
import librosa
from chatterbox.tts import ChatterboxTTS

import collections
import collections.abc
collections.MutableSequence = collections.abc.MutableSequence
np.float = np.float64
np.int = np.int32
import madmom

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CustomConversation:
    def __init__(self, text=""):
        self.past_user_inputs = []
        self.generated_responses = []
        if text:
            self.past_user_inputs.append(text)

    def add_user_input(self, text):
        self.past_user_inputs.append(text)

    def append_response(self, response):
        self.generated_responses.append(response)

def load_models():
    tts_processor, tts_model, tts_vocoder, speaker_model = None, None, None, None
    asr_pipeline, instrument_classifier, chatbot_pipeline = None, None, None
    musicgen_processor, musicgen_model = None, None

    try:
        print("Loading Chatterbox TTS model...")
        tts_model = ChatterboxTTS.from_pretrained(device=DEVICE)
        tts_processor = tts_model.processor
        tts_vocoder = None
        speaker_model = None
        print("Successfully loaded Chatterbox TTS model.")
    except Exception as e:
        print(f"Failed to load Chatterbox TTS model: {e}")

    try:
        asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo")
    except Exception as e:
        print(f"Failed to load ASR pipeline: {e}")

    try:
        instrument_classifier = pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593")
    except Exception as e:
        print(f"Failed to load instrument classifier: {e}")

    try:
        chatbot_pipeline = pipeline("image-text-to-text", model="Qwen/Qwen2.5-VL-7B-Instruct")
    except Exception as e:
        print(f"Failed to load chatbot pipeline: {e}")

    try:
        musicgen_processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        musicgen_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(DEVICE)
    except Exception as e:
        print(f"Failed to load MusicGen model: {e}")

    return (tts_processor, tts_model, tts_vocoder, speaker_model, asr_pipeline, 
            instrument_classifier, chatbot_pipeline, musicgen_processor, musicgen_model)

(tts_processor, tts_model, tts_vocoder, speaker_model, asr_pipeline, 
 instrument_classifier, chatbot_pipeline, processor, generation_model) = load_models()

ALL_LANGUAGES = {
    "English": "en", "Spanish": "es", "French": "fr", "German": "de", "Italian": "it",
    "Portuguese": "pt", "Polish": "pl", "Turkish": "tr", "Russian": "ru", "Dutch": "nl",
    "Czech": "cs", "Arabic": "ar", "Chinese": "zh", "Japanese": "ja", "Hungarian": "hu",
    "Korean": "ko", "Hebrew": "he", "Hindi": "hi", "Swedish": "sv", "Greek": "el",
    "Finnish": "fi", "Vietnamese": "vi", "Ukrainian": "uk", "Romanian": "ro"
}

def get_temp_file_path(suffix=".wav"):
    if not suffix.startswith("."): suffix = "." + suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
        return temp.name

def random_string(length=8):
    return ''.join(random.choice(string.ascii_lowercase) for i in range(length))

def delete_path(path):
    if not path or not os.path.exists(path): return
    try:
        if os.path.isfile(path): os.remove(path)
        elif os.path.isdir(path): shutil.rmtree(path)
    except OSError as e:
        print(f"Error deleting path {path}: {e}")

def export_audio(audio_segment, output_path_stem, format_choice):
    format_lower = format_choice.lower()
    if "mp3" in format_lower: file_format, bitrate, suffix = "mp3", "320k", ".mp3"
    elif "wav" in format_lower: file_format, bitrate, suffix = "wav", None, ".wav"
    elif "flac" in format_lower: file_format, bitrate, suffix = "flac", None, ".flac"
    else: raise ValueError(f"Unsupported format: {format_choice}")
    output_path = Path(output_path_stem).with_suffix(suffix)
    params = ["-acodec", "pcm_s16le"] if file_format == "wav" else None
    audio_segment.export(str(output_path), format=format_choice.lower(), bitrate=bitrate, parameters=params)
    return str(output_path)

def create_share_links(file_url, text_description):
    if not file_url: return ""
    encoded_text = quote(text_description)
    encoded_url = quote(file_url)
    twitter_link = f"https://twitter.com/intent/tweet?text={encoded_text}&url={encoded_url}"
    facebook_link = f"https://www.facebook.com/sharer/sharer.php?u={encoded_url}"
    reddit_link = f"https://www.reddit.com/submit?url={encoded_url}&title={encoded_text}"
    whatsapp_link = f"https://api.whatsapp.com/send?text={encoded_text}%20{encoded_url}"
    return f"""<div style='text-align:center; padding-top: 10px;'><p style='font-weight: bold;'>Share your creation!</p><a href='{twitter_link}' target='_blank' style='margin: 0 5px;'>X/Twitter</a> | <a href='{facebook_link}' target='_blank' style='margin: 0 5px;'>Facebook</a> | <a href='{reddit_link}' target='_blank' style='margin: 0 5px;'>Reddit</a> | <a href='{whatsapp_link}' target='_blank' style='margin: 0 5px;'>WhatsApp</a></div>"""

def save_text_to_file(text_content):
    if not text_content: return None
    temp_path = get_temp_file_path(".txt")
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(text_content)
    return temp_path

def _humanize_ai_output(audio_path):
    """Applies subtle effects to make AI-generated audio sound more natural."""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        noise = np.random.randn(len(y))
        y_noisy = y + 0.0001 * noise
        y_eq = y_noisy * (1 + 0.01 * np.sin(2 * np.pi * 1000 * np.arange(len(y)) / sr))
        sf.write(audio_path, y_eq, sr)
        return audio_path
    except Exception as e:
        print(f"Could not humanize AI output: {e}")
        return audio_path

@spaces.GPU(duration=180)
def _transcribe_audio_logic(audio_path, language):
    if asr_pipeline is None: raise gr.Error("Speech recognition pipeline is not available.")
    if not audio_path: raise gr.Error("Please upload an audio file to transcribe.")
    lang_code = ALL_LANGUAGES.get(language, None)
    return asr_pipeline(audio_path, generate_kwargs={"language": lang_code}, return_timestamps=True)["text"]

@spaces.GPU(duration=150)
def _generate_voice_logic(text, reference_audio, format_choice, humanize):
    if not tts_model:
        raise gr.Error("TTS model is not available.")
    if not text or not reference_audio: 
        raise gr.Error("Please provide text and a reference voice audio.")

    try:
        output_path_stem = get_temp_file_path(f"_generated_{random_string()}").replace(".wav", "")
        temp_wav_path = str(Path(output_path_stem).with_suffix(".wav"))
        
        tts_model.tts_to_file(
            text=text,
            speaker_wav=reference_audio,
            file_path=temp_wav_path,
            language="en"
        )

        if humanize:
            temp_wav_path = _humanize_ai_output(temp_wav_path)
            
        sound = pydub.AudioSegment.from_file(temp_wav_path)
        final_output_path = export_audio(sound, output_path_stem, format_choice)
        delete_path(temp_wav_path)
        return final_output_path
    except Exception as e:
        raise gr.Error(f"Generation failed: {e}")

@spaces.GPU(duration=480)
def _voice_conversion_logic(reference_audio, target_audio, language, format_choice):
    if not reference_audio or not target_audio:
        raise gr.Error("Please upload both a reference voice and a target song.")
    if not tts_model:
        raise gr.Error("TTS models are not available for voice conversion.")
    
    ref_dir, target_dir = tempfile.mkdtemp(), tempfile.mkdtemp()
    try:
        run_command(f'"{sys.executable}" -m demucs.separate -n htdemucs_ft --two-stems=vocals -o "{target_dir}" "{target_audio}"')
        target_vocals_path = Path(target_dir) / "htdemucs_ft" / Path(target_audio).stem / "vocals.wav"
        target_instrumental_path = Path(target_dir) / "htdemucs_ft" / Path(target_audio).stem / "no_vocals.wav"
        if not target_vocals_path.exists() or not target_instrumental_path.exists():
            raise gr.Error("Failed to separate vocals and instrumental from the target song.")
            
        transcribed_text = _transcribe_audio_logic(str(target_vocals_path), language)
        if not transcribed_text:
            raise gr.Error("Could not transcribe the lyrics from the target song.")
            
        new_vocals_path = _generate_voice_logic(transcribed_text, reference_audio, "wav", humanize=False)

        instrumental = pydub.AudioSegment.from_file(str(target_instrumental_path))
        new_vocals = pydub.AudioSegment.from_file(new_vocals_path)
        
        new_vocals = new_vocals.apply_gain(instrumental.dBFS - new_vocals.dBFS)
        combined_audio = instrumental.overlay(new_vocals)
        
        output_stem = str(Path(target_audio).with_name(f"{Path(target_audio).stem}_voice_converted"))
        final_output_path = export_audio(combined_audio, output_stem, format_choice)
        
        delete_path(new_vocals_path)
        return final_output_path
    finally:
        delete_path(ref_dir)
        delete_path(target_dir)

def _master_logic(source_path, strength, format_choice):
    if not source_path: raise gr.Error("Please upload a track to master.")
    output_stem = Path(source_path).with_name(f"{Path(source_path).stem}_mastered")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            reference_path = Path(temp_dir) / "reference.wav"
            download_file_from_google_drive(file_id="1UF_FIuq4vbCdDfCVLHvD_9fXzJDoredh", dest_path=str(reference_path), unzip=False)
            def _master(current_source_path):
                result_wav_path = get_temp_file_path(".wav")
                mg.process(target=str(current_source_path), reference=str(reference_path), results=[mg.pcm24(str(result_wav_path))], config=mg.Config(max_length=15*60, threshold=0.99/strength, internal_sample_rate=44100))
                if "tmp" in str(current_source_path): delete_path(current_source_path)
                return result_wav_path
            processed_path = source_path
            for _ in range(math.floor(strength)): processed_path = _master(processed_path)
            final_sound = pydub.AudioSegment.from_file(processed_path) + (strength - 1.0) * 6
            output_path = export_audio(final_sound, output_stem, format_choice)
            delete_path(processed_path)
            return output_path
    except Exception as e: raise gr.Error(f"Mastering failed: {e}")

@spaces.GPU(duration=120)
def _generate_music_logic(prompt, duration_s, format_choice, humanize):
    if generation_model is None or processor is None: raise gr.Error("MusicGen model is not available.")
    if not prompt: raise gr.Error("Please provide a prompt for music generation.")
    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(DEVICE)
    max_new_tokens = int(duration_s * 50)
    audio_values = generation_model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=max_new_tokens)
    sampling_rate = generation_model.config.audio_encoder.sampling_rate
    wav_output = audio_values[0, 0].cpu().numpy()
    temp_wav_path = get_temp_file_path(".wav")
    write_wav(temp_wav_path, rate=sampling_rate, data=wav_output)
    if humanize:
        temp_wav_path = _humanize_ai_output(temp_wav_path)
    sound = pydub.AudioSegment.from_file(temp_wav_path)
    output_stem = Path(temp_wav_path).with_name(f"generated_{random_string()}")
    output_path = export_audio(sound, output_stem, format_choice)
    delete_path(temp_wav_path)
    return output_path

def _auto_dj_mix_logic(files, mix_type, target_bpm, transition_sec, format_choice):
    if not files or len(files) < 2: raise gr.Error("Please upload at least two audio files.")
    transition_ms = int(transition_sec * 1000)
    processed_tracks = []
    if target_bpm is None or target_bpm == 0:
        proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
        act = madmom.features.beats.RNNBeatProcessor()(files[0].name)
        target_bpm = np.median(60 / np.diff(proc(act)))

    for file in files:
        try:
            temp_stretched_path = None
            current_path = file.name
            if "beatmatched" in mix_type.lower():
                proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
                act = madmom.features.beats.RNNBeatProcessor()(current_path)
                original_bpm = np.median(60 / np.diff(proc(act)))

                if original_bpm > 0 and target_bpm > 0:
                    speed_factor = target_bpm / original_bpm
                    temp_stretched_path = get_temp_file_path(Path(current_path).suffix)
                    stretch_audio_cli(current_path, temp_stretched_path, speed_factor)
                    current_path = temp_stretched_path

            track_segment = pydub.AudioSegment.from_file(current_path)
            processed_tracks.append(track_segment)

            if temp_stretched_path:
                delete_path(temp_stretched_path)

        except Exception as e:
            print(f"Could not process track {Path(file.name).name}, skipping. Error: {e}")
            continue

    if not processed_tracks: raise gr.Error("No tracks could be processed.")
    final_mix = processed_tracks[0]
    for i in range(1, len(processed_tracks)): final_mix = final_mix.append(processed_tracks[i], crossfade=transition_ms)
    output_stem = get_temp_file_path("_dj_mix").replace(".wav", "")
    final_output_path = export_audio(final_mix, output_stem, format_choice)
    return final_output_path

@spaces.GPU(duration=240)
def _create_beat_visualizer_logic(image_path, audio_path, image_effect, animation_style, scale_intensity):
    if not image_path or not audio_path: raise gr.Error("Provide both an image and an audio file.")
    img = Image.open(image_path)
    effect_map = {"Blur": ImageFilter.BLUR, "Sharpen": ImageFilter.SHARPEN, "Contour": ImageFilter.CONTOUR, "Emboss": ImageFilter.EMBOSS}
    if image_effect in effect_map: img = img.filter(effect_map[image_effect])
    temp_img_path = get_temp_file_path(".png"); img.save(temp_img_path)
    output_path = get_temp_file_path(".mp4")
    audio_clip = AudioFileClip(audio_path)
    duration = audio_clip.duration
    y, sr = librosa.load(audio_path, sr=None)
    rms = librosa.feature.rms(y=y)[0]
    scales = 1.0 + (((rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-6)) * (scale_intensity - 1.0))
    def beat_resize_func(t):
        frame_index = min(int(t * sr / 512), len(scales) - 1)
        return scales[frame_index]
    image_clip = ImageClip(temp_img_path).set_duration(duration)
    if animation_style == "Zoom In": image_clip = image_clip.resize(lambda t: 1 + 0.1 * (t / duration))
    elif animation_style == "Zoom Out": image_clip = image_clip.resize(lambda t: 1.1 - 0.1 * (t / duration))
    final_clip = image_clip.resize(lambda t: image_clip.w * beat_resize_func(t) / image_clip.w).set_position(('center', 'center')).set_audio(audio_clip)
    final_clip.write_videofile(output_path, codec='libx264', fps=24, audio_codec='aac', logger=None)
    delete_path(temp_img_path)
    return output_path

@spaces.GPU(duration=300)
def _create_lyric_video_logic(audio_path, background_path, lyrics_text, text_position, language):
    if not audio_path or not lyrics_text: raise gr.Error("Audio and lyrics text are required.")
    audio_clip = AudioFileClip(audio_path)
    duration = audio_clip.duration
    if background_path:
        bg_clip_class = ImageClip if background_path.lower().endswith(('.png', '.jpg', '.jpeg')) else VideoFileClip
        background_clip = bg_clip_class(background_path).set_duration(duration)
    else: background_clip = ColorClip(size=(1280, 720), color=(0,0,0), duration=duration)
    background_clip = background_clip.resize(width=1280)
    lines = [line for line in lyrics_text.strip().split('\n') if line.strip()]
    if not lines: raise gr.Error("Lyrics text is empty.")
    line_duration = duration / len(lines)
    font = 'Arial'
    if language in ["Hebrew", "Arabic", "Hindi", "Chinese", "Japanese", "Korean", "Russian", "Ukrainian", "Greek"]:
        font = 'Arial'
    lyric_clips = [TextClip(line, fontsize=70, color='white', font=font, stroke_color='black', stroke_width=2).set_position(text_position).set_start(i * line_duration).set_duration(line_duration) for i, line in enumerate(lines)]
    final_clip = CompositeVideoClip([background_clip] + lyric_clips, size=background_clip.size).set_audio(audio_clip)
    output_path = get_temp_file_path(".mp4")
    final_clip.write_videofile(output_path, codec='libx264', fps=24, audio_codec='aac', logger=None)
    return output_path

def stretch_audio_cli(input_path, output_path, speed_factor, crispness=6):
    if not os.path.exists(input_path): return False
    command = ["rubberband", "--tempo", str(speed_factor), "--crispness", str(crispness), "-q", input_path, output_path]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        return output_path
    except Exception as e:
        print(f"Error during audio stretching with rubberband: {e}")
        return False

def _analyze_audio_features_logic(audio_path):
    if not audio_path: raise gr.Error("Please upload an audio file.")
    try:
        proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
        act = madmom.features.beats.RNNBeatProcessor()(audio_path)
        bpm = np.median(60 / np.diff(proc(act)))

        y, sr = librosa.load(audio_path)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        key_map = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key = key_map[np.argmax(np.sum(chroma, axis=1))]
        return f"{key}, {bpm:.2f} BPM"
    except Exception as e: raise gr.Error(f"Analysis failed: {e}")

def _change_audio_speed_logic(audio_path, speed_factor, preserve_pitch, format_choice):
    if not audio_path: raise gr.Error("Please upload an audio file.")
    sound_out = None
    if preserve_pitch:
        audio_path_out = get_temp_file_path(Path(audio_path).suffix)
        if stretch_audio_cli(audio_path, audio_path_out, speed_factor):
            sound_out = pydub.AudioSegment.from_file(audio_path_out)
            delete_path(audio_path_out)
        else:
            raise gr.Error("Failed to stretch audio while preserving pitch.")
    else:
        sound = pydub.AudioSegment.from_file(audio_path)
        new_frame_rate = int(sound.frame_rate * speed_factor)
        sound_out = sound._spawn(sound.raw_data, overrides={"frame_rate": new_frame_rate}).set_frame_rate(sound.frame_rate)

    if sound_out:
        output_stem = str(Path(audio_path).with_name(f"{Path(audio_path).stem}_speed_{speed_factor}x"))
        return export_audio(sound_out, output_stem, format_choice)
    else:
        raise gr.Error("Could not process audio speed change.")

@spaces.GPU(duration=150)
def _separate_stems_logic(audio_path, separation_type, format_choice):
    if not audio_path: raise gr.Error("Please upload an audio file.")
    output_dir = tempfile.mkdtemp()
    run_command(f'"{sys.executable}" -m demucs.separate -n htdemucs_ft --two-stems=vocals -o "{output_dir}" "{audio_path}"')
    separated_dir = Path(output_dir) / "htdemucs_ft" / Path(audio_path).stem
    vocals_path = separated_dir / "vocals.wav"
    accompaniment_path = separated_dir / "no_vocals.wav"
    if not vocals_path.exists() or not accompaniment_path.exists(): delete_path(output_dir); raise gr.Error("Stem separation failed.")
    chosen_stem_path, suffix = (vocals_path, "_acapella") if "acapella" in separation_type.lower() else (accompaniment_path, "_karaoke")
    sound = pydub.AudioSegment.from_file(chosen_stem_path)
    output_stem = str(Path(audio_path).with_name(Path(audio_path).stem + suffix))
    final_output_path = export_audio(sound, output_stem, format_choice)
    delete_path(output_dir)
    return final_output_path

@spaces.GPU(duration=180)
def _pitch_shift_vocals_logic(audio_path, pitch_shift, format_choice):
    if not audio_path: raise gr.Error("Please upload a song.")
    separation_dir = tempfile.mkdtemp()
    run_command(f'"{sys.executable}" -m demucs.separate -n htdemucs_ft --two-stems=vocals -o "{separation_dir}" "{audio_path}"')
    separated_dir = Path(separation_dir) / "htdemucs_ft" / Path(audio_path).stem
    vocals_path = separated_dir / "vocals.wav"
    instrumental_path = separated_dir / "no_vocals.wav"
    if not vocals_path.exists() or not instrumental_path.exists(): delete_path(separation_dir); raise gr.Error("Vocal separation failed.")
    y_vocals, sr = librosa.load(str(vocals_path), sr=None)
    y_shifted = librosa.effects.pitch_shift(y=y_vocals, sr=sr, n_steps=float(pitch_shift))
    shifted_vocals_path = get_temp_file_path("_shifted_vocals.wav")
    write_wav(shifted_vocals_path, sr, (y_shifted * 32767).astype(np.int16))
    instrumental = pydub.AudioSegment.from_file(instrumental_path)
    shifted_vocals = pydub.AudioSegment.from_file(shifted_vocals_path)
    combined = instrumental.overlay(shifted_vocals)
    output_stem = str(Path(audio_path).with_name(f"{Path(audio_path).stem}_vocal_pitch_shifted"))
    final_output_path = export_audio(combined, output_stem, format_choice)
    delete_path(separation_dir); delete_path(shifted_vocals_path)
    return final_output_path

def _create_spectrum_visualization_logic(audio_path):
    if not audio_path: raise gr.Error("Please upload an audio file.")
    try:
        y, sr = librosa.load(audio_path)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        fig, ax = plt.subplots(figsize=(10, 4), facecolor='#1f2937')
        ax.set_facecolor('#1f2937')
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax, cmap='viridis')
        ax.set_title('Spectrogram', color='white'); ax.set_xlabel('Time (s)', color='white'); ax.set_ylabel('Frequency (Hz)', color='white')
        ax.tick_params(colors='white')
        fig.tight_layout()
        temp_path = get_temp_file_path(".png")
        fig.savefig(temp_path, facecolor=fig.get_facecolor())
        plt.close(fig)
        return temp_path
    except Exception as e: raise gr.Error(f"Error creating spectrum: {e}")

@spaces.GPU(duration=240)
def _stem_mixer_logic(files, format_choice):
    if not files or len(files) < 2:
        raise gr.Error("Please upload at least two stem files.")
    processed_stems = []
    target_sr = None
    target_bpm = None

    for i, file in enumerate(files):
        y, sr = librosa.load(file.name, sr=None)
        if target_sr is None:
            target_sr = sr
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

        proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
        act = madmom.features.beats.RNNBeatProcessor()(file.name)
        tempo = np.median(60 / np.diff(proc(act)))

        if i == 0:
            target_bpm = tempo

        if tempo != target_bpm:
            speed_factor = target_bpm / tempo
            temp_stretched_path = get_temp_file_path(".wav")
            temp_original_path = get_temp_file_path(".wav")
            sf.write(temp_original_path, y, target_sr)
            stretch_audio_cli(temp_original_path, temp_stretched_path, speed_factor)
            y, _ = librosa.load(temp_stretched_path, sr=target_sr)
            delete_path(temp_original_path)
            delete_path(temp_stretched_path)

        processed_stems.append(y)

    max_length = max(len(y) for y in processed_stems)
    mixed_y = np.zeros(max_length)
    for y in processed_stems:
        mixed_y[:len(y)] += y
    mixed_y /= len(processed_stems)
    temp_wav_path = get_temp_file_path(".wav")
    write_wav(temp_wav_path, target_sr, (mixed_y * 32767).astype(np.int16))
    sound = pydub.AudioSegment.from_file(temp_wav_path)
    output_stem = Path(temp_wav_path).with_name(f"stem_mix_{random_string()}")
    output_path = export_audio(sound, output_stem, format_choice)
    delete_path(temp_wav_path)
    return output_path

@spaces.GPU(duration=60)
def _get_feedback_logic(audio_path):
    if not audio_path:
        raise gr.Error("Please upload an audio file for feedback.")
    try:
        y, sr = librosa.load(audio_path)
        rms = librosa.feature.rms(y=y)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
        act = madmom.features.beats.RNNBeatProcessor()(audio_path)
        tempo = np.median(60 / np.diff(proc(act)))

        feedback = "### AI Track Feedback\n\n"
        feedback += "#### Pros\n"
        if np.mean(rms) > 0.1:
            feedback += "- **Good Dynamics:** The track has a solid overall volume and dynamic range.\n"
        else:
            feedback += "- **Subtle Dynamics:** The track maintains a consistent, though quiet, dynamic level.\n"
        if np.mean(spectral_contrast) > 20:
             feedback += "- **Clear Frequencies:** The frequency spectrum is well-defined, suggesting good separation between instruments.\n"
        feedback += "\n#### Cons\n"
        if np.mean(rms) < 0.05:
            feedback += "- **Low Volume:** The overall volume of the track is quite low.\n"
        if np.std(rms) < 0.02:
            feedback += "- **Lack of Dynamic Variation:** The track feels a bit flat dynamically. There isn't much variation between loud and soft parts.\n"
        feedback += "\n#### Advice\n"
        if np.mean(rms) < 0.05:
            feedback += "- **Mastering:** Consider applying some compression and a limiter to increase the overall loudness and presence of the track.\n"
        if np.mean(spectral_contrast) < 18:
            feedback += "- **EQ Adjustments:** Some elements might be clashing in the frequency spectrum. Try using an equalizer (EQ) to carve out space for each instrument, especially in the low-mid range.\n"
        return feedback
    except Exception as e:
        raise gr.Error(f"Analysis failed: {e}")

@spaces.GPU(duration=360)
def _generate_video_logic(audio_path, prompt, format_choice):
    if not audio_path:
        raise gr.Error("Please upload an audio file.")
    y, sr = librosa.load(audio_path)
    duration = librosa.get_duration(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    act = madmom.features.beats.RNNBeatProcessor()(audio_path)
    beat_times = proc(act)
    beats = librosa.time_to_frames(beat_times, sr=sr)

    rms_norm = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-6)
    centroid_norm = (spectral_centroid - np.min(spectral_centroid)) / (np.max(spectral_centroid) - np.min(spectral_centroid) + 1e-6)
    w, h = 1280, 720
    fps = 30
    def make_frame(t):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame_idx = int(t * sr / 512)
        color_val = centroid_norm[min(frame_idx, len(centroid_norm)-1)]
        r = int(10 + color_val * 60)
        g = int(20 + color_val * 40)
        b = int(40 + color_val * 90)
        frame[:,:,:] = [r, g, b]
        radius = int(50 + rms_norm[min(frame_idx, len(rms_norm)-1)] * 200)
        center_x, center_y = w // 2, h // 2
        for beat_frame in beats:
            if abs(frame_idx - beat_frame) < 2:
                radius = int(radius * 1.5)
                break
        rr, cc = np.ogrid[:h, :w]
        circle_mask = (rr - center_y)**2 + (cc - center_x)**2 <= radius**2
        frame[circle_mask] = [int(200 + color_val * 55), int(150 - color_val * 50), int(100 + color_val * 50)]
        return frame
    output_path = get_temp_file_path(".mp4")
    animation = DataVideoClip(np.array([make_frame(t) for t in np.arange(0, duration, 1/fps)]), fps=fps)
    audio_clip = AudioFileClip(audio_path)
    final_clip = animation.set_audio(audio_clip)
    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
    return output_path

@spaces.GPU(duration=90)
def _identify_instruments_logic(audio_path):
    if not audio_path:
        raise gr.Error("Please upload an audio file to identify instruments.")
    if instrument_classifier is None:
        raise gr.Error("Instrument identification model is not available.")

    predictions = instrument_classifier(audio_path, top_k=10)

    instrument_list = [
        "guitar", "piano", "violin", "drum", "bass", "saxophone", "trumpet", "flute",
        "cello", "clarinet", "synthesizer", "organ", "accordion", "banjo", "harp", "voice", "speech"
    ]

    detected_instruments = "### Detected Instruments\n\n"
    found = False
    for p in predictions:
        label = p['label'].lower()
        if any(instrument in label for instrument in instrument_list):
            detected_instruments += f"- **{p['label'].title()}** (Score: {p['score']:.2f})\n"
            found = True

    if not found:
        detected_instruments += "Could not identify specific instruments with high confidence. Top sound events:\n"
        for p in predictions[:3]:
            detected_instruments += f"- {p['label'].title()} (Score: {p['score']:.2f})\n"

    return detected_instruments

@spaces.GPU(duration=300)
def _extend_audio_logic(audio_path, extend_duration_s, format_choice, humanize):
    if not audio_path:
        raise gr.Error("Please upload an audio file to extend.")
    if generation_model is None or processor is None:
        raise gr.Error("MusicGen model is not available for audio extension.")
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    prompt_duration_s = min(15.0, len(y) / sr)
    prompt_wav = y[-int(prompt_duration_s * sr):]
    inputs = processor(
        audio=prompt_wav,
        sampling_rate=sr,
        return_tensors="pt"
    ).to(DEVICE)
    total_duration_s = prompt_duration_s + extend_duration_s
    max_new_tokens = int(total_duration_s * 50)
    generated_audio_values = generation_model.generate(
        **inputs,
        do_sample=True,
        guidance_scale=3,
        max_new_tokens=max_new_tokens
    )
    generated_wav = generated_audio_values[0, 0].cpu().numpy()
    extension_start_sample = int(prompt_duration_s * generation_model.config.audio_encoder.sampling_rate)
    extension_wav = generated_wav[extension_start_sample:]

    temp_extension_path = get_temp_file_path(".wav")
    sf.write(temp_extension_path, extension_wav, generation_model.config.audio_encoder.sampling_rate)
    if humanize:
        temp_extension_path = _humanize_ai_output(temp_extension_path)

    original_sound = pydub.AudioSegment.from_file(audio_path)
    extension_sound = pydub.AudioSegment.from_file(temp_extension_path)

    if original_sound.channels != extension_sound.channels:
        extension_sound = extension_sound.set_channels(original_sound.channels)

    final_sound = original_sound + extension_sound
    output_stem = str(Path(audio_path).with_name(f"{Path(audio_path).stem}_extended"))
    final_output_path = export_audio(final_sound, output_stem, format_choice)
    delete_path(temp_extension_path)
    return final_output_path

def _chatbot_response_logic(message, history):
    if chatbot_pipeline is None:
        history.append((message, "My AI brain is offline right now, sorry! Please try again later."))
        return "", history

    system_prompt = """You are Fazzer, the official AI assistant for the 'Audio Studio Pro' application. Your personality is friendly, helpful, and enthusiastic about audio production. Your primary goal is to assist users by answering their questions about the application's features and guiding them on how to use the tools.

**Key Information about the project:**
- The application is called **Audio Studio Pro**.
- It was created by a developer named **Yaron** in **Israel**.
- Your name is **Fazzer**.

**Your Core Responsibilities:**
1. Explain the purpose of each tool in the application.
2. Provide simple instructions on how to use the features.
3. Maintain a concise, clear, and encouraging tone.
4. If you don't know the answer, politely say so. Do not make up features.

**Here is a complete list of the application's features you must be knowledgeable about:**
* **Mastering:** Automatically enhances a track's loudness and clarity to a professional level.
* **Vocal Auto-Tune:** Corrects the pitch of vocals in a song to make them sound more in-tune.
* **MIDI Tools:** A suite for converting audio to MIDI files, MIDI files back to audio, and using AI to enhance a simple MIDI melody into a richer piece of music.
* **Audio Extender:** Uses AI to seamlessly continue a piece of music, making it longer.
* **Stem Mixer:** Allows users to upload individual instrument tracks (stems) like drums, bass, and vocals, and mixes them together into a final song.
* **Track Feedback:** An AI tool that analyzes a user's track and provides constructive feedback on its technical aspects, like dynamics and frequency balance.
* **Instrument ID:** Identifies the different musical instruments present in an audio file.
* **AI Video Gen:** Creates a simple, abstract music visualizer video based on the audio's characteristics.
* **Speed & Pitch:** Changes the speed of an audio track, with an option to preserve the original pitch.
* **Stem Separation:** Splits a full song into two parts: 'Acapella' (vocals only) or 'Karaoke' (instrumental only).
* **Vocal Pitch Shifter:** Changes the pitch of only the vocals in a song, while leaving the instrumental untouched.
* **Voice Conversion:** Takes the voice from one person and applies it to the lyrics and melody of another song, essentially making it sound like the first person is singing the second song.
* **DJ AutoMix:** Automatically mixes multiple songs together with smooth, beatmatched transitions, like a DJ.
* **AI Music Gen:** Creates original music from a text description (e.g., 'upbeat synthwave').
* **AI Voice Gen:** Clones a person's voice from a short audio sample and uses it to say any text the user types (Text-to-Speech).
* **Analysis (BPM & Key):** Detects the musical key and Beats Per Minute (BPM) of a track.
* **Speech-to-Text:** Transcribes spoken words from an audio file into written text.
* **Spectrum Analyzer:** Creates a visual graph (a spectrogram) of the frequencies in an audio file over time.
* **Beat Visualizer:** Generates a video where a user-provided image pulses and animates in time with the beat of an audio track.
* **Lyric Video Creator:** Helps create simple lyric videos by overlaying text onto a background image or video, synchronized with the audio.
* **Support Chat:** That's you! The chatbot for helping users.

Always be ready to answer questions like 'What is Stem Mixing?' or 'How do I use the Vocal Pitch Shifter?' based on the descriptions above."""

    conversation = CustomConversation(system_prompt)
    for user_turn, bot_turn in history:
        conversation.add_user_input(user_turn)
        conversation.append_response(bot_turn)

    conversation.add_user_input(message)
    result = chatbot_pipeline(conversation)
    response = result.generated_responses[-1]
    history.append((message, response))
    return "", history

@spaces.GPU(duration=180)
def _audio_to_midi_logic(audio_path):
    if not audio_path:
        raise gr.Error("Please upload an audio file for MIDI conversion.")
    output_dir = tempfile.mkdtemp()
    predict_midi(audio_path, output_dir)
    midi_files = list(Path(output_dir).glob("*.mid"))
    if not midi_files:
        raise gr.Error("Failed to convert audio to MIDI.")
    return str(midi_files[0])

@spaces.GPU(duration=90)
def _midi_to_audio_logic(midi_path, format_choice):
    if not midi_path:
        raise gr.Error("Please upload a MIDI file for audio conversion.")
    fs = FluidSynth(sound_font="/usr/share/sounds/sf2/FluidR3_GM.sf2")
    temp_wav_path = get_temp_file_path(".wav")
    fs.midi_to_audio(midi_path, temp_wav_path)
    sound = pydub.AudioSegment.from_file(temp_wav_path)
    output_stem = str(Path(midi_path).with_name(f"{Path(midi_path).stem}_render"))
    final_output_path = export_audio(sound, output_stem, format_choice)
    delete_path(temp_wav_path)
    return final_output_path

@spaces.GPU(duration=240)
def _enhance_midi_logic(midi_path, format_choice, humanize):
    if not midi_path:
        raise gr.Error("Please upload a MIDI file to enhance.")
    temp_audio_prompt = _midi_to_audio_logic(midi_path, "wav")
    enhanced_audio = _extend_audio_logic(temp_audio_prompt, 10, format_choice, humanize)
    delete_path(temp_audio_prompt)
    return enhanced_audio

@spaces.GPU(duration=480)
def _autotune_vocals_logic(audio_path, strength, format_choice):
    if not audio_path:
        raise gr.Error("Please upload a song for vocal tuning.")

    separation_dir = tempfile.mkdtemp()
    run_command(f'"{sys.executable}" -m demucs.separate -n htdemucs_ft --two-stems=vocals -o "{separation_dir}" "{audio_path}"')
    separated_dir = Path(separation_dir) / "htdemucs_ft" / Path(audio_path).stem
    vocals_path = separated_dir / "vocals.wav"
    instrumental_path = separated_dir / "no_vocals.wav"

    if not vocals_path.exists() or not instrumental_path.exists():
        delete_path(separation_dir)
        raise gr.Error("Vocal separation failed.")

    y, sr = librosa.load(str(vocals_path), sr=None, mono=True)
    y_tuned = y.copy()

    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

    for i in range(len(f0)):
        if voiced_flag[i]:
            current_f0 = f0[i]
            target_midi = librosa.hz_to_midi(current_f0)
            rounded_midi = round(target_midi)
            target_f0 = librosa.midi_to_hz(rounded_midi)

            correction_factor = (target_f0 / current_f0 - 1) * strength
            corrected_f0 = current_f0 * (1 + correction_factor)
            f0[i] = corrected_f0

    y_tuned = librosa.effects.pitch_shift(y, sr=sr, n_steps=0, bins_per_octave=12, res_type='soxr_hq')

    temp_tuned_vocals_path = get_temp_file_path("_tuned_vocals.wav")
    sf.write(temp_tuned_vocals_path, y_tuned, sr)

    instrumental = pydub.AudioSegment.from_file(instrumental_path)
    tuned_vocals = pydub.AudioSegment.from_file(temp_tuned_vocals_path)

    if instrumental.channels == 2:
        tuned_vocals = tuned_vocals.set_channels(2)

    combined = instrumental.overlay(tuned_vocals)
    output_stem = str(Path(audio_path).with_name(f"{Path(audio_path).stem}_autotuned"))
    final_output_path = export_audio(combined, output_stem, format_choice)

    delete_path(separation_dir)
    delete_path(temp_tuned_vocals_path)

    return final_output_path

def main():
    theme = gr.themes.Base(primary_hue=gr.themes.colors.slate, secondary_hue=gr.themes.colors.indigo, font=(gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif")).set(
        body_background_fill_dark="#111827", block_background_fill_dark="#1f2937", block_border_width="1px",
        block_title_background_fill_dark="#374151", button_primary_background_fill_dark="linear-gradient(90deg, #4f46e5, #7c3aed)",
        button_primary_text_color_dark="#ffffff", button_secondary_background_fill_dark="#374151",
        button_secondary_text_color_dark="#ffffff", slider_color_dark="#6366f1"
    )
    css = """
        footer {display: none !important;}
        .gradio-container, main { min-width: 100% !important; margin: auto !important; }
        #main-row { gap: 20px; }
        #sidebar { background-color: #1f2937; border-radius: 15px; padding: 20px; min-width: min( 240px, 100%) !important; }
        .tool-container { padding: 30px !important; background: none !important; border: none !important; }
        .tool-container h2 { margin-bottom: 2em !important; text-align: center !important; }
        .tool-container .styler { background: none !important; }
        .tool-container .row { column-gap: 1em !important; }
        .tool-container .column { width: 100%; }
        .tool-container .column:not(:has(*)), .tool-container .column:not(:has(:not(div,span))) { display: none !important; }
        .nav-button:hover { border-color: #6366f1 !important; transform: scale(1.02); background: #374151 !important; }
        #header { text-align: center; padding: 25px; margin-bottom: 20px; }
    """
    format_choices = ["MP3", "WAV", "FLAC"]
    language_choices = sorted(list(ALL_LANGUAGES.keys()))
    
    tts_enabled = all([tts_model])

    with gr.Blocks(theme=theme, title="Audio Studio Pro", css=css) as app:
        gr.HTML("""<div id="header"><h1>Audio Studio Pro</h1><p>Your complete suite for professional audio production and AI-powered sound creation.</p></div>""")
        with gr.Row(elem_id="main-row"):
            with gr.Column(scale=1, elem_id="sidebar"):
                nav_master_btn = gr.Button("Mastering", variant="primary", elem_classes="nav-button")
                nav_autotune_btn = gr.Button("Vocal Auto-Tune", variant="secondary", elem_classes="nav-button")
                nav_midi_tools_btn = gr.Button("MIDI Tools", variant="secondary", elem_classes="nav-button")
                nav_audio_extender_btn = gr.Button("Audio Extender", variant="secondary", elem_classes="nav-button")
                nav_stem_mixer_btn = gr.Button("Stem Mixer", variant="secondary", elem_classes="nav-button")
                nav_feedback_btn = gr.Button("Track Feedback", variant="secondary", elem_classes="nav-button")
                nav_instrument_id_btn = gr.Button("Instrument ID", variant="secondary", elem_classes="nav-button")
                nav_video_gen_btn = gr.Button("AI Video Gen", variant="secondary", elem_classes="nav-button")
                nav_speed_btn = gr.Button("Speed & Pitch", variant="secondary", elem_classes="nav-button")
                nav_stem_btn = gr.Button("Stem Separation", variant="secondary", elem_classes="nav-button")
                nav_vps_btn = gr.Button("Vocal Pitch Shifter", variant="secondary", elem_classes="nav-button")
                nav_voice_conv_btn = gr.Button("Voice Conversion", variant="secondary", elem_classes="nav-button")
                nav_dj_btn = gr.Button("DJ AutoMix", variant="secondary", elem_classes="nav-button")
                nav_music_gen_btn = gr.Button("AI Music Gen", variant="secondary", elem_classes="nav-button")
                nav_voice_gen_btn = gr.Button("AI Voice Gen", variant="secondary", elem_classes="nav-button")
                nav_analysis_btn = gr.Button("Analysis", variant="secondary", elem_classes="nav-button")
                nav_stt_btn = gr.Button("Speech-to-Text", variant="secondary", elem_classes="nav-button")
                nav_spectrum_btn = gr.Button("Spectrum", variant="secondary", elem_classes="nav-button")
                nav_beat_vis_btn = gr.Button("Beat Visualizer", variant="secondary", elem_classes="nav-button")
                nav_lyric_vid_btn = gr.Button("Lyric Video", variant="secondary", elem_classes="nav-button")
                nav_chatbot_btn = gr.Button("Support Chat", variant="secondary", elem_classes="nav-button")
            with gr.Column(scale=4, elem_id="main-content"):
                with gr.Group(visible=True, elem_classes="tool-container") as view_master:
                    gr.Markdown("## Mastering")
                    with gr.Row():
                        with gr.Column():
                            master_input = gr.Audio(label="Upload Track", type='filepath')
                            master_strength = gr.Slider(1.0, 2.5, 1.5, step=0.1, label="Mastering Strength")
                            master_format = gr.Radio(format_choices, label="Output Format", value=format_choices[0])
                            with gr.Row(): master_btn = gr.Button("Master Audio", variant="primary"); clear_master_btn = gr.Button("Clear", variant="secondary")
                        with gr.Column():
                             with gr.Group(visible=False) as master_output_box:
                                master_output = gr.Audio(label="Mastered Output", interactive=False, show_download_button=True)
                                master_share_links = gr.Markdown()
                with gr.Group(visible=False, elem_classes="tool-container") as view_autotune:
                    gr.Markdown("## Vocal Auto-Tune")
                    with gr.Row():
                        with gr.Column():
                            autotune_input = gr.Audio(label="Upload Full Song", type='filepath')
                            autotune_strength = gr.Slider(0.1, 1.0, 0.7, step=0.1, label="Tuning Strength")
                            autotune_format = gr.Radio(format_choices, label="Output Format", value=format_choices[0])
                            with gr.Row(): autotune_btn = gr.Button("Auto-Tune Vocals", variant="primary"); clear_autotune_btn = gr.Button("Clear", variant="secondary")
                        with gr.Column():
                            with gr.Group(visible=False) as autotune_output_box:
                                autotune_output = gr.Audio(label="Auto-Tuned Song", interactive=False, show_download_button=True)
                                autotune_share_links = gr.Markdown()
                with gr.Group(visible=False, elem_classes="tool-container") as view_midi_tools:
                    gr.Markdown("## MIDI Tools")
                    with gr.Tabs():
                        with gr.TabItem("Audio to MIDI"):
                            with gr.Row():
                                with gr.Column():
                                    a2m_input = gr.Audio(label="Upload Audio", type='filepath')
                                    with gr.Row(): a2m_btn = gr.Button("Convert to MIDI", variant="primary"); clear_a2m_btn = gr.Button("Clear", variant="secondary")
                                with gr.Column():
                                    with gr.Group(visible=False) as a2m_output_box:
                                        a2m_output = gr.File(label="Output MIDI", interactive=False, show_download_button=True)
                        with gr.TabItem("MIDI to Audio"):
                            with gr.Row():
                                with gr.Column():
                                    m2a_input = gr.File(label="Upload MIDI", file_types=[".mid", ".midi"])
                                    m2a_format = gr.Radio(format_choices, label="Output Format", value=format_choices[0])
                                    with gr.Row(): m2a_btn = gr.Button("Convert to Audio", variant="primary"); clear_m2a_btn = gr.Button("Clear", variant="secondary")
                                with gr.Column():
                                    with gr.Group(visible=False) as m2a_output_box:
                                        m2a_output = gr.Audio(label="Output Audio", interactive=False, show_download_button=True)
                        with gr.TabItem("AI MIDI Enhancer"):
                            with gr.Row():
                                with gr.Column():
                                    enhance_midi_input = gr.File(label="Upload Melody MIDI", file_types=[".mid", ".midi"])
                                    enhance_midi_format = gr.Radio(format_choices, label="Output Format", value=format_choices[0])
                                    enhance_midi_humanize = gr.Checkbox(label="Humanize AI Output", value=True)
                                    with gr.Row(): enhance_midi_btn = gr.Button("Enhance MIDI", variant="primary"); clear_enhance_midi_btn = gr.Button("Clear", variant="secondary")
                                with gr.Column():
                                    with gr.Group(visible=False) as enhance_midi_output_box:
                                        enhance_midi_output = gr.Audio(label="Enhanced Audio", interactive=False, show_download_button=True)
                with gr.Group(visible=False, elem_classes="tool-container") as view_audio_extender:
                    gr.Markdown("## Audio Extender")
                    with gr.Row():
                        with gr.Column():
                            extender_input = gr.Audio(label="Upload Audio to Extend", type='filepath')
                            extender_duration = gr.Slider(5, 60, 15, step=1, label="Extend Duration (seconds)")
                            extender_format = gr.Radio(format_choices, label="Output Format", value=format_choices[0])
                            extender_humanize = gr.Checkbox(label="Humanize AI Output", value=True)
                            with gr.Row(): extender_btn = gr.Button("Extend Audio", variant="primary"); clear_extender_btn = gr.Button("Clear", variant="secondary")
                        with gr.Column():
                            with gr.Group(visible=False) as extender_output_box:
                                extender_output = gr.Audio(label="Extended Audio", interactive=False, show_download_button=True)
                                extender_share_links = gr.Markdown()
                with gr.Group(visible=False, elem_classes="tool-container") as view_stem_mixer:
                    gr.Markdown("## Stem Mixer")
                    with gr.Row():
                        with gr.Column():
                            stem_mixer_files = gr.File(label="Upload Stems (Drums, Bass, Vocals, etc.)", file_count="multiple", type="filepath")
                            stem_mixer_format = gr.Radio(format_choices, label="Output Format", value=format_choices[0])
                            with gr.Row(): stem_mixer_btn = gr.Button("Mix Stems", variant="primary"); clear_stem_mixer_btn = gr.Button("Clear", variant="secondary")
                        with gr.Column():
                            with gr.Group(visible=False) as stem_mixer_output_box:
                                stem_mixer_output = gr.Audio(label="Mixed Track", interactive=False, show_download_button=True)
                                stem_mixer_share_links = gr.Markdown()
                with gr.Group(visible=False, elem_classes="tool-container") as view_feedback:
                    gr.Markdown("## AI Track Feedback")
                    with gr.Row():
                        with gr.Column():
                            feedback_input = gr.Audio(label="Upload Your Track", type='filepath')
                            with gr.Row(): feedback_btn = gr.Button("Get Feedback", variant="primary"); clear_feedback_btn = gr.Button("Clear", variant="secondary")
                        with gr.Column():
                            feedback_output = gr.Markdown(label="Feedback")
                with gr.Group(visible=False, elem_classes="tool-container") as view_instrument_id:
                    gr.Markdown("## Instrument Identification")
                    with gr.Row():
                        with gr.Column():
                            instrument_id_input = gr.Audio(label="Upload Audio", type='filepath')
                            with gr.Row(): instrument_id_btn = gr.Button("Identify Instruments", variant="primary"); clear_instrument_id_btn = gr.Button("Clear", variant="secondary")
                        with gr.Column():
                            instrument_id_output = gr.Markdown(label="Detected Instruments")
                with gr.Group(visible=False, elem_classes="tool-container") as view_video_gen:
                    gr.Markdown("## AI Video Generation")
                    with gr.Row():
                        with gr.Column():
                            video_gen_audio = gr.Audio(label="Upload Audio", type='filepath')
                            video_gen_prompt = gr.Textbox(label="Visual Prompt (e.g., 'blue electric pulse, geometric shapes')", placeholder="Describe the visuals...")
                            video_gen_format = gr.Radio(["MP4"], label="Output Format", value="MP4")
                            with gr.Row(): video_gen_btn = gr.Button("Generate Video", variant="primary"); clear_video_gen_btn = gr.Button("Clear", variant="secondary")
                        with gr.Column():
                            with gr.Group(visible=False) as video_gen_output_box:
                                video_gen_output = gr.Video(label="Generated Video", interactive=False, show_download_button=True)
                                video_gen_share_links = gr.Markdown()
                with gr.Group(visible=False, elem_classes="tool-container") as view_speed:
                    gr.Markdown("## Speed & Pitch")
                    with gr.Row():
                        with gr.Column():
                            speed_input = gr.Audio(label="Upload Track", type='filepath')
                            speed_factor = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.05, label="Speed Factor")
                            preserve_pitch = gr.Checkbox(label="Preserve Pitch (higher quality)", value=True)
                            speed_format = gr.Radio(format_choices, label="Output Format", value=format_choices[0])
                            with gr.Row(): speed_btn = gr.Button("Change Speed", variant="primary"); clear_speed_btn = gr.Button("Clear", variant="secondary")
                        with gr.Column():
                            with gr.Group(visible=False) as speed_output_box:
                                speed_output = gr.Audio(label="Modified Audio", interactive=False, show_download_button=True)
                                speed_share_links = gr.Markdown()
                with gr.Group(visible=False, elem_classes="tool-container") as view_stem:
                    gr.Markdown("## Stem Separation")
                    with gr.Row():
                        with gr.Column():
                            stem_input = gr.Audio(label="Upload Full Mix", type='filepath')
                            stem_type = gr.Radio(["Acapella (Vocals Only)", "Karaoke (Instrumental Only)"], label="Separation Type", value="Acapella (Vocals Only)")
                            stem_format = gr.Radio(format_choices, label="Output Format", value=format_choices[0])
                            with gr.Row(): stem_btn = gr.Button("Separate Stems", variant="primary"); clear_stem_btn = gr.Button("Clear", variant="secondary")
                        with gr.Column():
                            with gr.Group(visible=False) as stem_output_box:
                                stem_output = gr.Audio(label="Separated Track", interactive=False, show_download_button=True)
                                stem_share_links = gr.Markdown()
                with gr.Group(visible=False, elem_classes="tool-container") as view_vps:
                    gr.Markdown("## Vocal Pitch Shifter")
                    with gr.Row():
                        with gr.Column():
                            vps_input = gr.Audio(label="Upload Full Song", type='filepath')
                            vps_pitch = gr.Slider(-12, 12, 0, step=1, label="Vocal Pitch Shift (Semitones)")
                            vps_format = gr.Radio(format_choices, label="Output Format", value=format_choices[0])
                            with gr.Row(): vps_btn = gr.Button("Shift Vocal Pitch", variant="primary"); clear_vps_btn = gr.Button("Clear", variant="secondary")
                        with gr.Column():
                            with gr.Group(visible=False) as vps_output_box:
                                vps_output = gr.Audio(label="Pitch Shifted Song", interactive=False, show_download_button=True)
                                vps_share_links = gr.Markdown()
                with gr.Group(visible=False, elem_classes="tool-container") as view_voice_conv:
                    gr.Markdown("## Voice Conversion")
                    with gr.Row():
                        with gr.Column():
                            vc_ref_audio = gr.Audio(label="Reference Voice Audio (Source Voice)", type='filepath')
                            vc_target_audio = gr.Audio(label="Target Song (Content & Melody)", type='filepath')
                            vc_language = gr.Dropdown(language_choices, label="Language of Target Song", value="English")
                            vc_format = gr.Radio(format_choices, label="Output Format", value=format_choices[0])
                            with gr.Row(): vc_btn = gr.Button("Convert Voice", variant="primary"); clear_vc_btn = gr.Button("Clear", variant="secondary")
                        with gr.Column():
                            with gr.Group(visible=False) as vc_output_box:
                                vc_output = gr.Audio(label="Converted Song", interactive=False, show_download_button=True)
                                vc_share_links = gr.Markdown()
                with gr.Group(visible=False, elem_classes="tool-container") as view_dj:
                    gr.Markdown("## DJ AutoMix")
                    with gr.Row():
                        with gr.Column():
                            dj_files = gr.File(label="Upload Audio Tracks", file_count="multiple", type="filepath")
                            dj_mix_type = gr.Radio(["Simple Crossfade", "Beatmatched Crossfade"], label="Mix Type", value="Beatmatched Crossfade")
                            dj_target_bpm = gr.Number(label="Target BPM (Optional)")
                            dj_transition = gr.Slider(1, 15, 5, step=1, label="Transition Duration (seconds)")
                            dj_format = gr.Radio(format_choices, label="Output Format", value=format_choices[0])
                            with gr.Row(): dj_btn = gr.Button("Create DJ Mix", variant="primary"); clear_dj_btn = gr.Button("Clear", variant="secondary")
                        with gr.Column():
                            with gr.Group(visible=False) as dj_output_box:
                                dj_output = gr.Audio(label="Final DJ Mix", interactive=False, show_download_button=True)
                                dj_share_links = gr.Markdown()
                with gr.Group(visible=False, elem_classes="tool-container") as view_music_gen:
                    gr.Markdown("## AI Music Generation")
                    if DEVICE == "cpu": gr.Markdown("<p style='color:orange;text-align:center;'>Running on a CPU. Music generation will be very slow.</p>")
                    with gr.Row():
                        with gr.Column():
                            gen_prompt = gr.Textbox(lines=4, label="Music Prompt", placeholder="e.g., '80s synthwave, retro, upbeat'")
                            gen_duration = gr.Slider(5, 30, 10, step=1, label="Duration (seconds)")
                            gen_format = gr.Radio(format_choices, label="Output Format", value=format_choices[0])
                            gen_humanize = gr.Checkbox(label="Humanize AI Output", value=True)
                            with gr.Row(): gen_btn = gr.Button("Generate Music", variant="primary", interactive=(generation_model is not None)); clear_gen_btn = gr.Button("Clear", variant="secondary")
                        with gr.Column():
                            with gr.Group(visible=False) as gen_output_box:
                                gen_output = gr.Audio(label="Generated Music", interactive=False, show_download_button=True)
                                gen_share_links = gr.Markdown()
                with gr.Group(visible=False, elem_classes="tool-container") as view_voice_gen:
                    gr.Markdown("## AI Voice Generation")
                    if not tts_enabled: gr.Markdown("<p style='color:red;text-align:center;'>Voice Generation model failed to load and is disabled.</p>")
                    with gr.Row():
                        with gr.Column():
                            vg_ref = gr.Audio(label="Reference Voice (Clear, 5-15s)", type='filepath')
                            vg_text = gr.Textbox(lines=4, label="Text to Speak", placeholder="Enter the text you want the generated voice to say...")
                            vg_format = gr.Radio(format_choices, label="Output Format", value=format_choices[0])
                            vg_humanize = gr.Checkbox(label="Humanize AI Output", value=True)
                            with gr.Row():
                                vg_btn = gr.Button("Generate Voice", variant="primary", interactive=tts_enabled);
                                clear_vg_btn = gr.Button("Clear", variant="secondary")
                        with gr.Column():
                            with gr.Group(visible=False) as vg_output_box:
                                vg_output = gr.Audio(label="Generated Voice Audio", interactive=False, show_download_button=True)
                                vg_share_links = gr.Markdown()
                with gr.Group(visible=False, elem_classes="tool-container") as view_analysis:
                    gr.Markdown("## BPM & Key Analysis")
                    with gr.Row():
                        with gr.Column(scale=1):
                            analysis_input = gr.Audio(label="Upload Audio", type="filepath")
                            with gr.Row(): analysis_btn = gr.Button("Analyze Audio", variant="primary"); clear_analysis_btn = gr.Button("Clear", variant="secondary")
                        with gr.Column(scale=1):
                            analysis_bpm_key_output = gr.Textbox(label="Detected Key & BPM", interactive=False)
                with gr.Group(visible=False, elem_classes="tool-container") as view_stt:
                    gr.Markdown("## Speech-to-Text")
                    if asr_pipeline is None: gr.Markdown("<p style='color:red;text-align:center;'>Speech recognition model failed to load and is disabled.</p>")
                    with gr.Row():
                        with gr.Column():
                            stt_input = gr.Audio(label="Upload Speech Audio", type="filepath")
                            stt_language = gr.Dropdown(language_choices, label="Language", value="English")
                            with gr.Row(): stt_btn = gr.Button("Transcribe Audio", variant="primary", interactive=asr_pipeline is not None); clear_stt_btn = gr.Button("Clear", variant="secondary")
                        with gr.Column():
                            stt_output = gr.Textbox(label="Transcription Result", interactive=False, lines=10)
                            stt_file_output = gr.File(label="Download Transcript", interactive=False, visible=False, show_download_button=True)
                with gr.Group(visible=False, elem_classes="tool-container") as view_spectrum:
                    gr.Markdown("## Spectrum Analyzer")
                    spec_input = gr.Audio(label="Upload Audio", type="filepath")
                    with gr.Row(): spec_btn = gr.Button("Generate Spectrum", variant="primary"); clear_spec_btn = gr.Button("Clear", variant="secondary")
                    spec_output = gr.Image(label="Spectrum Plot", interactive=False)
                with gr.Group(visible=False, elem_classes="tool-container") as view_beat_vis:
                    gr.Markdown("## Beat Visualizer")
                    with gr.Row():
                        with gr.Column():
                            vis_image_input = gr.Image(label="Upload Image", type="filepath"); vis_audio_input = gr.Audio(label="Upload Audio", type="filepath")
                        with gr.Column():
                            vis_effect = gr.Radio(["None", "Blur", "Sharpen", "Contour", "Emboss"], label="Image Effect", value="None")
                            vis_animation = gr.Radio(["None", "Zoom In", "Zoom Out"], label="Animation Style", value="None")
                            vis_intensity = gr.Slider(1.05, 1.5, 1.15, step=0.01, label="Beat Intensity")
                            with gr.Row(): vis_btn = gr.Button("Create Beat Visualizer", variant="primary"); clear_vis_btn = gr.Button("Clear", variant="secondary")
                    with gr.Group(visible=False) as vis_output_box:
                        vis_output = gr.Video(label="Visualizer Output", show_download_button=True); vis_share_links = gr.Markdown()
                with gr.Group(visible=False, elem_classes="tool-container") as view_lyric_vid:
                    gr.Markdown("## Lyric Video Creator")
                    with gr.Row():
                        with gr.Column():
                            lyric_audio = gr.Audio(label="Upload Song", type="filepath"); lyric_bg = gr.File(label="Upload Background (Image or Video)", type="filepath")
                            lyric_language = gr.Dropdown(language_choices, label="Lyric Language", value="English")
                            lyric_position = gr.Radio(["center", "bottom"], label="Text Position", value="bottom")
                            with gr.Row(): lyric_btn = gr.Button("Create Lyric Video", variant="primary"); clear_lyric_btn = gr.Button("Clear", variant="secondary")
                        with gr.Column():
                            lyric_text = gr.Textbox(label="Lyrics", lines=15, placeholder="Enter lyrics here, one line per phrase...")
                            load_transcript_btn = gr.Button("Get Lyrics from Audio (via Speech-to-Text)")
                    with gr.Group(visible=False) as lyric_output_box:
                        lyric_output = gr.Video(label="Lyric Video Output", show_download_button=True); lyric_share_links = gr.Markdown()
                with gr.Group(visible=False, elem_classes="tool-container") as view_chatbot:
                    gr.Markdown("## Support Chat with Fazzer")
                    chatbot_history = gr.Chatbot(label="Audio Studio Pro Support")
                    chatbot_msg = gr.Textbox(label="Your Message", placeholder="Ask me anything about the app...")
                    gr.Examples(
                        examples=[
                            "Who are you?",
                            "Who created this project?",
                            "How does the Mastering tool work?",
                            "What is Stem Mixing?",
                            "Tell me about Voice Conversion.",
                            "How can I extend a song?",
                        ],
                        inputs=chatbot_msg,
                        label="Preset Questions"
                    )
                    clear_chatbot_btn = gr.Button("Clear Chat")

        nav_buttons = {"master": nav_master_btn, "autotune": nav_autotune_btn, "midi_tools": nav_midi_tools_btn, "audio_extender": nav_audio_extender_btn, "stem_mixer": nav_stem_mixer_btn, "feedback": nav_feedback_btn, "instrument_id": nav_instrument_id_btn, "video_gen": nav_video_gen_btn, "speed": nav_speed_btn, "stem": nav_stem_btn, "vps": nav_vps_btn, "voice_conv": nav_voice_conv_btn, "dj": nav_dj_btn, "music_gen": nav_music_gen_btn, "voice_gen": nav_voice_gen_btn, "analysis": nav_analysis_btn, "stt": nav_stt_btn, "spectrum": nav_spectrum_btn, "beat_vis": nav_beat_vis_btn, "lyric_vid": nav_lyric_vid_btn, "chatbot": nav_chatbot_btn}
        views = {"master": view_master, "autotune": view_autotune, "midi_tools": view_midi_tools, "audio_extender": view_audio_extender, "stem_mixer": view_stem_mixer, "feedback": view_feedback, "instrument_id": view_instrument_id, "video_gen": view_video_gen, "speed": view_speed, "stem": view_stem, "vps": view_vps, "voice_conv": view_voice_conv, "dj": view_dj, "music_gen": view_music_gen, "voice_gen": view_voice_gen, "analysis": view_analysis, "stt": view_stt, "spectrum": view_spectrum, "beat_vis": view_beat_vis, "lyric_vid": view_lyric_vid, "chatbot": view_chatbot}

        def switch_view(selected_view):
            view_updates = {view: gr.update(visible=(name == selected_view)) for name, view in views.items()}
            button_updates = {btn: gr.update(variant=("primary" if name == selected_view else "secondary")) for name, btn in nav_buttons.items()}
            return {**view_updates, **button_updates}

        for name, btn in nav_buttons.items(): btn.click(fn=lambda view=name: switch_view(view), inputs=None, outputs=list(views.values()) + list(nav_buttons.values()))

        def create_ui_handler(btn, out_el, out_box, out_share, logic_func, *inputs):
            def ui_handler_generator(*args):
                yield (gr.update(value="Processing...", interactive=False), gr.update(visible=False), gr.update(value=None))
                try:
                    result = logic_func(*args)
                    yield (gr.update(value=btn.value, interactive=True), gr.update(visible=True), gr.update(value=result))
                except Exception as e:
                    raise gr.Error(str(e))
            btn.click(ui_handler_generator, inputs=inputs, outputs=[btn, out_box, out_el])

        create_ui_handler(master_btn, master_output, master_output_box, master_share_links, _master_logic, master_input, master_strength, master_format)
        create_ui_handler(autotune_btn, autotune_output, autotune_output_box, autotune_share_links, _autotune_vocals_logic, autotune_input, autotune_strength, autotune_format)
        create_ui_handler(a2m_btn, a2m_output, a2m_output_box, None, _audio_to_midi_logic, a2m_input)
        create_ui_handler(m2a_btn, m2a_output, m2a_output_box, None, _midi_to_audio_logic, m2a_input, m2a_format)
        create_ui_handler(enhance_midi_btn, enhance_midi_output, enhance_midi_output_box, None, _enhance_midi_logic, enhance_midi_input, enhance_midi_format, enhance_midi_humanize)
        create_ui_handler(extender_btn, extender_output, extender_output_box, extender_share_links, _extend_audio_logic, extender_input, extender_duration, extender_format, extender_humanize)
        create_ui_handler(stem_mixer_btn, stem_mixer_output, stem_mixer_output_box, stem_mixer_share_links, _stem_mixer_logic, stem_mixer_files, stem_mixer_format)
        create_ui_handler(video_gen_btn, video_gen_output, video_gen_output_box, video_gen_share_links, _generate_video_logic, video_gen_audio, video_gen_prompt, video_gen_format)
        create_ui_handler(speed_btn, speed_output, speed_output_box, speed_share_links, _change_audio_speed_logic, speed_input, speed_factor, preserve_pitch, speed_format)
        create_ui_handler(stem_btn, stem_output, stem_output_box, stem_share_links, _separate_stems_logic, stem_input, stem_type, stem_format)
        create_ui_handler(vps_btn, vps_output, vps_output_box, vps_share_links, _pitch_shift_vocals_logic, vps_input, vps_pitch, vps_format)
        create_ui_handler(vc_btn, vc_output, vc_output_box, vc_share_links, _voice_conversion_logic, vc_ref_audio, vc_target_audio, vc_language, vc_format)
        create_ui_handler(dj_btn, dj_output, dj_output_box, dj_share_links, _auto_dj_mix_logic, dj_files, dj_mix_type, dj_target_bpm, dj_transition, dj_format)
        create_ui_handler(gen_btn, gen_output, gen_output_box, gen_share_links, _generate_music_logic, gen_prompt, gen_duration, gen_format, gen_humanize)
        create_ui_handler(vg_btn, vg_output, vg_output_box, vg_share_links, _generate_voice_logic, vg_text, vg_ref, vg_format, vg_humanize)
        create_ui_handler(vis_btn, vis_output, vis_output_box, vis_share_links, _create_beat_visualizer_logic, vis_image_input, vis_audio_input, vis_effect, vis_animation, vis_intensity)
        create_ui_handler(lyric_btn, lyric_output, lyric_output_box, lyric_share_links, _create_lyric_video_logic, lyric_audio, lyric_bg, lyric_text, lyric_position, lyric_language)

        def feedback_ui(audio_path):
            yield {feedback_btn: gr.update(value="Analyzing...", interactive=False), feedback_output: ""}
            try:
                feedback_text = _get_feedback_logic(audio_path)
                yield {feedback_btn: gr.update(value="Get Feedback", interactive=True), feedback_output: feedback_text}
            except Exception as e:
                raise gr.Error(str(e))
        feedback_btn.click(feedback_ui, [feedback_input], [feedback_btn, feedback_output])

        def instrument_id_ui(audio_path):
            yield {instrument_id_btn: gr.update(value="Identifying...", interactive=False), instrument_id_output: ""}
            try:
                instrument_text = _identify_instruments_logic(audio_path)
                yield {instrument_id_btn: gr.update(value="Identify Instruments", interactive=True), instrument_id_output: instrument_text}
            except Exception as e:
                raise gr.Error(str(e))
        instrument_id_btn.click(instrument_id_ui, [instrument_id_input], [instrument_id_btn, instrument_id_output])

        def analysis_ui(audio_path):
            yield {analysis_btn: gr.update(value="Analyzing...", interactive=False), analysis_bpm_key_output: ""}
            try:
                bpm_key = _analyze_audio_features_logic(audio_path)
                yield {analysis_btn: gr.update(value="Analyze Audio", interactive=True), analysis_bpm_key_output: bpm_key}
            except Exception as e:
                raise gr.Error(str(e))
        analysis_btn.click(analysis_ui, [analysis_input], [analysis_btn, analysis_bpm_key_output])

        def stt_ui(audio_path, language):
            yield {stt_btn: gr.update(value="Transcribing...", interactive=False), stt_output: "", stt_file_output: gr.update(visible=False)}
            try:
                transcript = _transcribe_audio_logic(audio_path, language)
                file_path = save_text_to_file(transcript)
                yield {stt_btn: gr.update(value="Transcribe Audio", interactive=True), stt_output: transcript, stt_file_output: gr.update(visible=True, value=file_path)}
            except Exception as e:
                raise gr.Error(str(e))
        stt_btn.click(stt_ui, [stt_input, stt_language], [stt_btn, stt_output, stt_file_output])

        def spec_ui(audio_path):
            yield {spec_btn: gr.update(value="Generating...", interactive=False), spec_output: None}
            try:
                spec_image = _create_spectrum_visualization_logic(audio_path)
                yield {spec_btn: gr.update(value="Generate Spectrum", interactive=True), spec_output: spec_image}
            except Exception as e:
                raise gr.Error(str(e))
        spec_btn.click(spec_ui, [spec_input], [spec_btn, spec_output])

        chatbot_msg.submit(_chatbot_response_logic, [chatbot_msg, chatbot_history], [chatbot_msg, chatbot_history])
        clear_chatbot_btn.click(lambda: (None, None), None, [chatbot_msg, chatbot_history])

        def clear_ui(*components):
            updates = {}
            for comp in components:
                if isinstance(comp, (gr.Audio, gr.Video, gr.Image, gr.File, gr.Textbox, gr.Markdown)):
                    updates[comp] = None
                if isinstance(comp, gr.Group):
                    updates[comp] = gr.update(visible=False)
            return updates

        clear_master_btn.click(lambda: clear_ui(master_input, master_output, master_output_box), [], [master_input, master_output, master_output_box])
        clear_autotune_btn.click(lambda: clear_ui(autotune_input, autotune_output, autotune_output_box), [], [autotune_input, autotune_output, autotune_output_box])
        clear_a2m_btn.click(lambda: clear_ui(a2m_input, a2m_output, a2m_output_box), [], [a2m_input, a2m_output, a2m_output_box])
        clear_m2a_btn.click(lambda: clear_ui(m2a_input, m2a_output, m2a_output_box), [], [m2a_input, m2a_output, m2a_output_box])
        clear_enhance_midi_btn.click(lambda: clear_ui(enhance_midi_input, enhance_midi_output, enhance_midi_output_box), [], [enhance_midi_input, enhance_midi_output, enhance_midi_output_box])
        clear_extender_btn.click(lambda: clear_ui(extender_input, extender_output, extender_output_box), [], [extender_input, extender_output, extender_output_box])
        clear_stem_mixer_btn.click(lambda: clear_ui(stem_mixer_files, stem_mixer_output, stem_mixer_output_box), [], [stem_mixer_files, stem_mixer_output, stem_mixer_output_box])
        clear_feedback_btn.click(lambda: clear_ui(feedback_input, feedback_output), [], [feedback_input, feedback_output])
        clear_instrument_id_btn.click(lambda: clear_ui(instrument_id_input, instrument_id_output), [], [instrument_id_input, instrument_id_output])
        clear_video_gen_btn.click(lambda: {**clear_ui(video_gen_audio, video_gen_output, video_gen_output_box), **{video_gen_prompt: ""}}, [], [video_gen_audio, video_gen_output, video_gen_output_box, video_gen_prompt])
        clear_speed_btn.click(lambda: clear_ui(speed_input, speed_output, speed_output_box), [], [speed_input, speed_output, speed_output_box])
        clear_stem_btn.click(lambda: clear_ui(stem_input, stem_output, stem_output_box), [], [stem_input, stem_output, stem_output_box])
        clear_vps_btn.click(lambda: clear_ui(vps_input, vps_output, vps_output_box), [], [vps_input, vps_output, vps_output_box])
        clear_vc_btn.click(lambda: clear_ui(vc_ref_audio, vc_target_audio, vc_output, vc_output_box), [], [vc_ref_audio, vc_target_audio, vc_output, vc_output_box])
        clear_dj_btn.click(lambda: clear_ui(dj_files, dj_output, dj_output_box), [], [dj_files, dj_output, dj_output_box])
        clear_gen_btn.click(lambda: {**clear_ui(gen_output, gen_output_box), **{gen_prompt: ""}}, [], [gen_output, gen_output_box, gen_prompt])
        clear_vg_btn.click(lambda: {**clear_ui(vg_ref, vg_output, vg_output_box), **{vg_text: ""}}, [], [vg_ref, vg_output, vg_output_box, vg_text])
        clear_analysis_btn.click(lambda: {**clear_ui(analysis_input), **{analysis_bpm_key_output: ""}}, [], [analysis_input, analysis_bpm_key_output])
        clear_stt_btn.click(lambda: clear_ui(stt_input, stt_output, stt_file_output), [], [stt_input, stt_output, stt_file_output])
        clear_spec_btn.click(lambda: clear_ui(spec_input, spec_output), [], [spec_input, spec_output])
        clear_vis_btn.click(lambda: clear_ui(vis_image_input, vis_audio_input, vis_output, vis_output_box), [], [vis_image_input, vis_audio_input, vis_output, vis_output_box])
        clear_lyric_btn.click(lambda: {**clear_ui(lyric_audio, lyric_bg, lyric_output, lyric_output_box), **{lyric_text: ""}}, [], [lyric_audio, lyric_bg, lyric_output, lyric_output_box, lyric_text])

        load_transcript_btn.click(lambda audio, lang: _transcribe_audio_logic(audio, lang), [lyric_audio, lyric_language], [lyric_text])

    app.queue(max_size=20).launch(debug=True)

if __name__ == "__main__":
    main()
