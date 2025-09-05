import gradio as gr
import spaces

from definers import (
    apt_install,
    install_faiss,
    install_ffmpeg,
    install_audio_effects,
    init_pretrained_model,
    set_system_message,
    transcribe_audio,
    generate_voice,
    cwd,
    convert_vocal_rvc,
    train_model_rvc,
    master,
    generate_music,
    dj_mix,
    beat_visualizer,
    lyric_video,
    stretch_audio,
    get_audio_feedback,
    music_video,
    change_audio_speed,
    analyze_audio_features,
    separate_stems,
    pitch_shift_vocals,
    create_spectrum_visualization,
    stem_mixer,
    identify_instruments,
    extend_audio,
    audio_to_midi,
    midi_to_audio,
    autotune_vocals,
    language_codes,
    save_temp_text as save_text_to_file,
    init_chat,
    device,
    random_string,
    create_share_links
)

apt_install()
install_faiss()
install_ffmpeg()
install_audio_effects()

init_pretrained_model("tts")
init_pretrained_model("svc")
init_pretrained_model("speech-recognition")
init_pretrained_model("audio-classification")
init_pretrained_model("music")
init_pretrained_model("answer")

set_system_message(
    name="Fazzer",
    role="the official AI assistant for the 'Audio Studio Pro' application",
    tone="concise, clear, friendly, helpful, and encouraging tone about audio production",
    goals=[
        'answer users questions',
        'guide users with the application usage',
        'explain the purpose of each tool in the application',
        'provide simple, step-by-step instructions on how to use the features based on their UI',
    ],
    chattiness="provide detailed explanations",
    persona_data={
        "the name of the software you help with": "Audio Studio Pro",
        "the name of your creator": "Yaron Koresh",
        "the origin country of your creator": "Israel",
        "your name": "Fazzer",
        "the complete list of the software's features with usage instructions": """
a mastering tool to automatically enhances a track's loudness and clarity - upload your track, adjust the 'Mastering Strength' slider, choose an output format, and click 'Master Audio';
a vocal auto-tuning tools that correct the pitch of vocals and aligns them to the beat - upload a full song, set the 'Tuning Strength' (1.0 is maximum), and click 'Auto-Tune Vocals';
audio to midi converter - upload an audio file and click 'Convert to MIDI';
midi to audio converter - upload a MIDI file and click 'Convert to Audio';
AI midi enhancer which takes a simple midi melody and uses AI to generate a fuller musical arrangement - upload a MIDI melody file, choose an output format, choose if to 'Humanize AI Output', and click 'Enhance MIDI';
an audio extender that uses AI to seamlessly continue a piece of music - upload your audio, use the 'Extend Duration' slider to choose how many seconds to add, and click 'Extend Audio';
a stem mixer that mixes individual instrument tracks (stems) together - upload multiple audio files (e.g., drums.wav, bass.wav). The tool automatically beatmatches them to the first track and mixes them;
a track feedbacks generator that provides AI-powered analysis and advice on your mix - upload your track and click 'Get Feedback' for written analysis on its dynamics, stereo width, and frequency balance;
an instrument identifier from an audio file - upload an audio file and click 'Identify Instruments';
an AI video generator which creates a simple and abstract music visualizer - upload an audio file and click 'Generate Video' to create a video with a pulsing circle that reacts to the music;
a speed & pitch changer which changes the playback speed of a track - upload audio, use the 'Speed Factor' slider (e.g., 1.5x is faster), and check 'Preserve Pitch' for a more natural sound;
a stems separator which splits a song into vocals and instrumental - upload a full song and choose either 'Acapella (Vocals Only)' or 'Karaoke (Instrumental Only)';
a vocal pitch shifter which changes the pitch of only the vocals in a song - upload a song and use the 'Vocal Pitch Shift' slider to raise or lower the vocal pitch in semitones;
a RVC based voice lab tool for voice cloning & conversion tool which converts a song using a different AI voice, preserving the melody - upload your training audio files, click 'Train' to create a voice model, then use the 'Convert' tab to apply that voice to a new audio input;
a dj tool which automatically mixes multiple songs together - upload two or more tracks. Choose 'Beatmatched Crossfade' for a smooth, tempo-synced mix and adjust the 'Transition Duration';
an AI music generator which creates original music from a text description - write a description of the music you want (e.g., 'upbeat synthwave'), set the duration, and click 'Generate Music';
an AI voice generator which clones a voice to say anything you type - upload a clean 5-15 second 'Reference Voice' sample, type the 'Text to Speak', and click 'Generate Voice';
a bpm & key analysis tools which detects a track's musical key and tempo - upload your audio and click 'Analyze Audio';
a speech-to-text tool which transcribes speech from an audio file into text - upload an audio file with speech, select the language, and click 'Transcribe Audio'.
a spectrum analyzer which creates a visual graph (spectrogram) of an audio's frequencies - upload an audio file and click 'Generate Spectrum'.
a beat visualizer which creates a video where an image pulses to the music's beat - upload an image and an audio file. Adjust the 'Beat Intensity' slider to control how much the image reacts.
a lyric video creation tool which creates a simple lyric video - upload a song and a background image/video. Then, paste your lyrics into the text box, with each line representing a new phrase on screen.
a support chat (that's you!) which answer questions like 'What is Stem Mixing?' or 'How do I use the Vocal Pitch Shifter?' based on his knowledge-base;
"""
    },
    task_rules=[
        "If you don't know the answer, politely say so. Do not make up features"
    ],
    interaction_style="ask clarifying questions before answering if it will make your answer more accurate"
);

@spaces.GPU(duration=60)
def _transcribe_audio_logic(audio_path, language):
    return transcribe_audio(audio_path, language)

@spaces.GPU(duration=30)
def _generate_voice_logic(text, reference_audio, format_choice, humanize):
    return generate_voice(text, reference_audio, format_choice, humanize)

@spaces.GPU(duration=50)
def handle_conversion(experiment,inp,ptch):
    with cwd():
        return convert_vocal_rvc(experiment,inp,ptch)

@spaces.GPU(duration=350)
def handle_training(experiment,inp,lvl):
    with cwd():
        return train_model_rvc(experiment,inp,lvl), lvl+1

def _master_logic(source_path, strength, format_choice):
    return master(source_path, strength, format_choice)

@spaces.GPU(duration=60)
def _generate_music_logic(prompt, duration_s, format_choice, humanize):
    return generate_music(prompt, duration_s, format_choice, humanize)

def _auto_dj_mix_logic(files, mix_type, target_bpm, transition_sec, format_choice):
    return dj_mix(files, mix_type, target_bpm, transition_sec, format_choice)

@spaces.GPU(duration=180)
def _create_beat_visualizer_logic(image_path, audio_path, image_effect, animation_style, scale_intensity):
    return beat_visualizer(image_path, audio_path, image_effect, animation_style, scale_intensity)

@spaces.GPU(duration=180)
def _create_lyric_video_logic(audio_path, background_path, lyrics_text, text_position):
    return lyric_video(audio_path, background_path, lyrics_text, text_position)

def stretch_audio_cli(input_path, output_path, speed_factor, crispness):
    return stretch_audio(input_path, output_path, speed_factor, crispness)

def _analyze_audio_features_logic(audio_path):
    return analyze_audio_features(audio_path)

def _change_audio_speed_logic(audio_path, speed_factor, preserve_pitch, format_choice):
    return change_audio_speed(audio_path, speed_factor, preserve_pitch, format_choice)

@spaces.GPU(duration=120)
def _separate_stems_logic(audio_path, separation_type, format_choice):
    return separate_stems(audio_path, separation_type, format_choice)

@spaces.GPU(duration=120)
def _pitch_shift_vocals_logic(audio_path, pitch_shift, format_choice):
    return pitch_shift_vocals(audio_path, pitch_shift, format_choice)

def _create_spectrum_visualization_logic(audio_path):
    return create_spectrum_visualization(audio_path)

@spaces.GPU(duration=180)
def _stem_mixer_logic(files, format_choice):
    return stem_mixer(files, format_choice)

@spaces.GPU(duration=30)
def _get_feedback_logic(audio_path):
    return get_audio_feedback(audio_path)

@spaces.GPU(duration=180)
def _generate_video_logic(audio_path):
    return music_video(audio_path)

@spaces.GPU(duration=30)
def _identify_instruments_logic(audio_path):
    return identify_instruments(audio_path)

@spaces.GPU(duration=120)
def _extend_audio_logic(audio_path, extend_duration_s, format_choice, humanize):
    return extend_audio(audio_path, extend_duration_s, format_choice, humanize)

@spaces.GPU(duration=60)
def _audio_to_midi_logic(audio_path):
    return audio_to_midi(audio_path)

@spaces.GPU(duration=30)
def _midi_to_audio_logic(midi_path, format_choice):
    return midi_to_audio(midi_path, format_choice)

@spaces.GPU(duration=240)
def _autotune_vocals_logic(audio_path, strength, format_choice):
    return autotune_vocals(audio_path, strength, format_choice)

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
    language_choices = sorted(list(language_codes.values()))
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
                nav_voice_lab_btn = gr.Button("Voice Lab", variant="secondary", elem_classes="nav-button")
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
                            master_strength = gr.Slider(1.0, 2.0, 1.8, step=0.1, label="Mastering Strength")
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
                                        a2m_output = gr.File(label="Output MIDI", interactive=False)
                                        a2m_share_links = gr.Markdown()
                        with gr.TabItem("MIDI to Audio"):
                            with gr.Row():
                                with gr.Column():
                                    m2a_input = gr.File(label="Upload MIDI", file_types=[".mid", ".midi"])
                                    m2a_format = gr.Radio(format_choices, label="Output Format", value=format_choices[0])
                                    with gr.Row(): m2a_btn = gr.Button("Convert to Audio", variant="primary"); clear_m2a_btn = gr.Button("Clear", variant="secondary")
                                with gr.Column():
                                    with gr.Group(visible=False) as m2a_output_box:
                                        m2a_output = gr.Audio(label="Output Audio", interactive=False, show_download_button=True)
                                        m2a_share_links = gr.Markdown()
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
                    gr.Markdown("## AI Music Clip Generation")
                    with gr.Row():
                        with gr.Column():
                            video_gen_audio = gr.Audio(label="Upload Audio", type='filepath')
                            with gr.Row(): video_gen_btn = gr.Button("Generate Video", variant="primary"); clear_video_gen_btn = gr.Button("Clear", variant="secondary")
                        with gr.Column():
                            with gr.Group(visible=False) as video_gen_output_box:
                                video_gen_output = gr.Video(label="Generated Clip", interactive=False, show_download_button=True)
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
                with gr.Group(visible=False, elem_classes="tool-container") as view_voice_lab:
                    gr.Markdown("## 🔬 Voice Lab")
                    with gr.Row(visible=False):
                        experiment = gr.Textbox(
                            value=random_string()
                        )
                    with gr.Row():
                        inp = gr.File(
                            label="Input",
                            type="filepath"
                        )
                        outp = gr.File(
                            label="Output",
                            type="filepath",
                            file_count="multiple"
                        )
                    with gr.Row():
                        ptch = gr.Number(label="Pitch",value=0,minimum=-12,maximum=12,step=1)
                    with gr.Row(visible=False):
                        lvl = gr.Number(label="(re-)training step",value=1,minimum=1,step=1)
                    with gr.Row():
                        but1 = gr.Button("Train", variant="primary")
                        but1.click( fn=handle_training, inputs=[experiment,inp,lvl], outputs=[outp,lvl] )
                        but2 = gr.Button("Convert", variant="primary")
                        but2.click( fn=handle_conversion, inputs=[experiment,inp,ptch], outputs=[outp] )
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
                    if device() == "cpu": gr.Markdown("<p style='color:orange;text-align:center;'>Running on a CPU. Music generation will be very slow.</p>")
                    with gr.Row():
                        with gr.Column():
                            gen_prompt = gr.Textbox(lines=4, label="Music Prompt", placeholder="e.g., '80s synthwave, retro, upbeat'")
                            gen_duration = gr.Slider(5, 30, 10, step=1, label="Duration (seconds)")
                            gen_format = gr.Radio(format_choices, label="Output Format", value=format_choices[0])
                            gen_humanize = gr.Checkbox(label="Humanize AI Output", value=True)
                            with gr.Row(): gen_btn = gr.Button("Generate Music", variant="primary", interactive=True); clear_gen_btn = gr.Button("Clear", variant="secondary")
                        with gr.Column():
                            with gr.Group(visible=False) as gen_output_box:
                                gen_output = gr.Audio(label="Generated Music", interactive=False, show_download_button=True)
                                gen_share_links = gr.Markdown()
                with gr.Group(visible=False, elem_classes="tool-container") as view_voice_gen:
                    gr.Markdown("## AI Voice Generation")
                    with gr.Row():
                        with gr.Column():
                            vg_ref = gr.Audio(label="Reference Voice (Clear, 5-15s)", type='filepath')
                            vg_text = gr.Textbox(lines=4, label="Text to Speak", placeholder="Enter the text you want the generated voice to say...")
                            vg_format = gr.Radio(format_choices, label="Output Format", value=format_choices[0])
                            vg_humanize = gr.Checkbox(label="Humanize AI Output", value=True)
                            with gr.Row():
                                vg_btn = gr.Button("Generate Voice", variant="primary", interactive=True);
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
                    with gr.Row():
                        with gr.Column():
                            stt_input = gr.Audio(label="Upload Speech Audio", type="filepath")
                            stt_language = gr.Dropdown(language_choices, label="Language", value="english")
                            with gr.Row(): stt_btn = gr.Button("Transcribe Audio", variant="primary", interactive=True); clear_stt_btn = gr.Button("Clear", variant="secondary")
                        with gr.Column():
                            stt_output = gr.Textbox(label="Transcription Result", interactive=False, lines=10)
                            stt_file_output = gr.File(label="Download Transcript", interactive=False, visible=False)
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
                            lyric_position = gr.Radio(["center", "bottom"], label="Text Position", value="bottom")
                            lyric_language = gr.Dropdown(language_choices, label="Language", value="english")
                            with gr.Row(): lyric_btn = gr.Button("Create Lyric Video", variant="primary"); clear_lyric_btn = gr.Button("Clear", variant="secondary")
                        with gr.Column():
                            lyric_text = gr.Textbox(label="Lyrics", lines=15, placeholder="Enter lyrics here, one line per phrase...")
                            load_transcript_btn = gr.Button("Get Lyrics from Audio (via Speech-to-Text)")
                    with gr.Group(visible=False) as lyric_output_box:
                        lyric_output = gr.Video(label="Lyric Video Output", show_download_button=True); lyric_share_links = gr.Markdown()
                with gr.Group(visible=False, elem_classes="tool-container") as view_chatbot:
                    chat = init_chat(
                        "Audio Studio Pro support",
                    )
        nav_buttons = {"master": nav_master_btn, "autotune": nav_autotune_btn, "midi_tools": nav_midi_tools_btn, "audio_extender": nav_audio_extender_btn, "stem_mixer": nav_stem_mixer_btn, "feedback": nav_feedback_btn, "instrument_id": nav_instrument_id_btn, "video_gen": nav_video_gen_btn, "speed": nav_speed_btn, "stem": nav_stem_btn, "vps": nav_vps_btn, "voice_lab": nav_voice_lab_btn, "dj": nav_dj_btn, "music_gen": nav_music_gen_btn, "voice_gen": nav_voice_gen_btn, "analysis": nav_analysis_btn, "stt": nav_stt_btn, "spectrum": nav_spectrum_btn, "beat_vis": nav_beat_vis_btn, "lyric_vid": nav_lyric_vid_btn, "chatbot": nav_chatbot_btn}
        views = {"master": view_master, "autotune": view_autotune, "midi_tools": view_midi_tools, "audio_extender": view_audio_extender, "stem_mixer": view_stem_mixer, "feedback": view_feedback, "instrument_id": view_instrument_id, "video_gen": view_video_gen, "speed": view_speed, "stem": view_stem, "vps": view_vps, "voice_lab": view_voice_lab, "dj": view_dj, "music_gen": view_music_gen, "voice_gen": view_voice_gen, "analysis": view_analysis, "stt": view_stt, "spectrum": view_spectrum, "beat_vis": view_beat_vis, "lyric_vid": view_lyric_vid, "chatbot": view_chatbot}

        def switch_view(selected_view):
            view_updates = {view: gr.update(visible=(name == selected_view)) for name, view in views.items()}
            button_updates = {btn: gr.update(variant=("primary" if name == selected_view else "secondary")) for name, btn in nav_buttons.items()}
            return {**view_updates, **button_updates}

        for name, btn in nav_buttons.items(): btn.click(fn=lambda view=name: switch_view(view), inputs=None, outputs=list(views.values()) + list(nav_buttons.values()))

        def create_ui_handler(btn, out_el, out_box, out_share, logic_func, *inputs):
            def ui_handler_generator(*args):
                yield (gr.update(value="Processing...", interactive=False), gr.update(visible=False), gr.update(value=None), gr.update(value=""))
                try:
                    result = logic_func(*args)
                    share_text = "Check out this creation from Audio Studio Pro! 🎶"
                    share_html = create_share_links(result, share_text)
                    yield (gr.update(value=btn.value, interactive=True), gr.update(visible=True), gr.update(value=result), gr.update(value=share_html))
                except Exception as e:
                    yield (gr.update(value=btn.value, interactive=True), gr.update(visible=False), gr.update(value=None), gr.update(value=""))
                    raise gr.Error(str(e))
            btn.click(ui_handler_generator, inputs=inputs, outputs=[btn, out_box, out_el, out_share])

        create_ui_handler(master_btn, master_output, master_output_box, master_share_links, _master_logic, master_input, master_strength, master_format)
        create_ui_handler(autotune_btn, autotune_output, autotune_output_box, autotune_share_links, _autotune_vocals_logic, autotune_input, autotune_strength, autotune_format)
        create_ui_handler(a2m_btn, a2m_output, a2m_output_box, a2m_share_links, _audio_to_midi_logic, a2m_input)
        create_ui_handler(m2a_btn, m2a_output, m2a_output_box, m2a_share_links, _midi_to_audio_logic, m2a_input, m2a_format)
        create_ui_handler(extender_btn, extender_output, extender_output_box, extender_share_links, _extend_audio_logic, extender_input, extender_duration, extender_format, extender_humanize)
        create_ui_handler(stem_mixer_btn, stem_mixer_output, stem_mixer_output_box, stem_mixer_share_links, _stem_mixer_logic, stem_mixer_files, stem_mixer_format)
        create_ui_handler(video_gen_btn, video_gen_output, video_gen_output_box, video_gen_share_links, _generate_video_logic, video_gen_audio)
        create_ui_handler(speed_btn, speed_output, speed_output_box, speed_share_links, _change_audio_speed_logic, speed_input, speed_factor, preserve_pitch, speed_format)
        create_ui_handler(stem_btn, stem_output, stem_output_box, stem_share_links, _separate_stems_logic, stem_input, stem_type, stem_format)
        create_ui_handler(vps_btn, vps_output, vps_output_box, vps_share_links, _pitch_shift_vocals_logic, vps_input, vps_pitch, vps_format)
        create_ui_handler(dj_btn, dj_output, dj_output_box, dj_share_links, _auto_dj_mix_logic, dj_files, dj_mix_type, dj_target_bpm, dj_transition, dj_format)
        create_ui_handler(gen_btn, gen_output, gen_output_box, gen_share_links, _generate_music_logic, gen_prompt, gen_duration, gen_format, gen_humanize)
        create_ui_handler(vg_btn, vg_output, vg_output_box, vg_share_links, _generate_voice_logic, vg_text, vg_ref, vg_format, vg_humanize)
        create_ui_handler(vis_btn, vis_output, vis_output_box, vis_share_links, _create_beat_visualizer_logic, vis_image_input, vis_audio_input, vis_effect, vis_animation, vis_intensity)
        create_ui_handler(lyric_btn, lyric_output, lyric_output_box, lyric_share_links, _create_lyric_video_logic, lyric_audio, lyric_bg, lyric_text, lyric_position)
        
        def feedback_ui(audio_path):
            yield {feedback_btn: gr.update(value="Analyzing...", interactive=False), feedback_output: ""}
            try:
                feedback_text = _get_feedback_logic(audio_path)
                yield {feedback_btn: gr.update(value="Get Feedback", interactive=True), feedback_output: feedback_text}
            except Exception as e:
                yield {feedback_btn: gr.update(value="Get Feedback", interactive=True)}
                raise gr.Error(str(e))
        feedback_btn.click(feedback_ui, [feedback_input], [feedback_btn, feedback_output])

        def instrument_id_ui(audio_path):
            yield {instrument_id_btn: gr.update(value="Identifying...", interactive=False), instrument_id_output: ""}
            try:
                instrument_text = _identify_instruments_logic(audio_path)
                yield {instrument_id_btn: gr.update(value="Identify Instruments", interactive=True), instrument_id_output: instrument_text}
            except Exception as e:
                yield {instrument_id_btn: gr.update(value="Identify Instruments", interactive=True)}
                raise gr.Error(str(e))
        instrument_id_btn.click(instrument_id_ui, [instrument_id_input], [instrument_id_btn, instrument_id_output])

        def analysis_ui(audio_path):
            yield {analysis_btn: gr.update(value="Analyzing...", interactive=False), analysis_bpm_key_output: ""}
            try:
                bpm_key = _analyze_audio_features_logic(audio_path)
                yield {analysis_btn: gr.update(value="Analyze Audio", interactive=True), analysis_bpm_key_output: bpm_key}
            except Exception as e:
                yield {analysis_btn: gr.update(value="Analyze Audio", interactive=True)}
                raise gr.Error(str(e))
        analysis_btn.click(analysis_ui, [analysis_input], [analysis_btn, analysis_bpm_key_output])

        def stt_ui(audio_path, language):
            yield {stt_btn: gr.update(value="Transcribing...", interactive=False), stt_output: "", stt_file_output: gr.update(visible=False)}
            try:
                transcript = _transcribe_audio_logic(audio_path, language)
                file_path = save_text_to_file(transcript)
                yield {stt_btn: gr.update(value="Transcribe Audio", interactive=True), stt_output: transcript, stt_file_output: gr.update(visible=True, value=file_path)}
            except Exception as e:
                yield {stt_btn: gr.update(value="Transcribe Audio", interactive=True)}
                raise gr.Error(str(e))
        stt_btn.click(stt_ui, [stt_input, stt_language], [stt_btn, stt_output, stt_file_output])

        def spec_ui(audio_path):
            yield {spec_btn: gr.update(value="Generating...", interactive=False), spec_output: None}
            try:
                spec_image = _create_spectrum_visualization_logic(audio_path)
                yield {spec_btn: gr.update(value="Generate Spectrum", interactive=True), spec_output: spec_image}
            except Exception as e:
                yield {spec_btn: gr.update(value="Generate Spectrum", interactive=True)}
                raise gr.Error(str(e))
        spec_btn.click(spec_ui, [spec_input], [spec_btn, spec_output])

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
        clear_extender_btn.click(lambda: clear_ui(extender_input, extender_output, extender_output_box), [], [extender_input, extender_output, extender_output_box])
        clear_stem_mixer_btn.click(lambda: clear_ui(stem_mixer_files, stem_mixer_output, stem_mixer_output_box), [], [stem_mixer_files, stem_mixer_output, stem_mixer_output_box])
        clear_feedback_btn.click(lambda: clear_ui(feedback_input, feedback_output), [], [feedback_input, feedback_output])
        clear_instrument_id_btn.click(lambda: clear_ui(instrument_id_input, instrument_id_output), [], [instrument_id_input, instrument_id_output])
        clear_video_gen_btn.click(lambda: clear_ui(video_gen_audio, video_gen_output, video_gen_output_box), [], [video_gen_audio, video_gen_output, video_gen_output_box])
        clear_speed_btn.click(lambda: clear_ui(speed_input, speed_output, speed_output_box), [], [speed_input, speed_output, speed_output_box])
        clear_stem_btn.click(lambda: clear_ui(stem_input, stem_output, stem_output_box), [], [stem_input, stem_output, stem_output_box])
        clear_vps_btn.click(lambda: clear_ui(vps_input, vps_output, vps_output_box), [], [vps_input, vps_output, vps_output_box])
        clear_dj_btn.click(lambda: clear_ui(dj_files, dj_output, dj_output_box), [], [dj_files, dj_output, dj_output_box])
        clear_gen_btn.click(lambda: {**clear_ui(gen_output, gen_output_box), **{gen_prompt: ""}}, [], [gen_output, gen_output_box, gen_prompt])
        clear_vg_btn.click(lambda: {**clear_ui(vg_ref, vg_output, vg_output_box), **{vg_text: ""}}, [], [vg_ref, vg_output, vg_output_box, vg_text])
        clear_analysis_btn.click(lambda: {**clear_ui(analysis_input), **{analysis_bpm_key_output: ""}}, [], [analysis_input, analysis_bpm_key_output])
        clear_stt_btn.click(lambda: clear_ui(stt_input, stt_output, stt_file_output), [], [stt_input, stt_output, stt_file_output])
        clear_spec_btn.click(lambda: clear_ui(spec_input, spec_output), [], [spec_input, spec_output])
        clear_vis_btn.click(lambda: clear_ui(vis_image_input, vis_audio_input, vis_output, vis_output_box), [], [vis_image_input, vis_audio_input, vis_output, vis_output_box])
        clear_lyric_btn.click(lambda: {**clear_ui(lyric_audio, lyric_bg, lyric_output, lyric_output_box), **{lyric_text: ""}}, [], [lyric_audio, lyric_bg, lyric_output, lyric_output_box, lyric_text])

        load_transcript_btn.click(lambda audio, lang: _transcribe_audio_logic(audio, lang), [lyric_audio, lyric_language], [lyric_text])

    app.queue(max_size=3).launch(inbrowser=True)

if __name__ == "__main__":

    main()
