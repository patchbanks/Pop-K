from contextlib import nullcontext
from torch.nn import functional as F
from src.utils import TOKENIZER, Dataset
from pedalboard import Pedalboard, Reverb, Compressor, Gain, Limiter
from pedalboard.io import AudioFile
import pandas as pd
import subprocess
import pretty_midi
import gradio as gr
import time
import copy
import types
import torch
import random
import os

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

in_space = os.getenv("SYSTEM") == "spaces"

n_layer = 12
n_embd = 768
ctx_len = 2048

os.environ['RWKV_FLOAT_MODE'] = 'fp32'
os.environ['RWKV_RUN_DEVICE'] = 'cuda' if torch.cuda.is_available() else 'cpu'
model_type = 'RWKV'

MODEL_NAME = 'checkpoints/model'
LENGTH_PER_TRIAL = round((2000) / 13) * 13
TEMPERATURE = 1.0

from src.model_run import RWKV_RNN
model = RWKV_RNN(MODEL_NAME, os.environ['RWKV_RUN_DEVICE'], model_type, n_layer, n_embd, ctx_len)
tokenizer = TOKENIZER()

temp_dir = 'temp'
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

def clear_midi(dir):
    for file in os.listdir(dir):
        if file.endswith('.mid'):
            os.remove(os.path.join(dir, file))

clear_midi(temp_dir)


ctx_seed = "000000000000\n"
ctx = tokenizer.encode(ctx_seed)
src_len = len(ctx)
src_ctx = ctx.copy()


def generate_midi(LENGTH_PER_TRIAL, src_ctx, model, src_len, ctx_len, TEMPERATURE, top_k, tokenizer, ctx_seed, bpm):
    midi_seq = []
    
    for TRIAL in range(1):
        t_begin = time.time_ns()

        if TRIAL > 0:
            midi_seq.append("\n")

        ctx = src_ctx.copy()
        model.clear()
        midi_tokens = []

        if TRIAL == 0:
            init_state = types.SimpleNamespace()
            for i in range(src_len):
                x = ctx[:i+1]
                if i == src_len - 1:
                    init_state.out = model.run(x)
                else:
                    model.run(x)
            model.save(init_state)
        else:
            model.load(init_state)

        midi_seq.append(ctx_seed)
        
        for i in range(src_len, src_len + LENGTH_PER_TRIAL):
            x = ctx[:i+1]
            x = x[-ctx_len:]

            if i == src_len:
                out = copy.deepcopy(init_state.out)
            else:
                out = model.run(x)

            char = tokenizer.sample_logits(out, x, ctx_len, temperature=TEMPERATURE, top_k=top_k).item()
            midi_tokens.append(char)

            if len(midi_tokens) > 2:
                midi_tokens.pop(0)

            if midi_tokens == [11, 10]:  # stop token pattern
                break
            
            midi_seq.append(tokenizer.decode([int(char)]))
            
            if midi_tokens != [11, 10]:
                ctx += [char]
        
        t_end = time.time_ns()

    trim_seq = "".join(midi_seq)
    events = trim_seq.split("\n")
    
    midi_events = []
    sequence = []
    rndm_num = 895645

    for event in events:
        if event.strip() == "":
            midi_events.append(sequence)
            sequence = []
            rndm_num = random.randint(100000, 999999)
        try:
            pitch = int(event[0:2])
            velocity = int(event[2:4])
            start = int(event[4:8])
            end = int(event[8:12])
        except ValueError:
            pitch = 0
            velocity = 0
            start = 0
            end = 0

        sequence.append({'file_name': f'rwkv_{rndm_num}', 'pitch': pitch, 'velocity': velocity, 'start': start, 'end': end})

    if sequence:
        midi_events.append(sequence)

    midi_events = pd.DataFrame([pd.Series(event) for sequence in midi_events for event in sequence])
    midi_events = midi_events[['file_name', 'pitch', 'velocity', 'start', 'end']]
    midi_events = midi_events.sort_values(by=['file_name', 'start']).reset_index(drop=True)
    midi_events = midi_events[(midi_events['start'] < 3072) & (midi_events['end'] <= 3072)]

    for file_name, events in midi_events.groupby('file_name'):
        midi_obj = pretty_midi.PrettyMIDI(initial_tempo=bpm, resolution=96)
        instrument = pretty_midi.Instrument(0)
        midi_obj.instruments.append(instrument)

        for _, event in events.iterrows():
            note = pretty_midi.Note(
                pitch=event['pitch'], 
                velocity=event['velocity'], 
                start=midi_obj.tick_to_time(event['start']), 
                end=midi_obj.tick_to_time(event['end'])
            )
            instrument.notes.append(note)

        midi_path = os.path.join(temp_dir, 'output.mid')
        midi_obj.write(midi_path)

    return midi_path


def render_wav(midi_file, uploaded_sf2=None, output_level='2.0'):
    sf2_dir = 'sf2'
    audio_format = 's16'
    sample_rate = '44100'
    gain = str(output_level)

    if uploaded_sf2:
        sf2_file = uploaded_sf2
    else:
        sf2_files = [f for f in os.listdir(os.path.join(sf2_dir)) if f.endswith('.sf2')]
        if not sf2_files:
            raise ValueError("No SoundFont (.sf2) file found in directory.")
        sf2_file = os.path.join(sf2_dir, random.choice(sf2_files))

    #print(f"Using SoundFont: {sf2_file}")
    output_wav = os.path.join(temp_dir, 'output.wav')

    with open(os.devnull, 'w') as devnull:
        command = [
            'fluidsynth', '-ni', sf2_file, midi_file, '-F', output_wav, '-r', str(sample_rate), 
            '-o', f'audio.file.format={audio_format}', '-g', str(gain)
        ]
        subprocess.call(command, stdout=devnull, stderr=devnull)

    return output_wav


def generate_and_return_files(bpm, temperature, top_k, uploaded_sf2=None, output_level='2.0'):
    midi_events = generate_midi(
        LENGTH_PER_TRIAL, src_ctx, model, src_len, ctx_len, temperature, top_k, 
        tokenizer, ctx_seed, bpm
    )  
    
    midi_file = 'temp/output.mid'
    wav_raw = render_wav(midi_file, uploaded_sf2, output_level)
    wav_fx = os.path.join(temp_dir, 'output_fx.wav')

    sfx_settings = [
        {
            'board': Pedalboard([
                Reverb(room_size=0.50, wet_level=0.30, dry_level=0.75, width=1.0),
                Compressor(threshold_db=-4.0, ratio=4.0, attack_ms=0.0, release_ms=300.0),
            ])
        }
    ]

    for setting in sfx_settings:
        board = setting['board']

        with AudioFile(wav_raw) as f:
            with AudioFile(wav_fx, 'w', f.samplerate, f.num_channels) as o:
                while f.tell() < f.frames:
                    chunk = f.read(int(f.samplerate))
                    effected = board(chunk, f.samplerate, reset=False)
                    o.write(effected)

    return midi_file, wav_fx


custom_css = """
#generate-btn {
    background-color: #6366f1 !important;
    color: white !important;
    border: none !important;
    font-size: 16px;
    padding: 10px 20px;
    border-radius: 5px;
    cursor: pointer;
}
#generate-btn:hover {
    background-color: #4f51c5 !important;
}
"""

with gr.Blocks(css=custom_css, theme="soft") as iface:
    gr.Markdown("<h1 style='font-weight: bold; text-align: center;'>Pop-K</h1>")
    gr.Markdown("<p style='text-align:center;'>Pop-K is a small RWKV model that generates pop melodies in C major and A minor.</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            bpm = gr.Slider(minimum=50, maximum=200, step=1, value=120, label="BPM")
            temperature = gr.Slider(minimum=0.1, maximum=2.0, step=0.01, value=1.0, label="Temperature")
            top_k = gr.Slider(minimum=1, maximum=32, step=1, value=20, label="Top-K")
            output_level = gr.Slider(minimum=0, maximum=3, step=0.10, value=2.0, label="Output Level")
            soundfont = gr.File(label="Optional: Upload SoundFont (preset=0, bank=0, max_size=100mb)")
        
        with gr.Column(scale=1):
            midi_file = gr.File(label="MIDI File Output")
            audio_file = gr.Audio(label="Generated Audio Output", type="filepath")
            generate_button = gr.Button("Generate", elem_id="generate-btn")
    
    generate_button.click(
        fn=generate_and_return_files,
        inputs=[bpm, temperature, top_k, soundfont, output_level],
        outputs=[midi_file, audio_file]
    )

iface.launch(share=True)
