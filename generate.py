from torch.nn import functional as F
from src.utils import TOKENIZER
import pandas as pd
import pretty_midi
import argparse
import random
import time
import types
import copy
import torch
import os


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

parser = argparse.ArgumentParser(description="Script description")

parser.add_argument("--num_samples", type=int, default=10, help="Number of samples (default: 10)")
parser.add_argument("--temperature", type=float, default=1.0, help="temperature (default: 1.0)")
parser.add_argument("--top_k", type=float, default=16, help="Top-K (default: 16)")
parser.add_argument("--model_name", type=str, default='ckpt', help="Model name (default: ckpt)")

args = parser.parse_args()


MODEL_NAME = args.model_name
num_samples = args.num_samples
max_new_tokens = round((2000) / 13) * 13
temperature = args.temperature
top_k = args.top_k

n_layer = 12
n_embd = 768
ctx_len = 2048


os.environ['RWKV_FLOAT_MODE'] = 'fp32'
os.environ['RWKV_RUN_DEVICE'] = 'cuda' if torch.cuda.is_available() else 'cpu'
model_type = 'RWKV'

from src.model_run import RWKV_RNN
model = RWKV_RNN(MODEL_NAME, os.environ['RWKV_RUN_DEVICE'], model_type, n_layer, n_embd, ctx_len)
tokenizer = TOKENIZER()


midi_dir = 'midi_output'
if not os.path.exists(midi_dir):
    os.makedirs(midi_dir, exist_ok=True)

temp_dir = 'temp'
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir, exist_ok=True)


def clear_midi(dir):
    for file in os.listdir(dir):
        if file.endswith('.mid'):
            os.remove(os.path.join(dir, file))

clear_midi(midi_dir)


ctx_seed = f"000000000000\n" # start tokens
ctx = tokenizer.encode(ctx_seed)
src_len = len(ctx)
src_ctx = ctx.copy()

with open('temp/output.txt', 'w') as output_file:
    for sample in range(num_samples):
        t_begin = time.time_ns()

        if sample > 0:
            output_file.write("\n")

        ctx = src_ctx.copy()
        model.clear()

        midi_tokens = []

        if sample == 0:
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

        output_file.write(ctx_seed)

        for i in range(src_len, src_len + max_new_tokens):
            x = ctx[:i+1]
            x = x[-ctx_len:]

            if i == src_len:
                out = copy.deepcopy(init_state.out)
            else:
                out = model.run(x)

            char = tokenizer.sample_logits(out, x, ctx_len, temperature=temperature, top_k=top_k)
            char = char.item()

            midi_tokens.append(char)

            if len(midi_tokens) > 2:
                midi_tokens.pop(0)

            if midi_tokens == [11] + [10]: # start token pattern
                break

            token_output = tokenizer.decode([int(char)])
            output_file.write(token_output)

            if midi_tokens != [11] + [10]:
                ctx += [char]

        t_end = time.time_ns()
        print(f"sample {sample + 1}/{num_samples} {round((t_end - t_begin) / (10 ** 9), 2)}s")


with open('temp/output.txt', 'r') as file:
    import_midi = file.read()

events_data = import_midi.split('\n')
events = '\n'.join(events_data)

midi_events = []
sequence = []
rndm_num = 0

for event in events.split('\n'):
    if event.strip() == "000000000000":
        midi_events.append(sequence)
        sequence = []
        rndm_num = random.randint(100000, 999999)
    try:
        pitch = int(event[0:2])
        velocity = int(event[2:4])
        start_tick = int(event[4:8])
        end_tick = int(event[8:12])
    except ValueError:
        pitch = 0
        velocity = 0
        start_tick = 0
        end_tick = 0

    sequence.append({'file_name': f'pop-k_{rndm_num}', 'pitch': pitch, 'velocity': velocity, 'start_tick': start_tick, 'end_tick': end_tick})

if sequence:
    midi_events.append(sequence)

midi_events = pd.DataFrame([pd.Series(event) for sequence in midi_events for event in sequence])
midi_events = midi_events[['file_name', 'pitch', 'velocity', 'start_tick', 'end_tick']]


def trim_sequences(df, tick_max=3072):
    df = df[df['start_tick'] < tick_max]
    df['end_tick'] = df['end_tick'].clip(upper=tick_max)
    df = df[~((df['pitch'] == 0) & (df['velocity'] == 0) & (df['start_tick'] == 0) & (df['end_tick'] == 0))]
    return df

midi_events = trim_sequences(midi_events)


def write_midi(midi_events):
    midi_events_by_file = {}

    for index, event in midi_events.iterrows():
        file_name = event['file_name']
        if file_name not in midi_events_by_file:
            midi_events_by_file[file_name] = []
        midi_events_by_file[file_name].append(event)

    for file_name, events in midi_events_by_file.items():
        midi_events = pretty_midi.PrettyMIDI(initial_tempo=120, resolution=96)
        midi_events.time_signature_changes.append(pretty_midi.containers.TimeSignature(4, 4, 0))
        instrument = pretty_midi.Instrument(0)
        midi_events.instruments.append(instrument)
        for event in events:
            pitch = event['pitch']
            velocity = event['velocity']
            start = midi_events.tick_to_time(event['start_tick'])
            end = midi_events.tick_to_time(event['end_tick'])
            note = pretty_midi.Note(pitch=pitch, velocity=velocity, start=start, end=end)
            instrument.notes.append(note)
        midi_path = os.path.join(midi_dir, file_name + '.mid')
        midi_events.write(midi_path)

write_midi(midi_events)