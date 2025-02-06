# ====================================================================
# This script expects preprocessed MIDI files trimmed to 8-bar lengths
# and converted to C major and A minor. Only train on melodies that
# have similar or less complexity than the examples in data/midi_data.
# For best results, the model will require a few hundred thousand samples
# of a similar musical style. Training with mixed musical styles has not
# been tested.
#
# Usage:
# !python data/midi_to_text.py \
#     --midi_dir='data/midi_data' \
#     --dataset_name='pop_k_test'
#
# *** Use --test_mode and test_sample_size=10 for testing ***
# ====================================================================


from tqdm import tqdm
import pretty_midi
import pandas as pd
import argparse
import random
import mido
import glob
import os


parser = argparse.ArgumentParser()

parser.add_argument('--midi_dir', type=str, default='midi_data', help='directory containing midi files')
parser.add_argument('--dataset_name', type=str, default='pop_k', help='train data file name')
parser.add_argument('--test_mode', action='store_true', help='enables test mode')
parser.add_argument('--test_sample_size', type=int, default=10, help='number of midi files to sample')

args = parser.parse_args()

midi_dir = args.midi_dir
dataset_name = args.dataset_name
test_mode = args.test_mode
test_sample_size = args.test_sample_size


def traverse_dataset(midi_dir, test_mode=False, test_sample_size=10):
    print(f"compiling midi files...")
    midi_list = []

    for root, dirs, files in os.walk(midi_dir):
        for file in files:
            if file.lower().endswith(('.mid', '.midi', '.MID')):
                midi_list.append(os.path.join(root, file))

    if not midi_list:
        print("  no files in midi directory.")
        return []

    if test_mode:
        if len(midi_list) < test_sample_size:
            print("  test sample size exceeds total number of midi files. defaulting to 1")
            test_sample_size = 1
        midi_list = random.sample(midi_list, test_sample_size)
        print(f"  test mode - midi sample size: {test_sample_size}")

    return midi_list

midi_list = traverse_dataset(midi_dir, test_mode=test_mode, test_sample_size=test_sample_size)
total_files = len(midi_list)


midi_events = []
count = 0

delete_on_error = False  # set to True to delete files with errors

with tqdm(total=total_files, desc="  processing", unit="file", unit_scale=True, dynamic_ncols=True) as pbar:
    for midi_file in midi_list:
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_file)
        except (ValueError, EOFError, IOError, IndexError, mido.midifiles.meta.KeySignatureError) as e:
            if delete_on_error:
                os.remove(midi_file)
                print(f"  error: deleted {midi_file}")
            continue
        
        if len(midi_data.instruments) == 0 or len(midi_data.instruments[0].notes) < 2:
            os.remove(midi_file)
            print(f"  removed {midi_file} due to low note density")
            continue
        
        try:
            tempo_changes = midi_data.get_tempo_changes()
            if any(t == 0 for t in tempo_changes[1]):
                print(f"  skipping {midi_file} due to zero tempo")
                continue
            tempo = tempo_changes[1][0] if any(tempo_changes[1]) else 120.0
        except UserWarning as e:
            print(f"  warning in {midi_file}: {e}")
            tempo = 120.0

        file_name = os.path.splitext(os.path.basename(midi_file))[0]
        rndm_num = random.randint(1000000, 9999990)
        rndm_file_name = f"{rndm_num}_{file_name}"
        tick_resolution = midi_data.resolution

        for i, instrument in enumerate(midi_data.instruments):
            for note in instrument.notes:
                pitch = note.pitch
                velocity = note.velocity
                start = note.start
                end = note.end
                start_tick = midi_data.time_to_tick(start)
                end_tick = midi_data.time_to_tick(end)

                midi_events.append([rndm_file_name, pitch, velocity, start_tick, end_tick, tick_resolution, tempo])

        pbar.update(1)


midi_columns = ['file_name', 'pitch', 'velocity', 'start_tick', 'end_tick', 'tick_resolution', 'tempo']
midi_events = pd.DataFrame(midi_events, columns=midi_columns)


# limit to two digits
midi_events = midi_events[~((midi_events['pitch'] > 99) | (midi_events['pitch'] < 20))]
midi_events.loc[midi_events['velocity'] > 99, 'velocity'] = 99


# normalize tempo and tick resolution
def process_tick_timing(midi_events, tick_base=96, master_tempo=120):
    print(f"converting timings...")
    midi_events['start_tick'] = (midi_events['start_tick'] * tick_base / midi_events['tick_resolution'])
    midi_events['end_tick'] = (midi_events['end_tick'] * tick_base / midi_events['tick_resolution'])
    midi_events['tick_resolution'] = tick_base
    midi_events['tempo'] = master_tempo
    return midi_events

midi_events = process_tick_timing(midi_events)
midi_events = midi_events.sort_values(by=['file_name', 'start_tick'], ignore_index=True)


# trim to 8 bars
tick_max = 8 * 384 # 8 bars * 384 ticks
trim_bars = midi_events[midi_events['start_tick'] < tick_max].copy()
trim_bars.loc[:, 'end_tick'] = trim_bars['end_tick'].clip(upper=tick_max)
midi_events = trim_bars


# format events to text
def popk_format(df): 
    output_lines = []
    if not df.empty:
        last_file_name = ''
        for row in df.itertuples(index=False):
            if row.file_name != last_file_name:
                last_file_name = row.file_name
                output_lines.append('000000000000')

            pitch = int(row.pitch)
            velocity = int(row.velocity)
            start_tick = int(row.start_tick)
            end_tick = int(row.end_tick)
            output_lines.append(f"{pitch:02d}{velocity:02d}{start_tick:04d}{end_tick:04d}")

    total_characters = sum(len(line) for line in output_lines)
    return '\n'.join(output_lines) + '\n', total_characters


def evemts_to_text(df):
    global text_dataset_path
    output_text, _ = popk_format(df)
    text_dataset_path = f"data/train_data/{dataset_name}.txt"
    os.makedirs(os.path.dirname(text_dataset_path), exist_ok=True)
    with open(text_dataset_path, 'w') as new_f:
        new_f.write(output_text)

evemts_to_text(midi_events)


print(f"getting stats...")
def convert_to_serializable(value):
    if isinstance(value, pd.Timestamp):
        return value.strftime('%Y-%m-%d %H:%M:%S')
    elif pd.api.types.is_integer(value):
        return int(value)
    elif pd.api.types.is_float(value):
        return float(value)
    else:
        return str(value)


def print_min_max(dataframe):
    table_data = {}
    print(f"\n***** dataset summary *****")
    print(f"total files: {len(dataframe['file_name'].unique())}")
    print(f"total midi events: {len(dataframe)}")
    print("***** min-max *****")

    for column in dataframe.columns:
        if column != 'file_name':
            min_value = convert_to_serializable(dataframe[column].min())
            max_value = convert_to_serializable(dataframe[column].max())
            print(f"{column}: {min_value} - {max_value}")

print_min_max(midi_events)