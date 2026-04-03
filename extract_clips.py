#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#

"""
For each podcast episode:
* Get all clip information for that episode
* Save each clip as a new wav file.
"""

import os
import pathlib
import subprocess

import numpy as np
import pandas as pd
from scipy.io import wavfile

import argparse

parser = argparse.ArgumentParser(description='Extract clips from SEP-28k or FluencyBank.')
parser.add_argument('--labels', type=str, default='SEP-28k-Extended_clips.csv',
                   help='Clip CSV with Start_5_sec / Stop_5_sec (default: SEP-28k-Extended_clips.csv)')
parser.add_argument('--wavs', type=str, default='data/sep28k/wavs',
                   help='Directory of full-episode .wav files from download_audio.py')
parser.add_argument('--clips', type=str, default='data/sep28k/clips',
                   help='Output directory for 5 s clip .wav files (default: data/sep28k/clips)')
parser.add_argument("--progress", action="store_true",
                    help="Show progress")

args = parser.parse_args()
label_file = args.labels
data_dir = args.wavs
output_dir = args.clips


# Load label/clip file (SEP-28k Extended uses the same Show / EpId / ClipId / Start_5_sec / Stop_5_sec layout)
data = pd.read_csv(label_file, dtype={"EpId": str, "ClipId": str})

# Get label columns from data file
shows = data.Show
episodes = data.EpId
clip_idxs = data.ClipId
starts = data.Start_5_sec
stops = data.Stop_5_sec
labels = data.iloc[:,5:].values

n_items = len(shows)

loaded_wav = ""
cur_iter = range(n_items)
if args.progress:
        from tqdm import tqdm
        cur_iter = tqdm(cur_iter)

for i in cur_iter:
	clip_idx = clip_idxs[i]
	show_abrev = shows[i]
	episode = episodes[i].strip()

	# Setup paths
	wav_path = f"{data_dir}/{shows[i]}/{episode}.wav"
	clip_dir = pathlib.Path(f"{output_dir}/{show_abrev}/{episode}/")
	clip_path = f"{clip_dir}/{shows[i]}_{episode}_{clip_idx}.wav"

	if not os.path.exists(wav_path):
		print("Missing", wav_path)
		continue

	# Verify clip directory exists
	os.makedirs(clip_dir, exist_ok=True)

	# Load audio. For efficiency reasons don't reload if we've already open the file.
	if wav_path != loaded_wav:
		sample_rate, audio = wavfile.read(wav_path)
		assert sample_rate == 16000, "Sample rate must be 16 khz"

		# Keep track of the open file
		loaded_wav = wav_path

	# Save clip to file
	clip = audio[starts[i]:stops[i]]
	wavfile.write(clip_path, sample_rate, clip)