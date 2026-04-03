#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#

"""
For each podcast episode:
* Download the raw mp3/m4a file
* Convert it to a 16k mono wav file
# Remove the original file
"""

import os
import pathlib
import shutil
import subprocess

import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Download raw audio files for SEP-28k or FluencyBank and convert to 16k hz mono wavs.')
parser.add_argument('--episodes', type=str, default='SEP-28k_episodes.csv',
                   help='Episode list CSV (comma-space delimited, e.g. SEP-28k_episodes.csv)')
parser.add_argument('--wavs', type=str, default='data/sep28k/wavs',
                   help='Directory where converted 16 kHz mono .wav files are saved (default: data/sep28k/wavs)')


args = parser.parse_args()
episode_uri = args.episodes
wav_dir = args.wavs


def load_episode_table(path: str) -> np.ndarray:
	"""
	SEP-28k_episodes.csv rows are separated by ', ' (comma + space).
	np.loadtxt(..., delimiter=', ') no longer works on NumPy 1.26+/2.x (delimiter must be
	a single character), so we split lines and build a 2D ndarray — same indexing as before.
	"""
	rows: list = []
	with open(path, encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			parts = line.split(", ")
			if len(parts) < 5:
				raise ValueError(
					f"Expected at least 5 fields after ', ' split, got {len(parts)}: {line[:120]}..."
				)
			rows.append(parts)
	return np.asarray(rows, dtype=str)


# Load episode data (2D array: url at [:, 2], show at [:, -2], ep id at [:, -1])
table = load_episode_table(episode_uri)
n_items = len(table)

audio_types = [".mp3", ".m4a", ".mp4"]


for i in range(n_items):
	# Get show/episode IDs
	show_abrev = table[i, -2]
	ep_idx = table[i, -1]
	episode_url = table[i, 2]

	# Check file extension
	ext = ''
	for ext in audio_types:
		if ext in episode_url:
			break

	# Ensure the base folder exists for this episode
	episode_dir = pathlib.Path(f"{wav_dir}/{show_abrev}/")
	os.makedirs(episode_dir, exist_ok=True)

	# Get file paths
	audio_path_orig = pathlib.Path(f"{episode_dir}/{ep_idx}{ext}")
	wav_path = pathlib.Path(f"{episode_dir}/{ep_idx}.wav")

	# Check if this file has already been downloaded
	if os.path.exists(wav_path):
		continue

	# Drop bogus partial downloads (e.g. 0-byte) so we retry
	if audio_path_orig.exists() and audio_path_orig.stat().st_size < 1024:
		audio_path_orig.unlink()

	print("Processing", show_abrev, ep_idx)

	def _download(url: str, dest: pathlib.Path) -> None:
		# curl: -f fail on HTTP errors; -k skip TLS verify (some SEP-28k URLs use certs
		# that fail on stock macOS CA store; public podcast files only).
		if shutil.which("curl"):
			subprocess.run(
				[
					"curl",
					"-fL",
					"-k",
					"-sS",
					"--connect-timeout",
					"30",
					"-o",
					str(dest),
					url,
				],
				check=True,
			)
		elif shutil.which("wget"):
			subprocess.run(
				["wget", "-q", "--timeout=30", "-O", str(dest), url],
				check=True,
			)
		else:
			raise RuntimeError("Need curl or wget in PATH to download audio.")

	if not audio_path_orig.exists():
		try:
			_download(episode_url, audio_path_orig)
		except subprocess.CalledProcessError as e:
			if audio_path_orig.exists():
				audio_path_orig.unlink(missing_ok=True)
			print("SKIP download:", e.returncode, episode_url)
			continue

	if not audio_path_orig.exists():
		print("SKIP (no file after download)", episode_url)
		continue

	if audio_path_orig.stat().st_size < 1024:
		sz = audio_path_orig.stat().st_size
		audio_path_orig.unlink(missing_ok=True)
		print("SKIP (download too small:", sz, "bytes)", episode_url)
		continue

	# Convert to 16khz mono wav file
	try:
		subprocess.run(
			[
				"ffmpeg",
				"-nostdin",
				"-loglevel",
				"error",
				"-y",
				"-i",
				str(audio_path_orig),
				"-ac",
				"1",
				"-ar",
				"16000",
				str(wav_path),
			],
			check=True,
			capture_output=True,
			text=True,
		)
	except subprocess.CalledProcessError as e:
		if audio_path_orig.exists():
			audio_path_orig.unlink()
		print("SKIP ffmpeg:", (e.stderr or e.stdout or e).strip(), "|", episode_url)
		continue

	# Remove the original mp3/m4a file
	os.remove(audio_path_orig)
