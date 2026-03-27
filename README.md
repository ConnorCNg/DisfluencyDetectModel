# DisfluencyDetectModel

# Downloading & Processing Scripts


There are two scripts used to download the raw audio files and extract into clips that correspond to the clip annotations. `[WAV_DIR]` refers to the folder where you are storing all of the raw audio data and `[CLIP_DIR]` refers to where you want to place the clips. These may be the same folder. 

To download and extract clips from both datasets run the following from this directory

* `python download_audio.py --episodes SEP-28k_episodes.csv --wavs [WAV_DIR]`
* `python extract_clips.py --labels SEP-28k_labels.csv --wavs [DATA_DIR] --clips [CLIP_DIR]`
* `python download_audio.py --episodes fluencybank_episodes.csv --wavs [WAV_DIR]`
* `python extract_clips.py --labels fluencybank_labels.csv --wavs [DATA_DIR] --clips [CLIP_DIR]`

The raw SEP-28k wav files are 32 Gb and clipped SEP-28k wav files are 2.6 Gb. 

You should be able to use the individual csv files for T/D/E seperately to extract them into their own DIR
