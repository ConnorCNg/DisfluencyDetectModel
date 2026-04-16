#!/usr/bin/env python3

import argparse
import csv
import os
import random

import librosa as lr
import matplotlib.pyplot as pp
import numpy as np
import scipy as sp


def predict(path, plot=False):
    sr = 16000
    y, sr = lr.load(path=path, sr=sr)

    hop_length = int(0.010 * sr)
    rms = lr.feature.rms(y=y, hop_length=hop_length)
    rms_db = lr.amplitude_to_db(S=rms, ref=2e-5)
    times = lr.times_like(X=rms, sr=sr, hop_length=hop_length)

    n_fft = int(0.025 * sr)
    n_mels = 41
    n_mfcc = 13

    M = lr.feature.mfcc(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        n_mfcc=n_mfcc,
    )

    features = M
    dissimilarity = 100 * lr.util.normalize(
        [
            sp.spatial.distance.cosine(features[:, i], features[:, i + 1])
            for i in range(features.shape[1] - 1)
        ]
        + [0]
    )

    prolongations = []
    i = 0

    for j in range(len(times)):
        rms_limit = 55
        dissimilarity_limit = 10

        if rms_db[0, j] > rms_limit and dissimilarity[j] < dissimilarity_limit:
            continue

        duration_limit = 0.25
        if times[j] - times[i] > duration_limit:
            prolongations.append([i, j])

        i = j

    if plot:
        pp.clf()
        pp.title(path)
        pp.plot(times, rms_db[0])
        pp.plot(times, dissimilarity, color="orange")

        for i0, j0 in prolongations:
            pp.axvspan(times[i0], times[j0], alpha=0.5, color="red")
            pp.hlines(
                np.median(rms_db[0, i0:j0]),
                xmin=times[i0],
                xmax=times[j0],
                color="blue",
            )

        pp.show()

    return len(prolongations) > 0


def main():
    ap = argparse.ArgumentParser(
        description="Template-style prolongation-only evaluation (0 vs 3 votes)."
    )
    ap.add_argument("--csv", default="SEP-28k-Extended_clips.csv")
    ap.add_argument("--data-root", default="data/sep28k/clips")
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle negatives before balancing to positives.",
    )
    ap.add_argument(
        "--plot-index",
        type=int,
        default=-1,
        help="If >=0, plot that test item while evaluating.",
    )
    args = ap.parse_args()

    positives = []
    negatives = []

    with open(args.csv, "r", newline="") as file:
        for row in csv.DictReader(file):
            show = row["Show"]
            episode = row["EpId"]
            clip = row["ClipId"]
            path = os.path.join(
                args.data_root,
                show,
                episode,
                f"{show}_{episode}_{clip}.wav",
            )
            if not os.path.exists(path):
                continue

            v = int(row["Prolongation"])
            if v == 0:
                negatives.append(path)
            elif v == 3:
                positives.append(path)

    rng = random.Random(args.seed)
    rng.shuffle(negatives)
    tests = positives + negatives[: len(positives)]

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    if not tests:
        raise RuntimeError("No test clips found after CSV/path filtering.")

    for i, test in enumerate(tests):
        print(f"{100 * i / len(tests):.0f}%", flush=True)
        do_plot = args.plot_index >= 0 and i == args.plot_index
        prediction = predict(test, plot=do_plot)

        if test in positives:
            if prediction:
                tp += 1
            else:
                fn += 1
        else:
            if prediction:
                fp += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    print(f"counts: tp={tp} tn={tn} fp={fp} fn={fn}", flush=True)
    print(f"precision = {100 * precision:.2f}%", flush=True)
    print(f"recall = {100 * recall:.2f}%", flush=True)
    print(f"f1 = {100 * f1:.2f}%", flush=True)
    print(f"accuracy = {100 * accuracy:.2f}%", flush=True)


if __name__ == "__main__":
    main()
