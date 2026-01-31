import numpy as np
import librosa
import json
import os
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema

import matplotlib.pyplot as plt 




def extract_audio_beat(audio, sr):
    """
    borrow from https://github.com/lisiyao21/Bailando/blob/main/_prepro_aistpp_music.py and https://github.com/lisiyao21/Bailando/blob/main/extractor.py
    """

    # get_hpss
    audio_harmonic, audio_percussive = librosa.effects.hpss(audio)

    # get_onset_strength
    onset_env = librosa.onset.onset_strength(y=audio_percussive, aggregate=np.median, sr=sr)

    # get_onset_beat
    onset_tempo, onset_beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    beats_one_hot = np.zeros(len(onset_env))
    for idx in onset_beats:
        beats_one_hot[idx] = 1
    beats_one_hot = beats_one_hot.reshape(1, -1)

    return beats_one_hot



def get_mb(audio, sr, length=None):

    # extract beat feature
    onset_beat = extract_audio_beat(audio, sr)[0]

    if length is not None:
        beats = np.array(onset_beat)[:][:length]
    else:
        beats = np.array(onset_beat)

    beats = beats.astype(bool)
    beat_axis = np.arange(len(beats))
    beat_axis = beat_axis[beats]

    return beat_axis



def calc_db(motion_seq):
    """
    motion: (nframe, 68, 3) or (nframe, 6)
    """
    seq = np.array(motion_seq)
    velocity = np.mean(np.sqrt(np.sum((seq[1:] - seq[:-1]) ** 2, axis=2)), axis=1)
    velocity = gaussian_filter(velocity, 5)
    motion_beats = argrelextrema(velocity, np.less)
    
    return motion_beats, len(velocity)



def BA(music_beats, motion_beats):

    if len(music_beats) == 0:
        music_beats = np.array([0], dtype=np.int64)

    ba = 0
    for bb in music_beats:
        ba +=  np.exp(-np.min((motion_beats[0] - bb)**2) / 2 / 9)
    return (ba / len(music_beats))



def calc_ba_score(motion, audio, sr=16000):
    """
    motion: (nframe, 68, 3) or (nframe, 6)
    audio: (n,)
    sr: sample rate
    """

    # to cpu
    motion = motion.detach().cpu().numpy()
    audio = audio.detach().cpu().numpy()

    motion_beats, length = calc_db(motion)        
    audio_beats = get_mb(audio, sr, length=length)

    ba_score = BA(audio_beats, motion_beats)
    return ba_score



def calc_ba_score_batch(motion, audio, sr=16000):
    """
    audio: (bs, n)
    motion: (bs, nframe, 68, 3) or (bs, nframe, 6)
    """

    if len(motion.shape) != 4:
        motion = motion.unsqueeze(-2)

    ba_score_list = []
    for i in range(motion.shape[0]):
        ba_score = calc_ba_score(motion[i], audio[i], sr=sr)
        ba_score_list.append(ba_score)

    return np.mean(ba_score_list)

