#Todo
#create function for getting data
#create function for converting data
#create function for visualizing data
#make better visualizations

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path
import sys
from subprocess import call
from pydub import AudioSegment

def compute_melgram(audio_path):

    #constants for sizing
    SR = 12000
    DURA = 29.12
    #constants for mel spectrogram
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256

    src, sr = librosa.load(audio_path, sr=SR) #load the audio
    n_sample = src.shape[0] #the samples size
    n_sample_fit = int(DURA*SR) #our target size

#     print(n_sample)
#     print(n_sample_fit)

    if n_sample < n_sample_fit: #too short, resize by adding zeros
        src = np.hstack((src, np.zeros((int(DURA*SR) - n_sample,))))

    if n_sample > n_sample_fit: #too long, resize to length=DURA*SR in the middle of sample
        src = src[int((n_sample-n_sample_fit)/2):int((n_sample+n_sample_fit)/2)]


    melgram = librosa.feature.melspectrogram(y=src, sr=SR, hop_length=HOP_LEN,
                                   n_fft=N_FFT, n_mels=N_MELS)

    ret = librosa.power_to_db(melgram, ref=np.max)
    ret = ret[np.newaxis, np.newaxis, :]

    # print(ret.shape)
    # plt.figure(figsize=(10,4))
    # librosa.display.specshow(librosa.power_to_db(melgram, ref=np.max),
    #                          y_axis='mel', fmax=8000,
    #                          x_axis='time')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title(audio_path)
    # plt.tight_layout()
    # plt.show()

    return ret

path = ".\songs"


for filename in os.listdir(path):
    if filename.endswith('.mp3'):
        sound = AudioSegment.from_mp3(os.path.join(path, filename))
        newPath = os.path.normpath(os.path.join(path, filename[:-4]) + '.wav')
        print(newPath)
        sound.export(newPath, format='wav')

for filename in os.listdir(path):
    if filename.endswith('.wav'):
        compute_melgram(os.path.join(path, filename))
