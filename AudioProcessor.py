
# coding: utf-8

# In[28]:


import librosa 
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

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
        src = src[(n_sample-n_sample_fit)/2:(n_sample+n_sample_fit)/2]
    
    
    melgram = librosa.feature.melspectrogram(y=src, sr=SR, hop_length=HOP_LEN, 
                                   n_fft=N_FFT, n_mels=N_MELS)
    
    ret = librosa.power_to_db(melgram, ref=np.max)
    ret = ret[np.newaxis, np.newaxis, :]
    return ret
    
#     print ret.shape
#     plt.figure(figsize=(10,4))
#     librosa.display.specshow(librosa.power_to_db(melgram, ref=np.max),
#                              y_axis='mel', fmax=8000,
#                              x_axis='time')
#     plt.colorbar(format='%+2.0f dB')
#     plt.title('Mel Spectrogram')
#     plt.tight_layout()
#     plt.show()

    
    
compute_melgram(librosa.util.example_audio_file())
    


# In[20]:




