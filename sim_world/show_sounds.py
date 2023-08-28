"""
Load sound files (.wav) from a folder, modify the duration and amplitude,
and play them.

Author:
    Phil David, Parsons, June 2023.
"""


import numpy as np
import glob
import wavio
from scipy.io import wavfile
import matplotlib.pyplot as plt
import subprocess
import os

playtime = 4                      # how long to play each sound, in seconds
audio_max = 2**14                 # used to normalize audio signals

folder = './sim_world/sounds/'
# files = glob.glob(folder + 'car*.wav') + glob.glob(folder + 'truck*.wav')
files = glob.glob(folder + 'bird_01.wav') + \
        glob.glob(folder + 'bird_05.wav') + \
        glob.glob(folder + 'person_talking_01.wav') + \
        glob.glob(folder + 'robot_02.wav')

files.sort()

plt.ion()

for fname in files:
    for k in range(1):
        fparts = os.path.split(fname)
        basefname = fparts[1]
        print(f'\nv{k} - {basefname}:')

        wav = wavio.read(fname)

        signal = wav.data.squeeze()
        nsamples = signal.shape[0]           # number of samples
        nchannels = signal.ndim              # number of channels (mono or stereo?)
        samplerate = wav.rate                # samples/sec (Hz)
        samplebits = 8*wav.sampwidth         # bits/sample
        amp_max = 2**samplebits              # max amplitude
        duration = nsamples/samplerate       # duration of signal (seconds)
        t_max = 2*nsamples                   # max x-axis value in plots

        signal = signal.astype(float)
        if nchannels > 1:
            signal = signal[:,0]             # plot only one channel

        print(f'    nsamples = {nsamples:,d}')
        print(f'    nchannels = {nchannels}')
        print(f'    duration = {duration:.2f} sec.')
        print(f'    samplerate = {samplerate} Hz')
        print(f'    maxamp = {amp_max}')

        # Normalize the signal.
        mag = max(abs(signal.min()), abs(signal.max()))
        signal = audio_max*(signal/mag)

        if k == 0:
            ascale = tscale = 1.0
        else:
            ascale = np.random.rand() + 0.5       # amplitude scale in [0.5,1.5]
            tscale = np.random.rand() + 0.5
            print(f'    Ascale = {ascale:.2f}, Tscale = {tscale:.2f}')
            signal = ascale*signal
            xp = np.arange(0, nsamples)
            nsamples_new = int(np.ceil(tscale*nsamples))
            print(f'    nsamples_new = {nsamples_new}')
            x = np.linspace(0, nsamples-1, num=nsamples_new)
            signal = np.interp(x, xp, signal)
            nsamples = nsamples_new

        playnsamples = 20000000               # number of samples to play & save
        playnsamples = min(nsamples, playnsamples)
        start = int((nsamples - playnsamples)/2)
        signal = signal.astype(np.int16)
        signal = signal[start:start+playnsamples]

        plt.figure(figsize=(12,4))
        plt.title(f'v{k} - {fparts[1]}: A={ascale:.2f}, T={tscale:.2f}')
        plt.plot(signal, 'b-')
        plt.xlim(0, playnsamples)
        plt.ylim(-1.5*audio_max, 1.5*audio_max)
        plt.draw()
        plt.pause(0.1)

        # wavfile.write(fname, samplerate, signal)
        wavfile.write('tmp_sound.wav', samplerate, signal)
        subprocess.run(['mpv', 'tmp_sound.wav', '--length='+str(playtime)],
                       capture_output=True)
        subprocess.run(['rm', 'tmp_sound.wav'])

        plt.close()

