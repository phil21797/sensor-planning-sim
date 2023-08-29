"""
Load sound files (.wav) from a folder, modify the duration and amplitude
(optional), and play them. A fixed number of samples are played back from the
middle of each signal.

This program uses the system command "mpv" to play the audio files. If you don't
have this audio player on your system, you will need to change that line of the
code to use whatever audio player you do have on your system.

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

if __name__ == '__main__':

    rescale_resample = False       # should apply random scale and resampling?
    playtime = 4                   # how long to play each sound, in seconds
    audio_max = 2**14              # used to normalize audio signals

    # Get a list of audio files.
    folder = './sounds/'
    files = glob.glob(folder + '*.wav')
    files.sort()
    numfiles = len(files)
    print(f'Found {numfiles} audio files in {folder}')

    plt.ion()                                # enable interactive mode

    for fname in files:
        fparts = os.path.split(fname)
        basefname = fparts[1]
        print(f'\n{basefname}:')

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

        if rescale_resample:
            # Rescale the amplitude and resample the signal.
            ascale = np.random.rand() + 0.5       # amplitude scale in [0.5,1.5]
            tscale = np.random.rand() + 0.5
            print(f'    Ascale = {ascale:.2f}, Tscale = {tscale:.2f}')
            signal = ascale*signal
            xp = np.arange(0, nsamples)
            nsamples_new = int(np.ceil(tscale*nsamples))
            print(f'    nsamples_new = {nsamples_new}')
            x = np.linspace(0, nsamples-1, num=nsamples_new)
            signal = np.interp(x, xp, signal)                  # resample signal
            nsamples = nsamples_new
        else:
            ascale = tscale = 1.0

        # Extract `playsamples` in the middle of the signal to playback.
        playnsamples = 20000000               # number of samples to play & save
        playnsamples = min(nsamples, playnsamples)
        start = int((nsamples - playnsamples)/2)
        signal = signal.astype(np.int16)
        signal = signal[start:start+playnsamples]

        # Display the audio signal.
        plt.figure(figsize=(12,4))
        plt.title(f'{fparts[1]}: AmpScale={ascale:.2f}, TimeScale={tscale:.2f}')
        plt.plot(signal, 'b-')
        plt.xlim(0, playnsamples)
        plt.ylim(-1.5*audio_max, 1.5*audio_max)
        plt.draw()
        plt.pause(0.1)

        # Save the audio signal and then play the file using a system command.
        wavfile.write('tmp_sound.wav', samplerate, signal)
        subprocess.run(['mpv', 'tmp_sound.wav', '--length='+str(playtime)],
                       capture_output=True)
        subprocess.run(['rm', 'tmp_sound.wav'])

        plt.close()

