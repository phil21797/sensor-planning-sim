"""
Audio I/O - read, write, and play audio signals.

Curently, writing of audio files only seems to work for float arrays normailized
to [-1,1].

Author:
    Phil David, Parsons, August 2023.
"""



import wavio
from scipy.io import wavfile
import matplotlib.pyplot as plt
import subprocess
import glob
import numpy


def play_audio():
    """
    Play an audio signal.
    """


def read_audio(fname:str, mono: bool = False) -> (numpy.ndarray, int):
    """
    Read an audio signal from a WAV file.

    Usage:
        signal, samplerate = read_audio(fname:str, mono: bool = False)

    Arguments:
        fname: str -- Name of WAV file to read audio signal from.

        mono: bool -- Should returned signal be a single channel? Default is
        False.

    Returns:
        signal: numpy.ndarray -- The audio signal, a float array.

        samplerate: int -- Signal sample rate (samples/sec, Hz).
    """
    wav = wavio.read(fname)
    samplerate = wav.rate
    signal = wav.data.squeeze().astype(float)
    # signal = wav.data.squeeze().astype(numpy.int32)
    if mono and signal.ndim > 1:
        signal = signal[:,0]
    return signal, samplerate


def write_audio(fname:str, signal:numpy.ndarray, samplerate:int):
    """
    Save an audio signal to a WAV file.
    """
    smax = max(abs(signal.min()), abs(signal.max()))
    if smax < 1:
        smax = 1
    wavfile.write(fname, samplerate, signal/smax)
    # wavfile.write(fname, samplerate, signal.astype(numpy.int32))


def play_audio(signal:numpy.ndarray, samplerate:int, playtime:float=None):
    """
    Play an audio signal.
    """
    fname = 'tmp_sound_' + str(numpy.random.randint(999999)) + '.wav'
    lengtharg = '' if playtime == None else '--length='+str(playtime)
    smax = max(abs(signal.min()), abs(signal.max()))
    if smax < 1:
        smax = 1
    wavfile.write(fname, samplerate, signal/smax)
    # wavfile.write(fname, samplerate, signal.astype(numpy.int32))
    subprocess.run(['mpv', fname, lengtharg], capture_output=True)
    subprocess.run(['rm', fname])


if __name__ == '__main__':

    # Get a list of audio files to play.
    folder = '/home/phil/research/visual_search/sim_world_v2/'
    files = glob.glob(folder + '*.wav')
    files.sort()

    for fname in files:
        print(f'\n{fname}:')
        signal, samplerate = read_audio(fname, mono=False)

        nsamples = signal.shape[0]           # number of samples
        nchannels = signal.ndim              # number of channels (mono or stereo?)
        duration = nsamples/samplerate       # duration in seconds
        maxamp = max(abs(signal.min()),      # max amplitude
                     abs(signal.max()))

        print(f'    nsamples = {nsamples:,d}')
        print(f'    nchannels = {nchannels}')
        print(f'    samplerate = {samplerate} Hz')
        print(f'    duration = {duration:.2f} sec.')
        print(f'    maxamp  = {maxamp}')

        if nchannels > 1:
            signal = signal[:,0]             # plot only one channel

        plt.figure(figsize=(12,4))
        plt.title(f'{fname}')
        plt.plot(signal, 'b-')
        plt.ylim(-maxamp, maxamp)
        plt.draw()
        plt.pause(0.1)

        write_audio('tmp_sound.wav', signal, samplerate)

        play_audio(signal, samplerate, playtime=5)

        plt.close()

