import numpy as np
import scipy.fft
import librosa


# Global parameter settings
fftSize = 2048
hopSize = 10.   # frame duration in milliseconds between frames
targetFs = 44100

hopSamplesF = hopSize * targetFs / 1000.
hopSamples = int(round(hopSamplesF))
assert hopSamples == hopSamplesF, f'HopSamples must be integer, but received {hopSamplesF}'

windowArea = fftSize / 2    # hann window


def ensureFs(x: np.ndarray, fs: int) -> tuple[np.ndarray, int]:
    if len(x.shape) != 1:
        raise Exception(f'x is not mono (shape is {x.shape})')
    if fs != targetFs:
        x = librosa.resample(x, orig_sr=fs, target_sr=targetFs, res_type='fft')
        fs = targetFs
    return x, fs


