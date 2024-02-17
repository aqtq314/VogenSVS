import numpy as np
import scipy.fft
import librosa
import onnxruntime as ort
import pyworld as pw
from . import fftSize, hopSize, targetFs, windowArea



def f0Extract(x: np.ndarray, fs: int, mode='dio', useStonemask=True, f0Floor=60, f0Ceil=1400, hopSize=hopSize, **kwargs):
    f0, t = {
        'dio': lambda: pw.dio(x, fs, f0_floor=f0Floor, f0_ceil=f0Ceil, frame_period=hopSize, **kwargs),
        'harvest': lambda: pw.harvest(x, fs, f0_floor=f0Floor, f0_ceil=f0Ceil, frame_period=hopSize, **kwargs)
    }[mode]()

    if useStonemask:
        f0 = pw.stonemask(x, f0, t, fs)

    return f0, t

def cis(angle):
    return np.exp(angle * 1j)

def lerp(x, y, amount):
    return x * (np.ones_like(amount) - amount) + y * amount

def interp1d(inXMin, inXMax, inYs, outXs, lerp=lerp):
    '''
    Input:
        inXMin: shape=[], dtype=float32
        inXMax: shape=[], dtype=float32
        inYs: shape=[inSampleCount, *sampleShapes], dtype=float32
        outXs: shape=[outSampleCount], dtype=float32
    Output:
        outYs: shape=[outSampleCount, *sampleShapes], dtype=float32
    '''
    maxIndex = inYs.shape[0] - 1
    maxIndexF = np.array(maxIndex, outXs.dtype)
    indicesF = np.clip((outXs - inXMin) / (inXMax - inXMin) * maxIndexF, a_min=0., a_max=maxIndexF)
    loIndices = indicesF.astype(int)
    lerpAmount = indicesF - loIndices.astype(indicesF.dtype)

    loValues = inYs[loIndices]
    hiValues = inYs[np.clip(loIndices + 1, a_min=None, a_max=maxIndex)]
    for _ in inYs.shape[1:]:
        lerpAmount = lerpAmount[..., None]
    outYs = lerp(loValues, hiValues, lerpAmount)
    return outYs


class Kaiser():
    def __init__(self, windowRadius, zeroCrossings, rolloff, kaiserBeta):
        self.windowRadius = windowRadius
        self.zeroCrossings = zeroCrossings
        self.rolloff = rolloff
        self.kaiserBeta = kaiserBeta

        kaiserWindow = np.kaiser(windowRadius * 2 + 1, kaiserBeta)[windowRadius:]
        self.outWindow = kaiserWindow * np.sinc(np.linspace(0, zeroCrossings * rolloff, num=windowRadius + 1)) * rolloff
        self.xs = np.linspace(0., zeroCrossings, num=windowRadius + 1)

    def __call__(self, x):
        return np.interp(np.abs(x), self.xs, self.outWindow, right=0.)

class PulseTrain():
    def __init__(self, hopSize: float, fs: int, initUnitPhase=0.5, normKaiser=True, normAmp=True):
        self.hopSize = hopSize      # usually 10.
        self.fs = fs
        self.initUnitPhase = initUnitPhase
        self.normKaiser = normKaiser
        self.normAmp = normAmp

        self.hopSamplesF = self.hopSize * fs / 1000.
        self.hopSamples = int(round(self.hopSamplesF))
        assert self.hopSamples == self.hopSamplesF, f'HopSamples must be integer, but received {self.hopSamplesF}'

        # params stolen from https://github.com/bmcfee/resampy/blob/819621f1555742848826d7cf448c446aa0ccc08f/resampy/filters.py
        self.kaiser = Kaiser(32768, zeroCrossings=64, rolloff=0.9475937167399596, kaiserBeta=14.769656459379492)
        self.kaiserCompensation = np.abs(np.fft.rfft(self.kaiser(np.arange(-fftSize // 2, fftSize // 2, dtype=float))))

    def __call__(self, f0: np.ndarray):
        hopSize = self.hopSize
        fs = self.fs
        initUnitPhase = self.initUnitPhase
        hopSamplesF = self.hopSamplesF
        hopSamples = self.hopSamples
        kaiser = self.kaiser
        h = self.hopSize / 1000.
        pt = np.zeros((hopSamples * len(f0),),)

        pulseTime = []
        pulseValue = []
        y0 = initUnitPhase
        for i, (currF0, nextF0) in enumerate(zip(f0, np.array([*f0[1:], 0.]))):
            currSi = i * hopSamples
            unitSampleXs = np.arange(hopSamples, dtype=float) / hopSamples

            if currF0 > 0. and nextF0 > 0.:
                v0, v1 = currF0, nextF0

                y1 = y0 + h * (v0 + v1) / 2.
                k = np.arange(1, int(y1) + 1, dtype=float)
                pts = (k - y0) / v0 if v0 == v1 else (
                    (h * v0 - np.sqrt(h * (h * v0 * v0 - 2. * (v0 - v1) * (k - y0)))) / (v0 - v1))

                pss = np.ones_like(pts)
                pt[currSi:(currSi + hopSamples)] = -(v0 + (v1 - v0) * unitSampleXs) / fs

            elif currF0 > 0. or nextF0 > 0.:
                v = currF0 if currF0 > 0. else nextF0

                y1 = y0 + h * v
                k = np.arange(1, int(y1) + 1, dtype=float)
                pts = (k - y0) / v

                if currF0 <= 0.:    # fade in
                    pss = np.sin(pts / h * np.pi / 2.) ** 2.
                    pt[currSi:(currSi + hopSamples)] = -(v / fs * np.sin(unitSampleXs * np.pi / 2.) ** 2.)

                else:  # fade out
                    pss = np.cos(pts / h * np.pi / 2.) ** 2.
                    pt[currSi:(currSi + hopSamples)] = -(v / fs * np.cos(unitSampleXs * np.pi / 2.) ** 2.)

            else:
                continue

            pulseTime.extend(pts + i * h)
            pulseValue.extend(pss)

            y0 = y1 % 1.

        pulseTime = np.array(pulseTime)
        pulseValue = np.array(pulseValue)

        pulseSampleTime = pulseTime * fs
        ptWriteIndices2D = np.floor(pulseSampleTime).astype(np.int64) + np.arange(kaiser.zeroCrossings, -kaiser.zeroCrossings, step=-1)[:, None]
        ptWriteValues2D = kaiser(ptWriteIndices2D - pulseSampleTime) * pulseValue
        ptWriteIndexValid = (ptWriteIndices2D >= 0) & (ptWriteIndices2D < pt.shape[0])
        for i in range(pulseTime.shape[0]):
            ptWriteIndices = np.extract(ptWriteIndexValid[:, i], ptWriteIndices2D[:, i])
            ptWriteValues = np.extract(ptWriteIndexValid[:, i], ptWriteValues2D[:, i])
            pt[ptWriteIndices] += ptWriteValues

        # optional high-frequency amplitude compensation for kaiser window
        if self.normKaiser:
            pfftc = librosa.stft(pt, n_fft=fftSize, hop_length=hopSamples, pad_mode='reflect').T
            pfftc /= self.kaiserCompensation
            pt = librosa.istft(pfftc.T, hop_length=hopSamples)

        # amplitude normalization
        if not self.normAmp:
            for i, (currF0, nextF0) in enumerate(zip(f0, np.r_[f0[1:], [0.]])):
                currSi = i * hopSamples

                if currF0 > 0. and nextF0 > 0.:
                    v0, v1 = currF0, nextF0
                    pt[currSi:(currSi + hopSamples)] /= np.linspace(v0, v1, hopSamples, endpoint=False) / fs
                    
                elif currF0 > 0. or nextF0 > 0.:
                    v = currF0 if currF0 > 0. else nextF0
                    pt[currSi:(currSi + hopSamples)] /= v / fs

        return pt

class HarmonicsExcitation():
    def __init__(self, hopSize, fs, initPhase=[], randPhase=True, normAmp=True, hiFilter=lambda _: True, **kwargs):
        super().__init__(**kwargs)
        self.hopSize = hopSize      # usually 10 (ms)
        self.fs = fs
        self.randPhase = randPhase
        self.normAmp = normAmp
        self.hiFilter = hiFilter

        self.hopSamplesF = self.hopSize * fs / 1000.
        self.hopSamples = int(round(self.hopSamplesF))
        assert self.hopSamples == self.hopSamplesF, f'HopSamples must be integer, but received {self.hopSamplesF}'

        self.hmss = np.array([48, 144, 432])
        self.f0Bounds = fs / 2 / np.concatenate([self.hmss[0:1] / 3, self.hmss])
        self.pFftSize = 2 ** int(np.ceil(np.log2(self.hopSamples * 2.1)))
        self.argInitPhase = np.array(initPhase[:self.hmss[-1]])

        self.inFrameXs2x = np.arange(self.hopSamples * 2, dtype=np.float64) / (self.hopSamples * 2)
        self.inFrameTrans2x = np.square(np.sin(self.inFrameXs2x * (np.pi / 2))).astype(np.float32)

        genInitPhaseCount = self.hmss[-1] - self.argInitPhase.shape[0]
        self.genInitPhase = (
            np.zeros((genInitPhaseCount,)) if not self.randPhase else
            np.random.uniform(size=(genInitPhaseCount,), low=-np.pi, high=np.pi))
        self.initPhase = np.concatenate([self.argInitPhase, self.genInitPhase], axis=0)

        self.pys = []
        for hms in self.hmss:
            hi = np.arange(1, hms + 1)
            hiMask = np.vectorize(self.hiFilter)(np.arange(1, hms + 1)).astype(bool)

            px0 = np.linspace(0., np.pi * 2, hms * 512 * 2 + 1)    # 512 samples per zero-crossing
            pxs = (hi[:, None] * px0 % (np.pi * 2) + self.initPhase[:hms, None])[hiMask]
            py = np.sum(np.sin(pxs), axis=0).astype(np.float32)
            self.pys.append(py)

    def __call__(self, f0):
        f0padded = np.pad(f0, [(0, 1)])
        f0or1 = np.where(f0padded[:-1].astype(bool), f0padded[:-1], f0padded[1:])
        f1or0 = np.where(f0padded[1:].astype(bool), f0padded[1:], f0padded[:-1])
        f0cont = np.reshape(interp1d(0., 1., np.stack([f0or1, f1or0], axis=-2), self.inFrameXs2x).T, (-1,))
        samplePhaseOffsets2x = (np.cumsum(f0cont / (self.fs * 2), axis=0) % 1.).astype(np.float32)

        pOut2xs = []
        for maxF0, minF0, py in zip(self.f0Bounds, self.f0Bounds[1:], self.pys):
            vuvFBandedPadded = np.pad((f0 >= minF0) & (f0 < maxF0), [(0, 1)]).astype(np.float32)
            maskBanded2d2x = interp1d(0., 1., np.stack([vuvFBandedPadded[:-1], vuvFBandedPadded[1:]], axis=0), self.inFrameTrans2x).T
            maskBanded2x = np.reshape(maskBanded2d2x, (-1,))

            pBanded2x = maskBanded2x * interp1d(0., 1., py, samplePhaseOffsets2x)
            pOut2xs.append(pBanded2x)

        pOut2x = np.sum(pOut2xs, axis=0)
        if self.normAmp:
            pOut2x = pOut2x * f0cont.astype(np.float32) / self.fs

        pFftc2x = stft(pOut2x, self.pFftSize * 2, self.hopSamples * 2)
        pFftc = pFftc2x[:, :self.pFftSize // 2 + 1]
        pOut = istft(pFftc, self.hopSamples)
        return pOut

def cheaptrick(x, f0, t, fs, pwFftSize=None, normAmp=False):
    pwFftSize = pwFftSize or pw.get_cheaptrick_fft_size(fs)
    windowArea = pwFftSize / 2
    spxfft = pw.cheaptrick(x.astype(np.float64), f0, t, fs, fft_size=pwFftSize)
    if normAmp:
        spxfft = np.sqrt(spxfft * (fs / f0[:, None]))
    else:
        spxfft = np.sqrt(spxfft * (f0[:, None] / fs)) * windowArea
    spxfft = np.where((f0 > 0)[:, None], spxfft, 1e-12)
    return spxfft.astype(x.dtype)

def dctResample(input, inDim, outDim):
    output = scipy.fft.idct(scipy.fft.dct(input, type=1)[..., :inDim], type=1, n=outDim, norm='forward')
    output /= input.shape[-1] * 2  # manual dct & idct normalization
    return output

def dctResampleVar(input, f0, outDim):
    cepCutOff = targetFs / 2 / np.where(f0 > 0, f0, 173.)     # num of harmonics in spectrum
    cepCutOffMask = np.arange(outDim) <= cepCutOff[:, None]

    output = scipy.fft.idct(np.where(cepCutOffMask, scipy.fft.dct(input, type=1), 0), type=1, n=outDim, norm='forward')
    output /= input.shape[-1] * 2  # manual dct & idct normalization
    return output

def stft(xs, fftSize, hopSamples):
    return librosa.stft(xs, n_fft=fftSize, hop_length=hopSamples, window='hann', center=True, pad_mode='reflect').swapaxes(-1, -2)

def istft(ffts, hopSamples):
    return librosa.istft(ffts.swapaxes(-1, -2), hop_length=hopSamples, window='hann', center=True)


class Analyzer:
    defaultOnnxProvider = ['CPUExecutionProvider']

    def __init__(self, ortProviders=None):
        modelPath = 'vogensvs/assets/[vocv1-hbdecomp][220228-001211][img73]modelv5c/model.onnx'
        ortProviders = ortProviders if ortProviders is not None else Analyzer.defaultOnnxProvider
        self.onnxSess = ort.InferenceSession(modelPath, providers=ortProviders)

        hopSamplesF = hopSize / 1000. * targetFs
        hopSamples = int(round(hopSamplesF))
        assert hopSamples == hopSamplesF, f'HopSamples must be integer, but received {hopSamplesF}'
        self.hopSamples = hopSamples

    def decompose(self, x):
        hopSamples = self.hopSamples

        xfftc = stft(x, fftSize, hopSamples)

        onnxIn = {'X': np.stack([xfftc.real, xfftc.imag], axis=-1)[None].astype(np.float32)}
        onnxOut, = self.onnxSess.run(None, onnxIn)
        xhfftcDecomposed = (onnxOut[0, ..., 0] + onnxOut[0, ..., 1] * 1j)
        xnfftcDecomposed = (onnxOut[0, ..., 2] + onnxOut[0, ..., 3] * 1j)

        xh = istft(xhfftcDecomposed, hopSamples)
        xn = istft(xnfftcDecomposed, hopSamples)

        return xh, xn

    def extractHmEnvelope(self, x, xh, f0, t, sphSmoothen=False):
        spxfft = cheaptrick(x, f0, t, targetFs, normAmp=False).astype(np.float32)
        sphfft = cheaptrick(xh, f0, t, targetFs, normAmp=False).astype(np.float32)
        logSpxfft = np.log(spxfft)
        logSphfft = np.log(sphfft)
        logSph4fft = logSphfft if not sphSmoothen else logSpxfft - dctResample(logSpxfft - logSphfft, 16, fftSize // 2 + 1)
        sph4fft = np.exp(logSph4fft)

        return sph4fft

    def extractBrEnvelope(self, xn, f0):
        hopSamples = self.hopSamples

        xnfftcSW = stft(np.pad(xn, [(0, fftSize)], mode='reflect'), hopSamples * 2, hopSamples)[:len(f0)]
        xnfftSW = np.abs(xnfftcSW)
        logXnfftSW = np.log(xnfftSW)
        logSpnfftSW = dctResampleVar(logXnfftSW, f0, hopSamples + 1)
        spnfftSW = np.exp(logSpnfftSW)

        # spn power correction
        powerRatio = (
            np.sum(np.square(xnfftSW), axis=-1, keepdims=True) /
            np.sum(np.square(spnfftSW), axis=-1, keepdims=True))
        #logSpnfftSW = logSpnfftSW + np.log(powerRatio) / 2
        spnfftSW = spnfftSW * np.sqrt(powerRatio)

        return spnfftSW

    def analyze(self, x, f0, t, sphSmoothen=False):
        xh, xn = self.decompose(x)
        sph4fft = self.extractHmEnvelope(x, xh, f0, t, sphSmoothen=sphSmoothen)
        spnfftSW = self.extractBrEnvelope(xn, f0)
        return sph4fft, spnfftSW

class Synthesizer:
    def __init__(self, hopSize, fs):
        self.hopSamplesF = hopSamplesF = hopSize / 1000. * fs
        self.hopSamples = hopSamples = int(round(hopSamplesF))
        assert hopSamples == hopSamplesF, f'HopSamples must be integer, but received {hopSamplesF}'

        # self.heLayer = HarmonicsExcitation(hopSize, fs, randPhase=False, normAmp=False)
        self.heLayer = PulseTrain(hopSize, fs, normAmp=False)

    def synthBrRandom(self, spnfftSW, iteration=4):
        q2fftSW = cis(np.random.uniform(size=spnfftSW.shape, low=0., high=(2 * np.pi)))

        for i in range(iteration):
            yn2 = istft(spnfftSW * q2fftSW, self.hopSamples)
            yn2fftcSW = stft(yn2, self.hopSamples * 2, self.hopSamples)
            q2fftSW = cis(np.angle(yn2fftcSW))

        yn2 = istft(spnfftSW * q2fftSW, self.hopSamples)
        return yn2

    def synthHm(self, f0, sph4fft):
        p = self.heLayer(f0) / windowArea
        pfftc = stft(p, fftSize, self.hopSamples)[..., :sph4fft.shape[-2], :]
        yh4 = istft(sph4fft * pfftc, self.hopSamples)
        return yh4


def freqToMel(f):
    return 1127.01048 * np.log(f / 700. + 1)

def melToFreq(mel):
    return 700 * (np.exp(mel / 1127.01048) - 1)

def logSp2mgc(logSp, fs, ndim=60, useMelAxisFromWorld=False):
    ceilFreq = np.minimum(fs / 2., 20000)
    if not useMelAxisFromWorld:
        melAxis = np.linspace(freqToMel(40), freqToMel(ceilFreq), 1024)
    else:
        melAxis = np.linspace(freqToMel(40), freqToMel(ceilFreq), 1025)[:-1]
        ceilFreq = melToFreq(melAxis[-1])   # 19932.773471082863

    logSpWarped = np.transpose(interp1d(0., fs / 2., np.transpose(logSp), melToFreq(melAxis)))
    mgc = scipy.fft.dct(logSpWarped, type=2, norm='ortho')[..., :ndim]
    return mgc

def splog(sp):
    return np.log(np.maximum(sp, 1e-5))

def sp2mgc(sp, fs, **logSp2mgcKwargs):
    logSp = splog(sp)
    return logSp2mgc(logSp, fs, **logSp2mgcKwargs)

def mgc2logSp(mgc, fs, fftSize, useMelAxisFromWorld=False):
    ceilFreq = np.minimum(fs / 2., 20000)
    if not useMelAxisFromWorld:
        floorMel = freqToMel(40)
        ceilMel = freqToMel(ceilFreq)
    else:
        floorMel = freqToMel(40)
        ceilMel = np.linspace(freqToMel(40), freqToMel(ceilFreq), 1025)[-2]     # 3813.302734375
        ceilFreq = melToFreq(ceilMel)   # 19932.773471082863

    freqAxis = np.linspace(0., fs / 2, fftSize // 2 + 1)
    logSpWarped = scipy.fft.idct(mgc, type=2, n=1024, norm='ortho')
    logSp = np.transpose(interp1d(floorMel, ceilMel, np.transpose(logSpWarped), freqToMel(freqAxis)))
    return logSp

def mgc2sp(mgc, fs, fftSize, **mgc2logSpKwargs):
    logSp = mgc2logSp(mgc, fs, fftSize, **mgc2logSpKwargs)
    sp = np.exp(logSp)
    return sp


