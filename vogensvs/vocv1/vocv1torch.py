import numpy as np
import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F
from . import fftSize, hopSize, targetFs, windowArea


def cis(angle):
    return torch.exp(angle * 1j)

def interp1d(inXMin, inXMax, inYs, outXs, lerp=torch.lerp):
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
    indicesF = torch.clamp((outXs - inXMin) / (inXMax - inXMin) * maxIndex, 0., maxIndex)
    loIndices = indicesF.long()
    lerpAmount = indicesF - loIndices

    loValues = inYs[loIndices]
    hiValues = inYs[torch.clamp(loIndices + 1, max=maxIndex)]
    for _ in inYs.shape[1:]:
        lerpAmount = torch.unsqueeze(lerpAmount, -1)
    outYs = lerp(loValues, hiValues, lerpAmount)
    return outYs


class Kaiser(nn.Module):
    # kaiser default params as in https://github.com/bmcfee/resampy/blob/master/resampy/filters.py
    def __init__(self, windowRadius=32768, zeroCrossings=64, rolloff=0.9475937167399596, kaiserBeta=14.769656459379492, **kwargs):
        super().__init__(**kwargs)
        self.windowRadius = windowRadius
        self.zeroCrossings = zeroCrossings
        self.rolloff = rolloff
        self.kaiserBeta = kaiserBeta

        kaiserWindow = np.kaiser(self.windowRadius * 2 + 1, self.kaiserBeta)[self.windowRadius:]
        outWindow = kaiserWindow * np.sinc(np.linspace(0, self.zeroCrossings * self.rolloff, num=self.windowRadius + 1)) * self.rolloff
        outWindow[-1] = 0

        self.unitSampleCount = torch.tensor(self.windowRadius / self.zeroCrossings, dtype=torch.float32)
        self.outWindow = torch.tensor(outWindow, dtype=torch.float32)

    def forward(self, inputs):
        inputs = (inputs if isinstance(inputs, torch.Tensor) else torch.tensor(inputs)).float()
        inputShape = inputs.shape
        x = torch.reshape(inputs, (-1,))

        xIndicesF = torch.abs(x * self.unitSampleCount)
        xIndices = xIndicesF.long()
        xLerpAmount = xIndicesF - xIndices
        xLValues = torch.gather(self.outWindow, 0, torch.clamp(xIndices, max=self.windowRadius))
        xRValues = torch.gather(self.outWindow, 0, torch.clamp(xIndices + 1, max=self.windowRadius))
        y = torch.lerp(xLValues, xRValues, xLerpAmount)
        return torch.reshape(y, inputShape)

class PulseTrain(nn.Module):
    def __init__(self, hopSize, fs, initUnitPhase=0.5, **kwargs):
        super().__init__(**kwargs)
        self.hopSize = hopSize      # usually 10.
        self.fs = fs
        self.initUnitPhase = torch.tensor(initUnitPhase, dtype=torch.float64)

        self.hopSamplesF = self.hopSize / 1000. * fs
        self.hopSamples = int(round(self.hopSamplesF))
        assert self.hopSamples == self.hopSamplesF, f'HopSamples must be integer, but received {self.hopSamplesF}'

        self.kaiser = Kaiser()
        self.unitSampleXs = torch.arange(self.hopSamples, dtype=torch.float64) / self.hopSamples

    def forward(self, inputs):
        hopSize = self.hopSize
        fs = self.fs
        initUnitPhase = self.initUnitPhase
        hopSamplesF = self.hopSamplesF
        hopSamples = self.hopSamples
        unitSampleXs = self.unitSampleXs
        kaiser = self.kaiser
        h = self.hopSize / 1000.

        f00 = (inputs if isinstance(inputs, torch.Tensor) else torch.tensor(inputs)).double()
        f01 = torch.concat([f00[1:], torch.zeros((1,), dtype=torch.float64)])

        hasF0 = f00.bool() | f01.bool()
        v0 = f00[hasF0]
        v1 = f01[hasF0]
        v0or1 = torch.where(v0.bool(), v0, v1)
        v1or0 = torch.where(v1.bool(), v1, v0)

        hopPoints = torch.cumsum(torch.concat([initUnitPhase[None], h * (v0or1 + v1or0) / 2.]), dim=0)
        pulseCountPerFrame = torch.diff(hopPoints.long())
        pulseCount = hopPoints[-1].long()
        pulseHopOffset = torch.arange(1, pulseCount + 1, dtype=torch.float64) - torch.repeat_interleave(hopPoints[:-1], pulseCountPerFrame)

        v0or1p = torch.repeat_interleave(v0or1, pulseCountPerFrame)
        v1or0p = torch.repeat_interleave(v1or0, pulseCountPerFrame)
        pulseTimePhase = torch.where(
            v0or1p == v1or0p,
            pulseHopOffset / v0or1p,
            (h * v0or1p - torch.sqrt(h * (h * torch.square(v0or1p) - 2. * (v0or1p - v1or0p) * pulseHopOffset))) / (v0or1p - v1or0p))
        self.pulseTime = pulseTimePhase + torch.repeat_interleave(torch.arange(f00.shape[0], dtype=torch.float64)[hasF0] * h, pulseCountPerFrame)
        self.pulseSampleTime = pulseSampleTime = self.pulseTime * fs

        vuv0p = torch.repeat_interleave(v0.bool(), pulseCountPerFrame)
        vuv1p = torch.repeat_interleave(v1.bool(), pulseCountPerFrame)
        self.pulseValue = pulseValue = torch.maximum(
            (vuv0p & vuv1p).double(),
            torch.square(torch.sin((pulseTimePhase / h - vuv0p.double()) * (np.pi / 2.))))

        f00ud = f00[..., None]
        f01ud = f01[..., None]
        waveBaseOffset2d = torch.where(
            f00ud.bool() == f01ud.bool(),
            -(f00ud + (f01ud - f00ud) * unitSampleXs) / fs,
            -(torch.maximum(f00ud, f01ud) / fs * torch.square(torch.sin((unitSampleXs - f00ud.bool().double()) * (np.pi / 2.)))))
        self.waveBaseOffset = waveBaseOffset = torch.reshape(waveBaseOffset2d, (-1,))

        pulseWriteIndices2D = pulseSampleTime.long() - torch.arange(-kaiser.zeroCrossings, kaiser.zeroCrossings)[:, None]
        pulseWriteValues2D = kaiser((pulseWriteIndices2D - pulseSampleTime).float()) * pulseValue.float()

        pulseWriteIndices = pulseWriteIndices2D.reshape(-1)
        pulseWriteValues = pulseWriteValues2D.reshape(-1)
        pulseWriteIndicesValid = (pulseWriteIndices >= 0) & (pulseWriteIndices < waveBaseOffset.shape[0])
        pulseTrain = torch.scatter_add(waveBaseOffset.float(), 0,
            pulseWriteIndices[pulseWriteIndicesValid], pulseWriteValues[pulseWriteIndicesValid])

        return pulseTrain

class HarmonicsExcitation(nn.Module):
    def __init__(self, hopSize, fs, initPhase=[], randPhase=True, normAmp=True, hiFilter=lambda _: True, **kwargs):
        super().__init__(**kwargs)
        self.hopSize = hopSize      # usually 10 (ms)
        self.fs = fs
        self.randPhase = randPhase
        self.normAmp = normAmp
        self.hiFilter = hiFilter

        self.hopSamplesF = self.hopSize / 1000. * fs
        self.hopSamples = int(round(self.hopSamplesF))
        assert self.hopSamples == self.hopSamplesF, f'HopSamples must be integer, but received {self.hopSamplesF}'

        self.hmss = np.array([48, 144, 432])
        self.f0Bounds = fs / 2 / np.concatenate([self.hmss[0:1] / 3, self.hmss])
        self.pFftSize = 2 ** int(np.ceil(np.log2(self.hopSamples * 2.1)))
        self.argInitPhase = np.array(initPhase[:self.hmss[-1]])

        self.inFrameXs2x = torch.arange(self.hopSamples * 2, dtype=torch.float64) / (self.hopSamples * 2)
        self.inFrameTrans2x = torch.square(torch.sin(self.inFrameXs2x * (np.pi / 2))).float()

        genInitPhaseCount = self.hmss[-1] - self.argInitPhase.shape[0]
        self.genInitPhase = (
            torch.zeros((genInitPhaseCount,), dtype=torch.float64) if not self.randPhase else
            torch.rand(genInitPhaseCount, dtype=torch.float64) * (np.pi * 2) - np.pi)
        self.initPhase = torch.concat([torch.tensor(self.argInitPhase, dtype=torch.float64), self.genInitPhase], axis=0)

        self.pys = []
        for hms in self.hmss:
            hi = np.arange(1, hms + 1)
            hiMask = np.vectorize(self.hiFilter)(np.arange(1, hms + 1)).astype(bool)

            px0 = torch.linspace(0., np.pi * 2, hms * 512 * 2 + 1, dtype=torch.float64)    # 512 samples per zero-crossing
            pxs = (torch.tensor(hi, dtype=torch.float64)[:, None] * px0 % (np.pi * 2) + self.initPhase[:hms, None])[hiMask]
            py = torch.sum(torch.sin(pxs), dim=0).float()
            self.pys.append(py)

    def forward(self, inputs):
        f0 = (inputs if isinstance(inputs, torch.Tensor) else torch.tensor(inputs)).double()

        f0padded = F.pad(f0, (0, 1))
        f0or1 = torch.where(f0padded[:-1].bool(), f0padded[:-1], f0padded[1:])
        f1or0 = torch.where(f0padded[1:].bool(), f0padded[1:], f0padded[:-1])
        f0cont = interp1d(0., 1., torch.stack([f0or1, f1or0], dim=0), self.inFrameXs2x).transpose(0, 1).reshape(-1)
        samplePhaseOffsets2x = (torch.cumsum(f0cont / (self.fs * 2), dim=0) % 1.).float()

        pOut2xs = []
        for maxF0, minF0, py in zip(self.f0Bounds, self.f0Bounds[1:], self.pys):
            vuvFBandedPadded = F.pad((f0 >= minF0) & (f0 < maxF0), (0, 1)).float()
            maskBanded2d2x = interp1d(
                0., 1., torch.stack([vuvFBandedPadded[:-1], vuvFBandedPadded[1:]], dim=0), self.inFrameTrans2x).transpose(0, 1)
            maskBanded2x = torch.reshape(maskBanded2d2x, (-1,))

            pBanded2x = maskBanded2x * interp1d(0., 1., py, samplePhaseOffsets2x)
            pOut2xs.append(pBanded2x)

        pOut2x = torch.sum(torch.stack(pOut2xs, dim=0), dim=0)
        if self.normAmp:
            pOut2x = pOut2x * f0cont.float() / self.fs

        pFftc2x = stft(pOut2x, self.pFftSize * 2, self.hopSamples * 2)
        pFftc = pFftc2x[:, :self.pFftSize // 2 + 1]
        pOut = istft(pFftc, self.hopSamples)
        return pOut

def stft(xs, fftSize, hopSamples):
    return torch.stft(xs, fftSize, hopSamples,
        window=torch.hann_window(fftSize), center=True, pad_mode='reflect', return_complex=True).transpose(-1, -2)

def istft(ffts, hopSamples):
    fftSize = (ffts.shape[-1] - 1) * 2
    return torch.istft(torch.transpose(ffts, -1, -2), fftSize, hopSamples,
        window=torch.hann_window(fftSize), center=True)

class Synthesizer(nn.Module):
    def __init__(self, hopSize, fs):
        super().__init__()

        self.hopSamplesF = hopSamplesF = hopSize * fs / 1000.
        self.hopSamples = hopSamples = int(round(hopSamplesF))
        assert hopSamples == hopSamplesF, f'HopSamples must be integer, but received {hopSamplesF}'

        self.brGen = torch.jit.load(r'vogensvs/assets/[vocv1-brgen]220801a-d1+WNorm-c16c12/220801a-d1+WNorm-c16c12_0127.jit.pt')
        _ = self.brGen.eval()

        self.heLayer = HarmonicsExcitation(hopSize, fs, randPhase=False, normAmp=False)

    def synthBrRandom(self, spnfftSW, iteration=4):
        q2fftSW = cis(torch.rand(*spnfftSW.shape) * (2 * np.pi))

        for i in range(iteration):
            yn2 = istft(spnfftSW * q2fftSW, self.hopSamples)
            yn2fftcSW = stft(yn2, self.hopSamples * 2, self.hopSamples)
            q2fftSW = cis(torch.angle(yn2fftcSW))

        yn2 = istft(spnfftSW * q2fftSW, self.hopSamples)
        return yn2

    def synthBrNeural(self, spnfftSW, yh):
        yhfftcSW = stft(yh, self.hopSamples * 2, self.hopSamples).transpose(-1, -2)[None]
        spnfftSW = spnfftSW.transpose(-1, -2)[None]
        z = torch.randn_like(spnfftSW)
        with torch.no_grad():
            ynfftp_r = self.brGen(spnfftSW, yhfftcSW, z)
        return istft((ynfftp_r * spnfftSW).transpose(-1, -2), self.hopSamples)[0]

    def synthHm(self, f0, sph4fft):
        fftSize = (sph4fft.shape[-1] - 1) * 2
        windowArea = fftSize / 2    # hann window
        p = self.heLayer(f0) / windowArea
        pfftc = stft(p, fftSize, self.hopSamples)[..., :sph4fft.shape[-2], :]
        yh4 = istft(sph4fft * pfftc, self.hopSamples)
        return yh4


