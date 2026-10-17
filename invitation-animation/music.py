"""Gentle original background music (~25s), synthesized with numpy.

D major, 76 BPM: harp arpeggios + warm bell melody + soft string pad + bass,
with a simple convolution reverb. Writes a 44.1kHz stereo 16-bit WAV.
"""
import sys
import wave

import numpy as np

SR = 44100
BPM = 76.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT
TOTAL = 25.0
rng = np.random.default_rng(7)


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


NOTE = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6,
        'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}


def n(name):
    """'F#5' -> midi number."""
    return 12 * (int(name[-1]) + 1) + NOTE[name[:-1]]


N = int(SR * (TOTAL + 0.5))
L = np.zeros(N)
R = np.zeros(N)


def add(sig, start, pan=0.0, gain=1.0):
    i = int(start * SR)
    if i >= N:
        return
    sig = sig[: N - i] * gain
    lg = np.cos((pan + 1) * np.pi / 4)
    rg = np.sin((pan + 1) * np.pi / 4)
    L[i:i + len(sig)] += sig * lg
    R[i:i + len(sig)] += sig * rg


def harp(m, dur=2.2):
    f = midi_hz(m)
    t = np.arange(int(dur * SR)) / SR
    s = np.zeros_like(t)
    for k, a in enumerate([1.0, 0.45, 0.22, 0.12, 0.06], start=1):
        fk = f * k * (1 + 0.0004 * k * k)
        s += a * np.sin(2 * np.pi * fk * t) * np.exp(-t * (1.6 + 1.1 * k))
    att = np.minimum(t / 0.004, 1.0)
    return s * att


def bell(m, dur=3.0):
    """Warm piano/celesta-like melody voice."""
    f = midi_hz(m)
    t = np.arange(int(dur * SR)) / SR
    s = (np.sin(2 * np.pi * f * t) * np.exp(-t * 1.1)
         + 0.35 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t * 2.2)
         + 0.12 * np.sin(2 * np.pi * 3.01 * f * t) * np.exp(-t * 3.5)
         + 0.05 * np.sin(2 * np.pi * 4.2 * f * t) * np.exp(-t * 6.0))
    att = np.minimum(t / 0.008, 1.0)
    return s * att


def pad(midis, dur):
    """Soft string pad: detuned sine stacks, slow swell."""
    t = np.arange(int(dur * SR)) / SR
    s = np.zeros_like(t)
    for m in midis:
        f = midi_hz(m)
        for det in (-0.12, 0.0, 0.11):
            ff = f * 2 ** (det / 12)
            ph = rng.uniform(0, 2 * np.pi)
            vib = 0.0025 * np.sin(2 * np.pi * 4.8 * t + ph)
            s += np.sin(2 * np.pi * ff * t * (1 + vib) + ph)
            s += 0.25 * np.sin(2 * np.pi * 2 * ff * t + ph)
    env = np.minimum(t / 1.2, 1.0) * np.minimum((dur - t) / 1.0, 1.0)
    return s * np.clip(env, 0, 1) / (len(midis) * 3)


def bass(m, dur):
    f = midi_hz(m)
    t = np.arange(int(dur * SR)) / SR
    s = (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t)) * np.exp(-t * 0.6)
    return s * np.minimum(t / 0.01, 1.0)


# chord: (bass note, chord tones for arpeggio/pad)
chords = [
    ('D2', ['D3', 'A3', 'D4', 'F#4', 'A4']),
    ('C#2', ['A2', 'E3', 'A3', 'C#4', 'E4']),
    ('B1', ['B2', 'F#3', 'B3', 'D4', 'F#4']),
    ('G1', ['G2', 'D3', 'G3', 'B3', 'D4']),
    ('F#1', ['D3', 'A3', 'D4', 'F#4', 'A4']),
    ('E2', ['E3', 'B3', 'E4', 'G4', 'B4']),
    ('A1', ['A2', 'E3', 'A3', 'C#4', 'E4']),
    ('D2', ['D3', 'A3', 'D4', 'F#4', 'A4']),
]
arp_idx = [0, 1, 2, 3, 4, 3, 2, 1]

for b, (bn, tones) in enumerate(chords):
    t0 = b * BAR
    last = b == len(chords) - 1
    add(bass(n(bn) + 12, BAR * (1.6 if last else 1.2)), t0, pan=0.0, gain=0.32)
    pd = pad([n(x) for x in tones[1:4]], BAR * (1.4 if last else 1.15))
    add(pd, t0, pan=0.0, gain=0.10)
    if last:
        # final rolled chord
        for k, tn in enumerate(tones + ['D5']):
            add(harp(n(tn), 4.0), t0 + k * 0.07, pan=-0.3 + 0.12 * k, gain=0.22)
        continue
    for k, ai in enumerate(arp_idx):
        tn = tones[ai]
        add(harp(n(tn)), t0 + k * BEAT / 2, pan=-0.35 + 0.1 * ai, gain=0.20)

# melody: (bar, beat offset, note, length in beats)
melody = [
    (0, 2.0, 'A4', 1), (0, 3.0, 'D5', 1),
    (1, 0.0, 'E5', 2), (1, 2.0, 'C#5', 1), (1, 3.0, 'A4', 1),
    (2, 0.0, 'D5', 1.5), (2, 1.5, 'C#5', 0.5), (2, 2.0, 'B4', 2),
    (3, 0.0, 'B4', 1), (3, 1.0, 'D5', 1), (3, 2.0, 'G5', 1.5), (3, 3.5, 'F#5', 0.5),
    (4, 0.0, 'F#5', 2), (4, 2.0, 'A5', 1), (4, 3.0, 'F#5', 1),
    (5, 0.0, 'G5', 1.5), (5, 1.5, 'F#5', 0.5), (5, 2.0, 'E5', 1), (5, 3.0, 'D5', 1),
    (6, 0.0, 'C#5', 1.5), (6, 1.5, 'D5', 0.5), (6, 2.0, 'E5', 2),
    (7, 0.0, 'D5', 4),
]
for bar, off, nm, ln in melody:
    st = bar * BAR + off * BEAT
    # slight human timing
    st += rng.normal(0, 0.006)
    add(bell(n(nm), max(ln * BEAT + 1.8, 2.5)), st, pan=0.08, gain=0.26)

# --- reverb: decaying stereo noise impulse response, FFT convolution ---
ir_len = int(2.6 * SR)
ti = np.arange(ir_len) / SR


def make_ir():
    noise = rng.normal(0, 1, ir_len)
    # gentle lowpass via moving average to darken the tail
    k = 12
    noise = np.convolve(noise, np.ones(k) / k, mode='same')
    ir = noise * np.exp(-ti * 2.6)
    ir[: int(0.012 * SR)] = 0  # predelay
    return ir / np.sqrt(np.sum(ir ** 2))


def conv(x, h):
    size = 1 << int(np.ceil(np.log2(len(x) + len(h))))
    y = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(h, size), size)
    return y[: len(x)]


wet = 0.28
Lr = conv(L, make_ir())
Rr = conv(R, make_ir())
outL = L * (1 - wet) + Lr * wet * 1.0
outR = R * (1 - wet) + Rr * wet * 1.0

# trim, fades
outL = outL[: int(TOTAL * SR)]
outR = outR[: int(TOTAL * SR)]
t = np.arange(len(outL)) / SR
fade = np.minimum(t / 0.8, 1.0) * np.clip((TOTAL - t) / 2.5, 0, 1) ** 1.5
outL *= fade
outR *= fade

# soft saturation + normalize
st = np.stack([outL, outR], axis=1)
st /= np.max(np.abs(st)) + 1e-9
st = np.tanh(st * 1.2) / np.tanh(1.2)
st *= 0.89

out = sys.argv[1] if len(sys.argv) > 1 else 'music.wav'
with wave.open(out, 'wb') as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((st * 32767).astype('<i2').tobytes())
print('wrote', out, len(st) / SR, 's')
