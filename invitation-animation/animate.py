"""Animated wedding invitation (~25s) rendered from a single still image.

The invitation text is removed from the image ("blank" plate) and then
revealed line by line, right-to-left, with a soft golden pen-light edge.
A slow camera move, gold shimmer sweeps, sparkles and floating gold dust
complete the effect. Frames are piped to ffmpeg together with music.wav.

usage: python3 animate.py SRC.jpg MUSIC.wav OUT.mp4 [--preview T1,T2,...]
"""
import multiprocessing as mp
import subprocess
import sys

import numpy as np
from PIL import Image, ImageFilter

FPS = 30
DURATION = 25.0
OW, OH = 1080, 1620
SRC_W, SRC_H = 1080, 1620  # source image size
CREAM = np.array([246, 242, 233], np.float32)
GLOW = np.array([255, 226, 160], np.float32)
SPARK = np.array([255, 246, 215], np.float32)

# Reveal elements: name, bbox (x0, y0, x1, y1), start, duration, mode, gold
ELEMENTS = [
    ('monogram', (134, 248, 506, 441), 0.8, 1.9, 'radial', True),
    ('bezh', (630, 560, 750, 588), 5.0, 0.6, 'fade', False),
    ('header', (424, 589, 968, 657), 5.3, 1.6, 'rtl', True),
    ('l3', (480, 664, 900, 695), 6.9, 0.9, 'rtl', False),
    ('l4', (560, 696, 820, 716), 7.5, 0.6, 'rtl', False),
    ('l5', (520, 717, 900, 742), 7.9, 0.8, 'rtl', False),
    ('labels', (420, 746, 968, 769), 8.5, 0.7, 'rtl', False),
    ('names', (386, 770, 968, 809), 8.9, 1.3, 'rtl', True),
    ('d1', (470, 813, 900, 842), 10.1, 0.7, 'rtl', False),
    ('d2', (470, 843, 900, 866), 10.5, 0.7, 'rtl', False),
    ('d3', (470, 867, 900, 889), 10.9, 0.7, 'rtl', False),
    ('d4', (470, 890, 900, 912), 11.3, 0.7, 'rtl', False),
    ('d5', (470, 913, 900, 940), 11.7, 0.7, 'rtl', False),
    ('nismach', (520, 943, 860, 972), 12.3, 0.8, 'rtl', False),
    ('plabels', (440, 976, 960, 997), 12.9, 0.6, 'rtl', False),
    ('pnames', (386, 998, 972, 1023), 13.2, 1.1, 'rtl', True),
    ('cities', (440, 1024, 940, 1046), 13.8, 0.6, 'rtl', False),
    ('ornament', (480, 1086, 900, 1166), 14.6, 1.4, 'center', True),
    ('gam', (420, 1228, 968, 1261), 15.7, 1.0, 'rtl', False),
    ('gp1', (386, 1264, 968, 1296), 16.4, 0.9, 'rtl', False),
    ('gp2', (386, 1297, 968, 1332), 17.0, 0.9, 'rtl', False),
    ('banner', (28, 1452, 1028, 1516), 19.9, 2.0, 'rtl', False),
]
# Card frame lines that must never be treated as text.
FRAME_COLS = [(370, 391), (972, 986)]

# Gold shimmer sweeps: element names (or 'all'), start, duration
SHIMMERS = [
    (['monogram'], 2.5, 1.2),
    (['header'], 7.0, 1.0),
    (['names'], 10.2, 1.0),
    (['pnames'], 14.3, 0.9),
    (['ornament'], 16.0, 1.0),
    (['monogram', 'header', 'names', 'pnames', 'ornament'], 22.0, 1.8),
]
# Sparkle bursts: element names, count, t0, t1
SPARKLES = [
    (['monogram'], 16, 1.6, 5.0),
    (['header'], 4, 6.8, 8.2),
    (['names'], 7, 10.0, 12.2),
    (['pnames'], 4, 14.0, 15.6),
    (['ornament'], 6, 15.4, 17.6),
    (['monogram', 'header', 'names', 'pnames', 'ornament'], 20, 21.4, 24.6),
]
# Camera keyframes: t, cx, cy, crop width (height = width * 1.5)
CAMERA = [
    (0.0, 320, 480, 640),
    (3.7, 346, 520, 692),
    (5.7, 700, 800, 740),
    (17.6, 700, 1010, 740),
    (21.6, 540, 810, 1080),
    (25.0, 540, 810, 1080),
]


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def ease_io(x):
    x = min(max(x, 0.0), 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * x)


def box_blur(a, r):
    """Approximate gaussian: three box-blur passes over both axes."""
    out = a.astype(np.float32)
    for _ in range(3):
        for axis in (0, 1):
            pad = [(0, 0)] * out.ndim
            pad[axis] = (r + 1, r)
            c = np.cumsum(np.pad(out, pad, mode='edge'), axis=axis, dtype=np.float64)
            hi = np.take(c, np.arange(2 * r + 1, c.shape[axis]), axis=axis)
            lo = np.take(c, np.arange(0, c.shape[axis] - 2 * r - 1), axis=axis)
            out = ((hi - lo) / (2 * r + 1)).astype(np.float32)
    return out


def norm_conv(img, w, r):
    num = box_blur(img * w[..., None], r)
    den = box_blur(w, r)
    return num / np.maximum(den, 1e-4)[..., None], den


def mask_filter(m, f):
    return np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(f), np.float32) / 255.0


def build_plates(S):
    """Return blank plate B, per-element text masks and gold masks."""
    B = S.copy()
    info = {}
    H, W, _ = S.shape
    for name, (x0, y0, x1, y1), *_ in ELEMENTS:
        m = 34
        X0, Y0, X1, Y1 = max(x0 - m, 0), max(y0 - m, 0), min(x1 + m, W), min(y1 + m, H)
        reg = S[Y0:Y1, X0:X1]
        lum = reg.mean(2)
        rb = reg[..., 0] - reg[..., 2]
        ref1 = box_blur(reg, 20)
        t1 = (lum < ref1.mean(2) - 10) | (rb > (ref1[..., 0] - ref1[..., 2]) + 20)
        ref2, _ = norm_conv(reg, (~t1).astype(np.float32), 20)
        lref = ref2.mean(2)
        rbref = ref2[..., 0] - ref2[..., 2]
        if name == 'banner':
            # dark lettering on a translucent strip over satin: colour is noise
            t2 = lum < lref - 10
        else:
            t2 = (lum < lref - 7) | (rb > rbref + 15)
        if name not in ('monogram', 'banner'):
            # frame lines are faint: inside them only count clearly inked pixels
            strong = (lum < lref - 20) | (rb > rbref + 25)
            for c0, c1 in FRAME_COLS:
                a, b = max(c0 - X0, 0), max(min(c1 - X0, X1 - X0), 0)
                t2[:, a:b] = strong[:, a:b]
        # Inpaint every inked pixel of the padded region (neighbouring lines
        # included, so they never bleed into the fill), with a ring wide enough
        # to swallow JPEG ringing; only the element bbox is written back.
        td = mask_filter(t2.astype(np.float32), ImageFilter.MaxFilter(9)) > 0.5
        init, den = norm_conv(reg, (~td).astype(np.float32), 10)
        init = np.where((den < 0.04)[..., None], ref2, init)
        fill = harmonic_fill(reg, td, init, 500)
        tf = mask_filter(td.astype(np.float32), ImageFilter.GaussianBlur(1.2))
        blank = reg * (1 - tf[..., None]) + fill * tf[..., None]
        sl = (slice(y0 - Y0, y1 - Y0), slice(x0 - X0, x1 - X0))
        B[y0:y1, x0:x1] = blank[sl]
        gold = t2 & (rb > rbref + 14)
        gold = mask_filter(gold.astype(np.float32), ImageFilter.GaussianBlur(0.8))
        text = mask_filter(t2.astype(np.float32), ImageFilter.MaxFilter(3))
        info[name] = {'text': text[sl], 'gold': gold[sl]}
    return B, info


def harmonic_fill(reg, mask, init, iters):
    """Solve Laplace's equation inside mask (Jacobi), boundary = image."""
    f = np.where(mask[..., None], init, reg).astype(np.float32)
    m3 = np.repeat(mask[..., None], 3, axis=2)
    for _ in range(iters):
        p = np.pad(f, ((1, 1), (1, 1), (0, 0)), mode='edge')
        avg = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) * 0.25
        np.copyto(f, avg, where=m3)
    return f


def star_sprite(size):
    r = size
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    d = np.sqrt(x * x + y * y) / r
    core = np.exp(-(d / 0.16) ** 2)
    halo = 0.35 * np.exp(-(d / 0.45) ** 2)
    ray_h = np.exp(-(y / (0.045 * r)) ** 2) * np.clip(1 - np.abs(x) / r, 0, 1) ** 2.2
    ray_v = np.exp(-(x / (0.045 * r)) ** 2) * np.clip(1 - np.abs(y) / r, 0, 1) ** 2.2
    u, v = (x + y) / np.sqrt(2), (x - y) / np.sqrt(2)
    diag = 0.35 * (np.exp(-(v / (0.04 * r)) ** 2) * np.clip(1 - np.abs(u) / (0.55 * r), 0, 1) ** 2
                   + np.exp(-(u / (0.04 * r)) ** 2) * np.clip(1 - np.abs(v) / (0.55 * r), 0, 1) ** 2)
    return np.clip(core + halo + ray_h + ray_v + diag, 0, 1.4)


def disc_sprite(r):
    y, x = np.mgrid[-2 * r:2 * r + 1, -2 * r:2 * r + 1].astype(np.float32)
    d = np.sqrt(x * x + y * y) / r
    return np.clip(1.15 - d, 0, 1) ** 1.6 * 0.85 + 0.15 * np.exp(-(d / 1.2) ** 2)


def add_sprite(img, spr, cx, cy, color, alpha):
    h, w = spr.shape
    x0, y0 = int(round(cx)) - w // 2, int(round(cy)) - h // 2
    ax0, ay0 = max(x0, 0), max(y0, 0)
    ax1, ay1 = min(x0 + w, img.shape[1]), min(y0 + h, img.shape[0])
    if ax0 >= ax1 or ay0 >= ay1:
        return
    s = spr[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]
    img[ay0:ay1, ax0:ax1] += (s * alpha)[..., None] * color


def camera_at(t):
    for (ta, *a), (tb, *b) in zip(CAMERA, CAMERA[1:]):
        if ta <= t <= tb:
            k = ease_io((t - ta) / (tb - ta))
            cx, cy, w = [pa + (pb - pa) * k for pa, pb in zip(a, b)]
            break
    else:
        cx, cy, w = CAMERA[-1][1:]
    h = w * OH / OW
    cx = min(max(cx, w / 2), SRC_W - w / 2)
    cy = min(max(cy, h / 2), SRC_H - h / 2)
    return cx - w / 2, cy - h / 2, w, h


G = {}


def setup(src):
    S = np.asarray(Image.open(src).convert('RGB'), np.float32)
    B, info = build_plates(S)
    G['S'], G['B'], G['D'], G['info'] = S, B, S - B, info
    G['el'] = {e[0]: e for e in ELEMENTS}
    rng = np.random.default_rng(18)
    sparks = []
    for names, count, t0, t1 in SPARKLES:
        cands = []
        for nm in names:
            x0, y0, *_ = G['el'][nm][1]
            ys, xs = np.nonzero(info[nm]['gold'] > 0.6)
            cands += list(zip(xs + x0, ys + y0))
        if not cands:
            continue
        for i in range(count):
            x, y = cands[rng.integers(len(cands))]
            st = t0 + (t1 - t0 - 0.8) * (i + rng.uniform(0, 1)) / count
            sparks.append((float(x), float(y), st, rng.uniform(0.6, 0.95), int(rng.integers(9, 20))))
    G['sparks'] = sparks
    G['star'] = {s: star_sprite(s) for s in range(9, 20)}
    dust = []
    for _ in range(26):
        dust.append(dict(x=rng.uniform(0, OW), y=rng.uniform(0, OH), vy=rng.uniform(14, 38),
                         amp=rng.uniform(8, 30), fq=rng.uniform(0.08, 0.25), ph=rng.uniform(0, 6.28),
                         r=int(rng.integers(3, 11)), a=rng.uniform(0.06, 0.2), tw=rng.uniform(0.3, 0.9)))
    G['dust'] = dust
    G['disc'] = {r: disc_sprite(r) for r in range(3, 11)}


def reveal_mask(mode, shape, p, bbox):
    h, w = shape
    if mode == 'fade':
        return np.full(shape, smooth(p), np.float32), None
    xs = np.arange(w, dtype=np.float32)[None, :]
    if mode == 'rtl':
        F = 46.0
        e = (w - 1) - p * (w + F)
        m = np.clip((xs - e) / F, 0, 1)
        edge = np.exp(-((xs - e - F * 0.35) / 22.0) ** 2)
        return np.broadcast_to(m, shape), np.broadcast_to(edge, shape)
    ys = np.arange(h, dtype=np.float32)[:, None]
    if mode == 'center':
        F = 50.0
        d = np.abs(xs - w / 2) + 0 * ys
        rad = p * (w / 2 + F)
    else:  # radial
        F = 70.0
        cx, cy = 320 - bbox[0], 340 - bbox[1]
        d = np.sqrt((xs - cx) ** 2 + ((ys - cy) * 1.15) ** 2)
        rad = p * (np.max(d) + F)
    m = np.clip((rad - d) / F, 0, 1)
    edge = np.exp(-((d - rad + F * 0.35) / 24.0) ** 2)
    return m, edge


def render_frame(i):
    t = i / FPS
    S, B, D, info = G['S'], G['B'], G['D'], G['info']
    frame = B.copy()
    for name, bbox, st, dur, mode, _gold in ELEMENTS:
        p = (t - st) / dur
        if p <= 0:
            continue
        x0, y0, x1, y1 = bbox
        if p >= 1:
            frame[y0:y1, x0:x1] = S[y0:y1, x0:x1]
            continue
        p = float(smooth(p) * 0.35 + p * 0.65)
        m, edge = reveal_mask(mode, (y1 - y0, x1 - x0), p, bbox)
        region = B[y0:y1, x0:x1] + D[y0:y1, x0:x1] * m[..., None]
        if edge is not None:
            region = region + (edge * info[name]['text'] * 0.75)[..., None] * (GLOW - region) * 0.9
        frame[y0:y1, x0:x1] = region
    for names, st, dur in SHIMMERS:
        p = (t - st) / dur
        if not 0 < p < 1:
            continue
        for nm in names:
            x0, y0, x1, y1 = G['el'][nm][1]
            h, w = y1 - y0, x1 - x0
            ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            if len(names) > 1:  # one diagonal sweep across the whole picture
                u = xs + 0.55 * ys
                pos = -200 + p * (1080 + 0.55 * 1620 + 400)
                wid = 70.0
            else:
                u = (xs - x0) + 0.55 * (ys - y0)
                pos = -60 + p * (w + 0.55 * h + 120)
                wid = 38.0
            band = np.exp(-((u - pos) / wid) ** 2) * info[nm]['gold'] * 0.85
            frame[y0:y1, x0:x1] += band[..., None] * (SPARK - frame[y0:y1, x0:x1]) * 0.85
    for x, y, st, amp, size in G['sparks']:
        a = (t - st) / 0.8
        if 0 < a < 1:
            env = np.sin(np.pi * a) ** 1.5 * amp
            add_sprite(frame, G['star'][size], x, y, SPARK, env)
    np.clip(frame, 0, 255, out=frame)
    img = Image.fromarray(frame.astype(np.uint8))
    cx0, cy0, cw, ch = camera_at(t)
    s = cw / OW
    out = img.transform((OW, OH), Image.AFFINE,
                        (s, 0, cx0 + 0.5 * s - 0.5, 0, ch / OH, cy0 + 0.5 * ch / OH - 0.5),
                        resample=Image.BICUBIC)
    if s < 0.95:
        out = out.filter(ImageFilter.UnsharpMask(radius=1.4, percent=int(70 * (1 - s) / 0.4 + 10), threshold=2))
    o = np.asarray(out, np.float32).copy()
    dust_fade = smooth(t / 2.0) * smooth((DURATION - t) / 1.5)
    for d in G['dust']:
        y = (d['y'] - d['vy'] * t) % (OH + 60) - 30
        x = d['x'] + d['amp'] * np.sin(2 * np.pi * d['fq'] * t + d['ph'])
        tw = 0.65 + 0.35 * np.sin(2 * np.pi * d['tw'] * t + d['ph'])
        add_sprite(o, G['disc'][d['r']], x, y, GLOW, d['a'] * tw * dust_fade)
    # soft vignette
    if 'vig' not in G:
        yy, xx = np.mgrid[0:OH, 0:OW].astype(np.float32)
        r = np.sqrt(((xx - OW / 2) / (OW / 2)) ** 2 + ((yy - OH / 2) / (OH / 2)) ** 2)
        G['vig'] = (1 - 0.14 * smooth((r - 0.75) / 0.7))[..., None]
    o *= G['vig']
    k = smooth(t / 1.3) * smooth((DURATION - t) / 1.1)
    o = CREAM + (o - CREAM) * k
    return np.clip(o, 0, 255).astype(np.uint8).tobytes()


def main():
    src, music, out = sys.argv[1:4]
    setup(src)
    if len(sys.argv) > 4 and sys.argv[4] == '--preview':
        for tt in sys.argv[5].split(','):
            i = int(float(tt) * FPS)
            Image.frombytes('RGB', (OW, OH), render_frame(i)).save(f'{out}_{tt}.jpg', quality=90)
        Image.fromarray(np.clip(G['B'], 0, 255).astype(np.uint8)).save(f'{out}_blank.jpg', quality=92)
        return
    n = int(DURATION * FPS)
    ff = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{OW}x{OH}', '-r', str(FPS), '-i', '-',
        '-i', music,
        '-c:v', 'libx264', '-preset', 'slow', '-crf', '17', '-pix_fmt', 'yuv420p',
        '-profile:v', 'high', '-tune', 'film',
        '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', out,
    ], stdin=subprocess.PIPE)
    with mp.get_context('fork').Pool(mp.cpu_count()) as pool:
        for k, buf in enumerate(pool.imap(render_frame, range(n), chunksize=4)):
            ff.stdin.write(buf)
            if k % 75 == 0:
                print(f'frame {k}/{n}', flush=True)
    ff.stdin.close()
    ff.wait()
    print('done', out)


if __name__ == '__main__':
    main()
