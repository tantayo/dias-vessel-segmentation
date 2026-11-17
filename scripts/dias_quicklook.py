"""Quick exploratory plots for the DIAS DSA dataset.

Images: image_s<S>_i<F>.png = frame F from sequence S.
Labels: label_s<S>.png = ONE reference mask for the entire sequence.

No training, GT leakage, or claim of clinically validated vessels:
  D[t] = Gaussian(I[t], BG_SIGMA) - Gaussian(I[t], NOISE_SIGMA)
  E[t] = max(0, D[0], ..., D[t])       (accumulated local contrast)
  T[t] = max_{k<=t} [I[0]-I[k]-median(I[0]-I[k])]_+
          (cumulative temporal darkening; smooth spatially if desired)
  S[t] = E[t] + temporal_weight * T[t]

  Compare E[t] alone vs S[t] = E[t] + alpha*T[t] using the SAME two
  numeric thresholds but INDEPENDENT strong seeds for each score:
    row 2: strong = score > high
    row 3: strong + connected weaker pixels where score > low
  Grow using 8-connectivity (hysteresis). No size-filter or gap bridging.
  Effective high threshold is max(high, low), to preserve all strong pixels.

Neither score is a calibrated probability. GT is for display/evaluation ONLY.
Arrows = previous/next; Space = play/pause; toolbar zoom/pan is linked.
'r' = reset zoom. All frames are discovered automatically.
"""
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

# -------------------------- Edit these ------------------------------------
DATA_DIR = Path('data/DIAS')  # Update to your local DIAS dataset path
SPLIT = 'training'                # training / validation / test / unlabeled_DSA
SEQUENCE_ID = 0                  # None = first available

BG_SIGMA = 50.0                  # Background Gaussian scale, in pixels
NOISE_SIGMA = 1.0                # 0: D=B-raw; >0: DoG
TEMPORAL_SMOOTH_SIGMA = 0.65     # Spatial smoothing of frame-to-first change

HIGH_THRESHOLD = 30.0           # Strong candidate seed threshold
LOW_THRESHOLD = 30.0            # Connected weaker candidate threshold
TEMPORAL_WEIGHT = 0.75         # S=E+TEMPORAL_WEIGHT*T, 0 disables time term

CONTRAST = (1, 99)              # Shared DISPLAY stretch only
FPS = 3                         # Playback rate, not acquisition frame rate
# --------------------------------------------------------------------------

PATTERN = re.compile(r'^image_s(\d+)_i(\d+)\.png$', re.I)
STRUCTURE = np.ones((3, 3), dtype=bool)


def find_sequences(folder):
    found = {}
    for p in folder.glob('image_s*_i*.png'):
        m = PATTERN.fullmatch(p.name)
        if m:
            sid, frame = map(int, m.groups())
            found.setdefault(sid, []).append((frame, p))
    return {sid: [p for _, p in sorted(items)]
            for sid, items in sorted(found.items())}


def read_image(path):
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        # DIAS scans are monochrome; RGB copies are assumed replicated grayscale.
        arr = arr[..., :3].astype(np.float32).mean(axis=-1)
    return arr.astype(np.float32)


def read_mask(path):
    arr = np.asarray(Image.open(path))
    return (np.any(arr[..., :3] > 0, axis=-1) if arr.ndim == 3
            else arr > 0).astype(bool)


def overlay(gray, label):
    rgb = np.repeat(gray[..., None], 3, axis=2)
    if label is not None:
        rgb[label] = 0.55 * rgb[label] + 0.45 * np.array([0, 1, 0])
    return rgb


def compare_with_gt(pred, label):
    rgb = np.zeros((*pred.shape, 3), dtype=np.float32)
    if label is not None:
        pred = pred.astype(bool)
        rgb[pred & label] = (0, 1, 0)       # true positives
        rgb[pred & ~label] = (1, 0, 0)      # apparent false positives
        rgb[~pred & label] = (0, .3, 1)     # false negatives
    return rgb


def metrics(pred, label):
    if label is None:
        return 'no GT'
    tp = np.count_nonzero(pred & label)
    fp = np.count_nonzero(pred & ~label)
    fn = np.count_nonzero(~pred & label)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    return f'Dice {dice:.3f}  P {precision:.3f}  R {recall:.3f}'


def adaptive_hysteresis(score, low, high):
    """Return strong seeds and 8-connected low-threshold growth (no GT used)."""
    effective_high = max(high, low)
    strong = score > effective_high
    weak = score > low
    accepted = ndi.binary_propagation(strong, structure=STRUCTURE, mask=weak)
    return accepted, strong


# ----------------------------- Load ---------------------------------------
print('DIAS overview:')
for name in ('training', 'validation', 'test', 'unlabeled_DSA'):
    root = DATA_DIR / name
    image_dir = root if name == 'unlabeled_DSA' else root / 'images'
    groups = find_sequences(image_dir)
    if groups:
        lengths = [len(v) for v in groups.values()]
        print(f'  {name:13}: {len(groups):3} sequences, '
              f'{sum(lengths):4} frames, {min(lengths)}–{max(lengths)} frames/seq')

root = DATA_DIR / SPLIT
image_dir = root if SPLIT == 'unlabeled_DSA' else root / 'images'
sequences = find_sequences(image_dir)
if not sequences:
    raise FileNotFoundError(f'No DIAS frames in {image_dir}')
sid = min(sequences) if SEQUENCE_ID is None else SEQUENCE_ID
if sid not in sequences:
    raise ValueError(f's{sid} not found. Choices: {list(sequences)}')
paths = sequences[sid]
frames = np.stack([read_image(p) for p in paths])
n_frames, height, width = frames.shape
label_path = root / 'labels' / f'label_s{sid}.png'
mask = read_mask(label_path) if label_path.is_file() else None
if mask is not None and mask.shape != (height, width):
    raise ValueError(f'GT {mask.shape} does not match image {(height, width)}')

# --------------------------- Signals --------------------------------------
# All computations use raw pixel intensities. No per-frame normalization.
background = ndi.gaussian_filter(frames, sigma=(0, BG_SIGMA, BG_SIGMA))
smoothed = (ndi.gaussian_filter(frames, sigma=(0, NOISE_SIGMA, NOISE_SIGMA))
            if NOISE_SIGMA > 0 else frames)
D = background - smoothed
E = np.maximum.accumulate(np.maximum(D, 0.0), axis=0)

# Relative-to-first darkening; subtract a framewise median to remove
# camera-wide brightness shifts. Validity assumes frames are aligned.
changes = frames[0][None] - frames
brightness_offset = np.median(changes.reshape(n_frames, -1), axis=1)
changes = changes - brightness_offset[:, None, None]
if TEMPORAL_SMOOTH_SIGMA > 0:
    changes = ndi.gaussian_filter(changes,
                                  sigma=(0, TEMPORAL_SMOOTH_SIGMA,
                                         TEMPORAL_SMOOTH_SIGMA))
T = np.maximum.accumulate(np.maximum(changes, 0), axis=0)

print("Max T at frame 0:", T[0].max())
print("Max S-E at frame 0:",
      np.abs(E[0] + TEMPORAL_WEIGHT*T[0] - E[0]).max())

lo, hi = np.percentile(frames, CONTRAST)
scale = max(float(hi - lo), 1e-6)
show_raw = lambda x: np.clip((x - lo) / scale, 0, 1)
frames_display = show_raw(frames)
overlays = np.stack([overlay(x, mask) for x in frames_display])

# Set fixed display limits across all frames and all slider positions.
d_lim = max(float(np.percentile(np.abs(D), 99.5)), 1e-6)
e_lim = max(float(np.percentile(E, 99.5)), 1e-6)
t_lim = max(float(np.percentile(T, 99.5)), 1e-6)
s_lim = max(float(np.percentile(E + 2 * T, 99.5)), 1e-6)
max_signal = max(float(np.percentile(np.maximum(E, 0), 99.9)),
                 float(np.percentile(np.maximum(E + 2*T, 0), 99.9)),
                 HIGH_THRESHOLD * 1.5, LOW_THRESHOLD * 1.5, 1.0)

print(f'\nDIAS {SPLIT}/s{sid}: {n_frames} frames, size {height}x{width}')
print('Frames:', ', '.join(p.name for p in paths))
print(f'Raw intensity range [{frames.min():g}, {frames.max():g}]')
print('D: local dark contrast; E: accumulated D; T: new darkening vs first frame')
print('Compare E and S=E+weight*T, each with its OWN strong seeds and shared numeric thresholds.')
print(f'GT: {label_path.name} (entire sequence)' if mask is not None else 'No GT')

# ------------------------ Interactive viewer ------------------------------
# Column 1: source/reference/individual signals
# Column 2: E only, local accumulated contrast
# Column 3: E + alpha*T, including temporal darkening
# Both model columns use the same numerical thresholds, but DIFFERENT seeds.
fig, ax = plt.subplots(4, 3, figsize=(17, 14.5), sharex=True, sharey=True)
fig.subplots_adjust(left=0.025, right=0.975, top=0.925,
                    bottom=0.230, hspace=0.25, wspace=0.06)

# Shared color scale so an E vs S brightness difference has a consistent meaning.
score_lim = max(float(np.percentile(E + 2*T, 99.5)), 1e-6)
titles = [
    'Original DSA frame', 'E: accumulated local contrast', 'S = E + alpha*T: combined evidence',
    'Sequence-level GT overlay', 'E > high: strong seeds', 'S > high: strong seeds',
    'Current local contrast D', 'E-only: seeded hysteresis', 'S: seeded hysteresis',
    'Temporal darkening T', 'E-only vs GT', 'S vs GT',
]
styles = [
    dict(cmap='gray', vmin=0, vmax=1),
    dict(cmap='inferno', vmin=0, vmax=score_lim),
    dict(cmap='inferno', vmin=0, vmax=score_lim),
    {},
    dict(cmap='gray', vmin=0, vmax=1),
    dict(cmap='gray', vmin=0, vmax=1),
    dict(cmap='RdBu_r', vmin=-d_lim, vmax=d_lim),
    dict(cmap='gray', vmin=0, vmax=1),
    dict(cmap='gray', vmin=0, vmax=1),
    dict(cmap='magma', vmin=0, vmax=t_lim),
    {}, {},
]


def make_panels(i, low, high, weight):
    e = E[i]
    s = e + weight * T[i]
    e_hyst, e_seeds = adaptive_hysteresis(e, low, high)
    s_hyst, s_seeds = adaptive_hysteresis(s, low, high)
    arrays = (
        frames_display[i], e, s,
        overlays[i], e_seeds, s_seeds,
        D[i], e_hyst, s_hyst,
        T[i], compare_with_gt(e_hyst, mask), compare_with_gt(s_hyst, mask),
    )
    return arrays, e_seeds, s_seeds, e_hyst, s_hyst


initial, *_ = make_panels(0, LOW_THRESHOLD, HIGH_THRESHOLD,
                          TEMPORAL_WEIGHT)
artists = []
for a, title, style, data in zip(ax.flat, titles, styles, initial):
    artists.append(a.imshow(data, interpolation='nearest', **style))
    a.set_title(title, fontsize=9)
    a.axis('off')
caption = fig.suptitle('')

# Widget controls
prev_a = fig.add_axes([0.047, .175, .11, .034])
next_a = fig.add_axes([0.172, .175, .11, .034])
play_a = fig.add_axes([0.295, .175, .11, .034])
frame_a = fig.add_axes([0.59, .183, .35, .016])
high_a = fig.add_axes([0.185, .126, .755, .016])
low_a = fig.add_axes([0.185, .081, .755, .016])
weight_a = fig.add_axes([0.185, .036, .755, .016])

prev_btn = Button(prev_a, '< Previous')
next_btn = Button(next_a, 'Next >')
play_btn = Button(play_a, 'Play')
frame_slider = Slider(frame_a, 'Frame', 0, max(1, n_frames - 1),
                      valinit=0, valstep=1, valfmt='%0.0f')
high_slider = Slider(high_a, 'Strong seed >', 0, max_signal,
                     valinit=HIGH_THRESHOLD, valfmt='%0.1f')
low_slider = Slider(low_a, 'Weak path >', 0, max_signal,
                    valinit=LOW_THRESHOLD, valfmt='%0.1f')
weight_slider = Slider(weight_a, 'Temporal weight', 0, 2,
                       valinit=TEMPORAL_WEIGHT, valfmt='%0.2f')

timer = fig.canvas.new_timer(interval=max(1, round(1000 / FPS)))
playing = False


def update(_=None):
    i = max(0, min(n_frames - 1, int(frame_slider.val)))
    low = float(low_slider.val)
    high = float(high_slider.val)
    weight = float(weight_slider.val)
    arrays, e_seeds, s_seeds, e_hyst, s_hyst = make_panels(
        i, low, high, weight
    )
    for artist, arr in zip(artists, arrays):
        artist.set_data(arr)
    ax[1, 1].set_title('E strong: ' + metrics(e_seeds, mask), fontsize=8.5)
    ax[1, 2].set_title('S strong: ' + metrics(s_seeds, mask), fontsize=8.5)
    ax[2, 1].set_title('E strong + connected weak', fontsize=9)
    ax[2, 2].set_title('S strong + connected weak', fontsize=9)
    ax[3, 1].set_title('E growth vs GT: ' + metrics(e_hyst, mask), fontsize=8.5)
    ax[3, 2].set_title('S growth vs GT: ' + metrics(s_hyst, mask), fontsize=8.5)
    caption.set_text(f'DIAS {SPLIT}/s{sid} | {paths[i].name} | {i+1}/{n_frames} | '
                     f'seeds E/S={np.count_nonzero(e_seeds)}/'
                     f'{np.count_nonzero(s_seeds)} | '
                     f'low={low:.1f}, high={max(high, low):.1f}, alpha={weight:.2f}')
    fig.canvas.draw_idle()


def reset_zoom():
    ax[0, 0].set_xlim(-0.5, width - 0.5)
    ax[0, 0].set_ylim(height - 0.5, -0.5)
    fig.canvas.draw_idle()


def pause():
    global playing
    playing = False
    timer.stop()
    play_btn.label.set_text('Play')
    fig.canvas.draw_idle()


def step(delta):
    pause()
    frame_slider.set_val(max(0, min(n_frames - 1, int(frame_slider.val) + delta)))


def toggle_play(_=None):
    global playing
    if playing:
        pause()
    elif n_frames > 1:
        if int(frame_slider.val) == n_frames - 1:
            frame_slider.set_val(0)
        playing = True
        play_btn.label.set_text('Pause')
        timer.start()
        fig.canvas.draw_idle()


def tick():
    if playing:
        frame_slider.set_val((int(frame_slider.val) + 1) % n_frames)
    return True


def on_key(event):
    if event.key == 'left': step(-1)
    elif event.key == 'right': step(1)
    elif event.key == ' ': toggle_play()
    elif event.key == 'r': reset_zoom()
    elif event.key == 'home': pause(); frame_slider.set_val(0)
    elif event.key == 'end': pause(); frame_slider.set_val(n_frames - 1)

frame_slider.on_changed(update)
high_slider.on_changed(update)
low_slider.on_changed(update)
weight_slider.on_changed(update)
prev_btn.on_clicked(lambda _: step(-1))
next_btn.on_clicked(lambda _: step(1))
play_btn.on_clicked(toggle_play)
timer.add_callback(tick)
fig.canvas.mpl_connect('key_press_event', on_key)
fig.canvas.mpl_connect('close_event', lambda _: timer.stop())
update()
print('Controls: Zoom/Pan linked, R reset; ←/→ frames, Space play/pause.')
print('Compare E-only (middle column) vs E+alpha*T (right column).')
print('Set temporal weight to 0: both columns MUST produce identical seeds and detections.')
plt.show()