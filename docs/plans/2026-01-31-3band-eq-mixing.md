# 3-Band EQ Mixing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace simple crossfade with professional 3-band EQ mixing using phrase-based timing.

**Architecture:** Split audio into low/mid/high bands using butterworth filters, apply per-band gain envelopes synced to 16-bar and 8-bar phrase boundaries, recombine for final mix.

**Tech Stack:** Python, numpy, librosa, scipy (new dependency for filters)

---

### Task 1: Add scipy dependency

**Files:**
- Create: `requirements.txt`

**Step 1: Create requirements file**

```
numpy
librosa
scipy
```

**Step 2: Install scipy**

Run: `source venv/bin/activate && pip install scipy`
Expected: Successfully installed scipy

**Step 3: Verify import works**

Run: `source venv/bin/activate && python -c "from scipy.signal import butter, sosfilt; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add requirements.txt
git commit -m "Add requirements.txt with scipy for EQ filtering"
```

---

### Task 2: Add band-splitting filter functions

**Files:**
- Modify: `dj_mix.py:1-28` (add imports and constants)
- Modify: `dj_mix.py` (add new functions after line 100)

**Step 1: Add scipy import and EQ constants**

At top of file after existing imports (line 15), add:

```python
from scipy.signal import butter, sosfilt
```

After `HOP_LENGTH = 128` (line 28), add:

```python
# EQ crossover frequencies (Hz)
LOW_CUTOFF = 250
HIGH_CUTOFF = 2500
FILTER_ORDER = 4  # 24dB/octave Linkwitz-Riley
```

**Step 2: Add filter creation functions**

After `time_stretch_stereo` function (after line 100), add:

```python
def create_lowpass(cutoff, sr, order=FILTER_ORDER):
    """Create lowpass filter coefficients."""
    nyq = sr / 2
    normalized_cutoff = cutoff / nyq
    sos = butter(order, normalized_cutoff, btype='low', output='sos')
    return sos


def create_highpass(cutoff, sr, order=FILTER_ORDER):
    """Create highpass filter coefficients."""
    nyq = sr / 2
    normalized_cutoff = cutoff / nyq
    sos = butter(order, normalized_cutoff, btype='high', output='sos')
    return sos


def create_bandpass(lowcut, highcut, sr, order=FILTER_ORDER):
    """Create bandpass filter coefficients."""
    nyq = sr / 2
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos


def apply_filter(audio, sos):
    """Apply filter to stereo audio."""
    left = sosfilt(sos, audio[:, 0])
    right = sosfilt(sos, audio[:, 1])
    return np.column_stack([left, right])


def split_to_bands(audio, sr):
    """Split audio into low, mid, high frequency bands."""
    low_sos = create_lowpass(LOW_CUTOFF, sr)
    mid_sos = create_bandpass(LOW_CUTOFF, HIGH_CUTOFF, sr)
    high_sos = create_highpass(HIGH_CUTOFF, sr)

    return {
        'low': apply_filter(audio, low_sos),
        'mid': apply_filter(audio, mid_sos),
        'high': apply_filter(audio, high_sos),
    }
```

**Step 3: Test filter functions manually**

Run: `source venv/bin/activate && python -c "
import numpy as np
from dj_mix import split_to_bands
# Create test signal: 100Hz + 1000Hz + 5000Hz
sr = 44100
t = np.linspace(0, 1, sr)
mono = np.sin(2*np.pi*100*t) + np.sin(2*np.pi*1000*t) + np.sin(2*np.pi*5000*t)
stereo = np.column_stack([mono, mono])
bands = split_to_bands(stereo, sr)
print(f'Low band shape: {bands[\"low\"].shape}')
print(f'Mid band shape: {bands[\"mid\"].shape}')
print(f'High band shape: {bands[\"high\"].shape}')
print('OK')
"`

Expected: Three band shapes matching input, prints `OK`

**Step 4: Commit**

```bash
git add dj_mix.py
git commit -m "Add 3-band EQ filter functions"
```

---

### Task 3: Add phrase timing calculation

**Files:**
- Modify: `dj_mix.py` (add function after `split_to_bands`)

**Step 1: Add phrase calculation function**

After `split_to_bands` function, add:

```python
def calculate_phrase_points(blend_start, master_interval, transition_duration):
    """
    Calculate phrase-aligned timing points for EQ transitions.

    Returns dict with sample-relative timing (seconds from blend_start):
    - bass_swap: 16-bar boundary for bass handoff
    - high_swap: 8-bar boundary (offset from bass) for highs handoff
    - hard_cut_start: 16-bar boundary near end for final cut
    - hard_cut_end: 1-2 bars after hard_cut_start
    """
    bar_length = 4 * master_interval  # 4 beats per bar
    phrase_8 = 8 * bar_length
    phrase_16 = 16 * bar_length

    # Bass swaps at middle 16-bar boundary
    mid_point = transition_duration / 2
    bass_swap = round(mid_point / phrase_16) * phrase_16
    if bass_swap == 0:
        bass_swap = phrase_16

    # Highs swap at 8-bar boundary, offset from bass
    # First swap is at 8 bars (before bass swap at 16)
    high_swap = phrase_8

    # Hard cut at last 16-bar boundary that fits, with 2 bars fade
    last_phrase_16 = int(transition_duration / phrase_16) * phrase_16
    if last_phrase_16 <= bass_swap:
        last_phrase_16 = bass_swap + phrase_16
    if last_phrase_16 > transition_duration:
        last_phrase_16 = transition_duration - (2 * bar_length)

    hard_cut_start = last_phrase_16
    hard_cut_end = hard_cut_start + (2 * bar_length)  # 2 bars fade

    return {
        'bass_swap': bass_swap,
        'high_swap': high_swap,
        'hard_cut_start': hard_cut_start,
        'hard_cut_end': min(hard_cut_end, transition_duration),
        'bar_length': bar_length,
        'phrase_8': phrase_8,
        'phrase_16': phrase_16,
    }
```

**Step 2: Test phrase calculation**

Run: `source venv/bin/activate && python -c "
from dj_mix import calculate_phrase_points
# 130 BPM = 0.4615s interval, 30s transition
interval = 60.0 / 130
points = calculate_phrase_points(0, interval, 30.0)
print(f'Bar length: {points[\"bar_length\"]:.2f}s')
print(f'8-bar phrase: {points[\"phrase_8\"]:.2f}s')
print(f'16-bar phrase: {points[\"phrase_16\"]:.2f}s')
print(f'Bass swap at: {points[\"bass_swap\"]:.2f}s')
print(f'High swap at: {points[\"high_swap\"]:.2f}s')
print(f'Hard cut: {points[\"hard_cut_start\"]:.2f}s - {points[\"hard_cut_end\"]:.2f}s')
print('OK')
"`

Expected: Reasonable phrase timings, `OK`

**Step 3: Commit**

```bash
git add dj_mix.py
git commit -m "Add phrase timing calculation for EQ transitions"
```

---

### Task 4: Add EQ mix function

**Files:**
- Modify: `dj_mix.py` (add main mixing function after `calculate_phrase_points`)

**Step 1: Add the mix_with_eq function**

After `calculate_phrase_points`, add:

```python
def mix_with_eq(outgoing_audio, incoming_audio, outgoing_start_sample,
                incoming_start_sample, blend_start_sample, blend_duration_samples,
                sr, master_interval):
    """
    Mix two tracks using 3-band EQ with phrase-based timing.

    Returns the mixed audio for the transition region.
    """
    blend_duration_sec = blend_duration_samples / sr
    timing = calculate_phrase_points(0, master_interval, blend_duration_sec)

    # Split both tracks into bands
    out_bands = split_to_bands(outgoing_audio, sr)
    in_bands = split_to_bands(incoming_audio, sr)

    # Pre-calculate timing in samples
    bass_swap_sample = int(timing['bass_swap'] * sr)
    high_swap_sample = int(timing['high_swap'] * sr)
    hard_cut_start_sample = int(timing['hard_cut_start'] * sr)
    hard_cut_end_sample = int(timing['hard_cut_end'] * sr)
    bar_samples = int(timing['bar_length'] * sr)
    beat_samples = bar_samples // 4

    # Build output array
    output = np.zeros((blend_duration_samples, 2), dtype=np.float64)

    for i in range(blend_duration_samples):
        blend_pos = i  # Position within blend region

        out_pos = (blend_start_sample + i) - outgoing_start_sample
        in_pos = (blend_start_sample + i) - incoming_start_sample

        # Get samples from each band (with bounds checking)
        out_low = out_bands['low'][out_pos] if 0 <= out_pos < len(out_bands['low']) else np.zeros(2)
        out_mid = out_bands['mid'][out_pos] if 0 <= out_pos < len(out_bands['mid']) else np.zeros(2)
        out_high = out_bands['high'][out_pos] if 0 <= out_pos < len(out_bands['high']) else np.zeros(2)

        in_low = in_bands['low'][in_pos] if 0 <= in_pos < len(in_bands['low']) else np.zeros(2)
        in_mid = in_bands['mid'][in_pos] if 0 <= in_pos < len(in_bands['mid']) else np.zeros(2)
        in_high = in_bands['high'][in_pos] if 0 <= in_pos < len(in_bands['high']) else np.zeros(2)

        # === BASS: swap at 16-bar boundary, no overlap ===
        if blend_pos < bass_swap_sample:
            out_low_gain = 1.0
            in_low_gain = 0.0
        else:
            out_low_gain = 0.0
            in_low_gain = 1.0

        # === HIGHS: swap at 8-bar boundary (offset from bass) ===
        if blend_pos < high_swap_sample:
            out_high_gain = 1.0
            in_high_gain = 0.0
        else:
            out_high_gain = 0.0
            in_high_gain = 1.0

        # === MIDS: gradual blend ===
        # Outgoing: 100% -> 30% over transition, then hard cut over 2 beats
        # Incoming: 0% -> 30% over 4 beats, then rise to 100%

        progress = blend_pos / blend_duration_samples

        # Incoming mid: 0% -> 30% over first 4 beats, then to 100%
        four_beats = 4 * beat_samples
        if blend_pos < four_beats:
            in_mid_gain = 0.3 * (blend_pos / four_beats)
        else:
            # Rise from 30% to 100% over remaining transition
            remaining_progress = (blend_pos - four_beats) / (blend_duration_samples - four_beats)
            in_mid_gain = 0.3 + 0.7 * remaining_progress

        # Outgoing mid: 100% -> 30%, then hard cut over 2 beats at end
        two_beats = 2 * beat_samples
        if blend_pos < hard_cut_start_sample:
            # Fade from 100% to 30%
            fade_progress = blend_pos / hard_cut_start_sample
            out_mid_gain = 1.0 - 0.7 * fade_progress
        elif blend_pos < hard_cut_end_sample:
            # Hard cut from 30% to 0% over 2 beats
            cut_progress = (blend_pos - hard_cut_start_sample) / (hard_cut_end_sample - hard_cut_start_sample)
            out_mid_gain = 0.3 * (1.0 - cut_progress)
        else:
            out_mid_gain = 0.0

        # Apply hard cut to all outgoing bands at the end
        if blend_pos >= hard_cut_start_sample:
            cut_progress = (blend_pos - hard_cut_start_sample) / (hard_cut_end_sample - hard_cut_start_sample)
            cut_progress = min(1.0, cut_progress)
            out_low_gain *= (1.0 - cut_progress)
            out_high_gain *= (1.0 - cut_progress)

        # Combine
        sample = (out_low * out_low_gain + in_low * in_low_gain +
                  out_mid * out_mid_gain + in_mid * in_mid_gain +
                  out_high * out_high_gain + in_high * in_high_gain)

        output[i] = sample

    return output
```

**Step 2: Verify function compiles**

Run: `source venv/bin/activate && python -c "from dj_mix import mix_with_eq; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add dj_mix.py
git commit -m "Add mix_with_eq function for 3-band EQ transitions"
```

---

### Task 5: Integrate EQ mixing into main function

**Files:**
- Modify: `dj_mix.py:278-309` (replace crossfade loop)

**Step 1: Replace the crossfade section**

Find this code block (lines 278-309):

```python
        blend_start_sample = int(blend_beat_time * sr)
        t = np.linspace(0, 1, crossfade_samples).reshape(-1, 1)
        fade_out = np.sqrt(1 - t)
        fade_in = np.sqrt(t)

        blend_end_sample = blend_start_sample + crossfade_samples

        for j in range(crossfade_samples):
            pos = blend_start_sample + j
            if pos >= len(mix):
                break

            out_pos = pos - current_track_start
            in_pos = pos - incoming_start_sample

            sample = np.zeros(2)
            if 0 <= out_pos < len(outgoing['audio_stretched']):
                sample += outgoing['audio_stretched'][out_pos] * fade_out[j, 0]
            if 0 <= in_pos < len(incoming['audio_stretched']):
                sample += incoming['audio_stretched'][in_pos] * fade_in[j, 0]

            mix[pos] = sample
```

Replace with:

```python
        blend_start_sample = int(blend_beat_time * sr)
        blend_end_sample = blend_start_sample + crossfade_samples

        print(f"  Mixing with 3-band EQ...")
        eq_mixed = mix_with_eq(
            outgoing['audio_stretched'],
            incoming['audio_stretched'],
            current_track_start,
            incoming_start_sample,
            blend_start_sample,
            crossfade_samples,
            sr,
            master_interval
        )

        # Apply the EQ-mixed transition to the main mix
        for j in range(len(eq_mixed)):
            pos = blend_start_sample + j
            if pos < len(mix):
                mix[pos] = eq_mixed[j]
```

**Step 2: Test full mix generation**

Run: `source venv/bin/activate && python dj_mix.py`

Expected: Mix completes with "Mixing with 3-band EQ..." messages for each transition

**Step 3: Commit**

```bash
git add dj_mix.py
git commit -m "Integrate 3-band EQ mixing into main mix function"
```

---

### Task 6: Update CLAUDE.md with EQ mixing notes

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add EQ mixing section**

After the "### Crossfade" section, add:

```markdown
### 3-Band EQ Mixing
- Use scipy butterworth filters (24dB/octave) for clean band separation
- Crossovers: Low 0-250Hz, Mid 250-2500Hz, High 2500Hz+
- Bass swaps on 16-bar boundaries (never both tracks at once)
- Highs swap on 8-bar boundaries, offset from bass
- Mids blend gradually with hard cut at end
```

**Step 2: Update dependencies section**

Change:
```bash
pip install numpy librosa
```

To:
```bash
pip install numpy librosa scipy
```

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "Document 3-band EQ mixing in CLAUDE.md"
```

---

## Summary

6 tasks total:
1. Add scipy dependency
2. Add band-splitting filter functions
3. Add phrase timing calculation
4. Add EQ mix function
5. Integrate into main function
6. Update documentation
