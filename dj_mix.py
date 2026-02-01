#!/usr/bin/env python3
"""
DJ Mix with Mathematical Beat Grid
- Detects tempo and first beat once per track
- Creates mathematical beat grid (not re-detecting)
- Time-stretches track 2 and scales beat grid accordingly
"""

import wave
import numpy as np
import librosa
import os

TRACK_1 = "tracks/Steinmetz - STNMTZ001 - Kraft - 01 Steinmetz - Kraft (Original Mix).wav"
TRACK_2 = "tracks/Steinmetz - STNMTZ001 - Kraft - 02 Steinmetz - Werk (Original Mix).wav"
OUTPUT = "mixes/kraft_werk_mix.wav"

BLEND_START_SECONDS = 96.0
CROSSFADE_SECONDS = 30.0
HOP_LENGTH = 128


def load_wav_float(filepath):
    with wave.open(filepath, 'rb') as w:
        n_frames = w.getnframes()
        framerate = w.getframerate()
        raw = w.readframes(n_frames)
        audio = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
        return audio.astype(np.float64) / 32768.0, framerate


def save_wav(filepath, audio, sr):
    audio = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
    with wave.open(filepath, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio.tobytes())


def detect_tempo_and_first_beat(filepath, sr):
    """Detect tempo and first beat position"""
    y, _ = librosa.load(filepath, sr=sr, mono=True)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)

    # Calculate tempo from median interval (more accurate)
    intervals = np.diff(beat_times[:100])
    actual_tempo = 60.0 / np.median(intervals)

    # First beat time
    first_beat = beat_times[0]

    return actual_tempo, first_beat


def create_beat_grid(first_beat, tempo, duration):
    """Create mathematical beat grid"""
    beat_interval = 60.0 / tempo
    n_beats = int(duration / beat_interval) + 10
    beat_times = first_beat + np.arange(n_beats) * beat_interval
    return beat_times


def time_stretch_stereo(audio, rate):
    """Time-stretch stereo audio"""
    left = librosa.effects.time_stretch(audio[:, 0].astype(np.float32), rate=rate)
    right = librosa.effects.time_stretch(audio[:, 1].astype(np.float32), rate=rate)
    return np.column_stack([left, right]).astype(np.float64)


def main():
    print("Loading tracks...")
    track1, sr = load_wav_float(TRACK_1)
    track2, _ = load_wav_float(TRACK_2)

    print(f"Track 1: {len(track1)/sr:.1f}s")
    print(f"Track 2: {len(track2)/sr:.1f}s")

    print("\nDetecting tempo and first beat...")
    bpm1, first_beat1 = detect_tempo_and_first_beat(TRACK_1, sr)
    bpm2, first_beat2 = detect_tempo_and_first_beat(TRACK_2, sr)

    print(f"Track 1: {bpm1:.2f} BPM, first beat at {first_beat1*1000:.1f}ms")
    print(f"Track 2: {bpm2:.2f} BPM, first beat at {first_beat2*1000:.1f}ms")

    # Create beat grids
    grid1 = create_beat_grid(first_beat1, bpm1, len(track1)/sr)
    grid2 = create_beat_grid(first_beat2, bpm2, len(track2)/sr)

    print(f"\nBeat grids created:")
    print(f"  Track 1: {len(grid1)} beats, interval {60/bpm1*1000:.2f}ms")
    print(f"  Track 2: {len(grid2)} beats, interval {60/bpm2*1000:.2f}ms")

    # Time-stretch track 2 to match track 1's tempo
    # rate < 1 slows down (longer), rate > 1 speeds up (shorter)
    # We want to slow down from 130 BPM to 128.4 BPM
    stretch_rate = bpm1 / bpm2  # 128.4 / 130.01 = 0.9876 (slower)
    print(f"\nTime-stretching track 2 (rate: {stretch_rate:.4f})...")
    track2_stretched = time_stretch_stereo(track2, stretch_rate)
    print(f"  New length: {len(track2_stretched)/sr:.1f}s (was {len(track2)/sr:.1f}s)")

    # Scale track 2's beat grid
    # When we slow down audio (rate < 1), beat times get proportionally longer
    # New beat time = original beat time / rate
    grid2_stretched = grid2 / stretch_rate
    new_interval = 60.0 / bpm1

    print(f"  Stretched beat interval: {new_interval*1000:.2f}ms (matches track 1)")

    # Find blend point on track 1's grid (nearest beat to target)
    blend_beat_idx = np.argmin(np.abs(grid1 - BLEND_START_SECONDS))
    blend_time = grid1[blend_beat_idx]

    print(f"\nBlend point: beat {blend_beat_idx} at {blend_time:.2f}s")

    # Calculate track 2 start position
    # Track 2's first beat should align with blend_time
    track2_start_time = blend_time - grid2_stretched[0]

    print(f"Track 2 starts at: {track2_start_time:.4f}s")

    # Verify beat alignment
    print(f"\nBeat alignment verification:")
    for i in range(8):
        t1 = grid1[blend_beat_idx + i] if blend_beat_idx + i < len(grid1) else None
        t2 = track2_start_time + grid2_stretched[i] if i < len(grid2_stretched) else None
        if t1 and t2:
            diff = (t2 - t1) * 1000
            print(f"  Beat {i}: T1={t1:.4f}s, T2={t2:.4f}s, diff={diff:+.2f}ms")

    # Build mix
    print(f"\nBuilding mix...")

    blend_start_sample = int(blend_time * sr)
    crossfade_samples = int(CROSSFADE_SECONDS * sr)
    track2_start_sample = int(track2_start_time * sr)

    total_length = track2_start_sample + len(track2_stretched)
    mix = np.zeros((int(total_length), 2), dtype=np.float64)

    # Track 1 solo
    print("  Track 1 solo...")
    mix[:blend_start_sample] = track1[:blend_start_sample]

    # Crossfade
    print("  Crossfade...")
    t = np.linspace(0, 1, crossfade_samples).reshape(-1, 1)
    fade_out = np.sqrt(1 - t)
    fade_in = np.sqrt(t)

    blend_end_sample = blend_start_sample + crossfade_samples

    for i in range(crossfade_samples):
        pos = blend_start_sample + i
        if pos >= total_length:
            break

        t1_pos = pos
        t2_pos = pos - track2_start_sample

        sample = np.zeros(2)
        if 0 <= t1_pos < len(track1):
            sample += track1[t1_pos] * fade_out[i, 0]
        if 0 <= t2_pos < len(track2_stretched):
            sample += track2_stretched[t2_pos] * fade_in[i, 0]

        mix[pos] = sample

    # Track 2 solo
    print("  Track 2 solo...")
    for i in range(blend_end_sample, int(total_length)):
        t2_pos = i - track2_start_sample
        if 0 <= t2_pos < len(track2_stretched):
            mix[i] = track2_stretched[t2_pos]

    # Normalize
    peak = np.max(np.abs(mix))
    if peak > 0.99:
        print(f"  Normalizing (peak: {peak:.2f})...")
        mix = mix * 0.95 / peak

    # Save
    os.makedirs("mixes", exist_ok=True)
    save_wav(OUTPUT, mix, sr)

    print(f"\n{'='*60}")
    print(f"MIX COMPLETE: {OUTPUT}")
    print(f"Duration: {len(mix)/sr/60:.1f} min")
    print(f"Track 1: {bpm1:.2f} BPM")
    print(f"Track 2: {bpm2:.2f} -> {bpm1:.2f} BPM (stretched)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
