#!/usr/bin/env python3
"""
DJ Mix Bot - Multi-track beatmatched mixing

Features:
- Interval-based tempo matching (handles alternating beat patterns)
- Automatic time-stretching to match tempos
- Optimal beat alignment for seamless transitions
- Equal-power crossfade
"""

import wave
import numpy as np
import librosa
import os
from scipy.signal import butter, sosfilt

TRACKS = [
    "tracks/Steinmetz - STNMTZ001 - Kraft - 01 Steinmetz - Kraft (Original Mix).wav",
    "tracks/Steinmetz - STNMTZ001 - Kraft - 02 Steinmetz - Werk (Original Mix).wav",
    "tracks/Steinmetz - STNMTZ001 - Kraft - 03 Steinmetz - Ruhe (Original Mix).wav",
    "tracks/Steinmetz - STNMTZ001 - Kraft - 04 Steinmetz - Kraft (Edit Select Remix).wav",
    "tracks/Steinmetz - STNMTZ001 - Kraft - 05 Steinmetz - Werk (Takaaki Itoh Remix).wav",
]
OUTPUT = "mixes/mix.wav"

BLEND_BEFORE_END_SECONDS = 60.0
CROSSFADE_SECONDS = 30.0
HOP_LENGTH = 128

# EQ crossover frequencies (Hz)
LOW_CUTOFF = 250
HIGH_CUTOFF = 2500
FILTER_ORDER = 4  # 24dB/octave Linkwitz-Riley


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


def detect_beats_and_interval(filepath, sr):
    """
    Detect beats and calculate the actual median interval.
    Also detects if there's an alternating pattern (8th note detection).
    """
    y, _ = librosa.load(filepath, sr=sr, mono=True)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)

    if len(beat_times) < 20:
        raise ValueError(f"Too few beats in {filepath}")

    intervals = np.diff(beat_times)

    # Check for alternating pattern (8th note detection)
    # If intervals alternate between two distinct values, we have 8th notes
    # Use sum of pairs as the actual beat interval
    is_alternating = False
    if len(intervals) >= 10:
        even_intervals = intervals[::2][:20]
        odd_intervals = intervals[1::2][:20]

        even_median = np.median(even_intervals)
        odd_median = np.median(odd_intervals)

        # If even and odd intervals are consistently different, it's alternating
        diff_ratio = abs(even_median - odd_median) / min(even_median, odd_median)
        if 0.005 < diff_ratio < 0.02:  # ~0.5-2% difference is alternating pattern
            is_alternating = True
            # Use sum of pairs as beat interval
            pair_intervals = even_intervals + odd_intervals[:len(even_intervals)]
            actual_interval = np.median(pair_intervals) / 2
            print(f"    Detected alternating pattern: {even_median*1000:.1f}ms / {odd_median*1000:.1f}ms")
            print(f"    Using averaged interval: {actual_interval*1000:.1f}ms")
        else:
            actual_interval = np.median(intervals)
    else:
        actual_interval = np.median(intervals)

    actual_bpm = 60.0 / actual_interval

    return beat_times, actual_interval, actual_bpm, is_alternating


def time_stretch_stereo(audio, rate):
    left = librosa.effects.time_stretch(audio[:, 0].astype(np.float32), rate=rate)
    right = librosa.effects.time_stretch(audio[:, 1].astype(np.float32), rate=rate)
    return np.column_stack([left, right]).astype(np.float64)


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

        # === HIGHS: 4-bar crossfade centered on 8-bar boundary ===
        high_fade_duration = 4 * bar_samples
        high_fade_start = high_swap_sample - (high_fade_duration // 2)
        high_fade_end = high_swap_sample + (high_fade_duration // 2)

        if blend_pos < high_fade_start:
            out_high_gain = 1.0
            in_high_gain = 0.0
        elif blend_pos < high_fade_end:
            # Equal-power crossfade over 4 bars
            fade_progress = (blend_pos - high_fade_start) / high_fade_duration
            out_high_gain = np.sqrt(1.0 - fade_progress)
            in_high_gain = np.sqrt(fade_progress)
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

        # Apply global fade-in to incoming track (0.1s) to avoid click
        fade_in_samples = int(0.1 * sr)
        if blend_pos < fade_in_samples:
            fade_in_mult = blend_pos / fade_in_samples
            in_low_gain *= fade_in_mult
            in_mid_gain *= fade_in_mult
            in_high_gain *= fade_in_mult

        # Combine
        sample = (out_low * out_low_gain + in_low * in_low_gain +
                  out_mid * out_mid_gain + in_mid * in_mid_gain +
                  out_high * out_high_gain + in_high * in_high_gain)

        output[i] = sample

    return output


def main():
    print("=" * 70)
    print("MULTI-TRACK DJ MIX")
    print("=" * 70)

    # Load all tracks and analyze beats
    print("\nAnalyzing tracks...")
    track_data = []

    for i, path in enumerate(TRACKS):
        audio, sr = load_wav_float(path)
        duration = len(audio) / sr

        track_name = os.path.basename(path).split(" - ")[-1].replace(".wav", "")
        print(f"\nTrack {i+1}: {track_name} ({duration:.1f}s)")

        beats, interval, bpm, is_alt = detect_beats_and_interval(path, sr)
        print(f"    BPM: {bpm:.2f}, interval: {interval*1000:.2f}ms")
        print(f"    First beat: {beats[0]:.3f}s, {len(beats)} beats total")

        track_data.append({
            'audio': audio,
            'beats': beats,
            'interval': interval,
            'bpm': bpm,
            'is_alternating': is_alt,
            'duration': duration,
            'name': track_name,
            'path': path,
        })

    # Use first track's interval as master
    master_interval = track_data[0]['interval']
    master_bpm = 60.0 / master_interval

    print(f"\n{'='*70}")
    print(f"Master: {master_bpm:.2f} BPM (interval: {master_interval*1000:.2f}ms)")
    print(f"{'='*70}")

    # Time-stretch all tracks to master interval
    print("\nTime-stretching tracks to master tempo...")
    for i, td in enumerate(track_data):
        # Stretch rate based on INTERVALS, not BPM
        # rate = master_interval / track_interval
        # rate > 1 speeds up (shortens intervals)
        # rate < 1 slows down (lengthens intervals)
        stretch_rate = td['interval'] / master_interval

        td['stretch_rate'] = stretch_rate

        if i == 0:
            td['audio_stretched'] = td['audio']
            td['beats_stretched'] = td['beats']
            td['interval_stretched'] = td['interval']
        else:
            if abs(stretch_rate - 1.0) > 0.001:
                print(f"  Track {i+1}: interval {td['interval']*1000:.2f}ms -> {master_interval*1000:.2f}ms (rate: {stretch_rate:.4f})")
                td['audio_stretched'] = time_stretch_stereo(td['audio'], stretch_rate)
            else:
                print(f"  Track {i+1}: already at master tempo")
                td['audio_stretched'] = td['audio']

            # Scale beat times
            td['beats_stretched'] = td['beats'] / stretch_rate
            td['interval_stretched'] = td['interval'] / stretch_rate

        td['duration_stretched'] = len(td['audio_stretched']) / sr

    # Verify stretching worked
    print("\nVerifying tempo after stretch...")
    for i, td in enumerate(track_data):
        if i == 0:
            continue
        # Check intervals after stretching
        stretched_intervals = np.diff(td['beats_stretched'][:20])
        median_stretched = np.median(stretched_intervals)
        print(f"  Track {i+1}: stretched interval = {median_stretched*1000:.2f}ms (target: {master_interval*1000:.2f}ms)")

    # Build the mix
    print(f"\n{'='*70}")
    print("Building mix...")
    print(f"{'='*70}")

    crossfade_samples = int(CROSSFADE_SECONDS * sr)

    # Start with first track
    first_track = track_data[0]
    mix = first_track['audio_stretched'].copy()

    blend_timestamps = []
    current_track_start = 0

    for i in range(1, len(track_data)):
        outgoing = track_data[i-1]
        incoming = track_data[i]

        # Calculate blend point
        outgoing_end = current_track_start + len(outgoing['audio_stretched'])
        blend_target = (outgoing_end / sr) - BLEND_BEFORE_END_SECONDS

        # Find nearest beat on outgoing track
        outgoing_beats_in_mix = outgoing['beats_stretched'] + current_track_start / sr
        blend_beat_idx = np.argmin(np.abs(outgoing_beats_in_mix - blend_target))
        blend_beat_time = outgoing_beats_in_mix[blend_beat_idx]

        print(f"\nTransition {i} -> {i+1}")
        print(f"  Blend at {blend_beat_time:.2f}s ({blend_beat_time/60:.1f} min)")

        # Find best alignment for incoming track
        # Try aligning each of the first few incoming beats to the blend point
        best_start = None
        best_error = float('inf')
        best_beat_idx = 0

        for try_idx in range(min(30, len(incoming['beats_stretched']))):
            try_beat = incoming['beats_stretched'][try_idx]
            try_start = blend_beat_time - try_beat

            # Calculate cumulative alignment error over crossfade
            in_beats_in_mix = incoming['beats_stretched'] + try_start
            xfade_in_beats = in_beats_in_mix[(in_beats_in_mix >= blend_beat_time) &
                                              (in_beats_in_mix <= blend_beat_time + CROSSFADE_SECONDS)]

            if len(xfade_in_beats) < 5:
                continue

            # Calculate error against master grid
            # Master grid: blend_beat_time + n * master_interval
            total_error = 0
            for beat in xfade_in_beats[:10]:
                offset_from_blend = beat - blend_beat_time
                n = round(offset_from_blend / master_interval)
                expected = blend_beat_time + n * master_interval
                error = abs(beat - expected)
                total_error += error

            avg_error = total_error / min(10, len(xfade_in_beats))

            if avg_error < best_error:
                best_error = avg_error
                best_start = try_start
                best_beat_idx = try_idx

        if best_start is None:
            # Fallback
            best_start = blend_beat_time - incoming['beats_stretched'][0]
            best_beat_idx = 0
            best_error = 0

        incoming_start_sample = int(best_start * sr)
        print(f"  Incoming beat {best_beat_idx} aligns with blend point")
        print(f"  Incoming starts at {best_start:.3f}s")
        print(f"  Avg alignment error: {best_error*1000:.1f}ms")

        # Verify alignment
        in_beats_in_mix = incoming['beats_stretched'] + best_start
        xfade_in_beats = in_beats_in_mix[(in_beats_in_mix >= blend_beat_time) &
                                          (in_beats_in_mix <= blend_beat_time + CROSSFADE_SECONDS)]
        xfade_out_beats = outgoing_beats_in_mix[(outgoing_beats_in_mix >= blend_beat_time) &
                                                 (outgoing_beats_in_mix <= blend_beat_time + CROSSFADE_SECONDS)]

        print(f"  Beat check:")
        for j in range(min(5, len(xfade_out_beats), len(xfade_in_beats))):
            out_b = xfade_out_beats[j]
            in_b = xfade_in_beats[j]
            diff = (in_b - out_b) * 1000
            print(f"    Beat {j}: out={out_b:.4f}s, in={in_b:.4f}s, diff={diff:+.1f}ms")

        # Build crossfade
        required_length = incoming_start_sample + len(incoming['audio_stretched'])
        if required_length > len(mix):
            new_mix = np.zeros((required_length, 2), dtype=np.float64)
            new_mix[:len(mix)] = mix
            mix = new_mix

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

        # Incoming solo
        incoming_audio = incoming['audio_stretched']
        for j in range(blend_end_sample, incoming_start_sample + len(incoming_audio)):
            if j >= len(mix):
                break
            in_pos = j - incoming_start_sample
            if 0 <= in_pos < len(incoming_audio):
                mix[j] = incoming_audio[in_pos]

        current_track_start = incoming_start_sample
        blend_timestamps.append({
            'from_name': outgoing['name'],
            'to_name': incoming['name'],
            'blend_start': blend_beat_time,
            'blend_end': blend_beat_time + CROSSFADE_SECONDS,
            'alignment_error_ms': best_error * 1000,
        })

    # Normalize
    peak = np.max(np.abs(mix))
    if peak > 0.99:
        print(f"\nNormalizing (peak: {peak:.2f})...")
        mix = mix * 0.95 / peak

    # Save
    os.makedirs("mixes", exist_ok=True)
    save_wav(OUTPUT, mix, sr)

    # Summary
    print(f"\n{'='*70}")
    print(f"MIX COMPLETE: {OUTPUT}")
    print(f"Duration: {len(mix)/sr/60:.1f} min")
    print(f"Master tempo: {master_bpm:.2f} BPM")
    print(f"{'='*70}")

    print(f"\nBLEND TIMESTAMPS FOR QA:")
    print("-" * 70)
    for bt in blend_timestamps:
        start_mm = int(bt['blend_start'] // 60)
        start_ss = bt['blend_start'] % 60
        end_mm = int(bt['blend_end'] // 60)
        end_ss = bt['blend_end'] % 60
        print(f"  {bt['from_name'][:30]:30} -> {bt['to_name'][:30]}")
        print(f"    Blend: {start_mm}:{start_ss:05.2f} - {end_mm}:{end_ss:05.2f}")
        print(f"    Alignment error: {bt['alignment_error_ms']:.1f}ms")
        print()


if __name__ == "__main__":
    main()
