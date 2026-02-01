# DJ Mix Bot

Creates beatmatched DJ mixes from multiple tracks.

## Key Learnings

### Tempo Detection
- Librosa's default `beat_track` rounds tempo, hiding small BPM differences
- Use `hop_length=128` (not default 512) for ~3ms precision instead of ~12ms
- Calculate actual BPM from `60 / median(beat_intervals)`, not librosa's returned tempo
- **Alternating beat patterns**: Some tracks have librosa detecting 8th notes (alternating short/long intervals like 467ms/470ms). Detect this pattern and average the intervals to get true quarter-note tempo

### Time-Stretching
- `librosa.effects.time_stretch(audio, rate)`: rate < 1 slows down, rate > 1 speeds up
- **Use interval ratios, not BPM ratios**: `stretch_rate = source_interval / target_interval`
- After stretching, scale beat times: `new_beat_times = old_beat_times / stretch_rate`

### Beat Alignment
- Don't re-detect beats after stretching (inconsistent results)
- **Tracks have tempo drift**: A single phase value won't work for the whole track
- Use actual detected beat times for alignment, not mathematical grids
- Try multiple alignment points (first 30 beats) and pick the one with lowest cumulative error
- **Ambient intros**: Some tracks have 30+ seconds before first detected beat - handle gracefully

### Multi-Track Mixing
- Chain tracks by calculating blend point relative to outgoing track end
- Each incoming track needs independent tempo matching to master
- Track cumulative position in mix to correctly place each track

### Crossfade
- Equal-power crossfade prevents volume dip: `fade_out = sqrt(1-t)`, `fade_in = sqrt(t)`

### 3-Band EQ Mixing
- Use scipy butterworth filters (24dB/octave) for clean band separation
- Crossovers: Low 0-250Hz, Mid 250-2500Hz, High 2500Hz+
- Bass swaps on 16-bar boundaries (never both tracks at once)
- Highs swap on 8-bar boundaries, offset from bass
- Mids blend gradually with hard cut at end

## Usage

```bash
source venv/bin/activate && python dj_mix.py
```

## Dependencies

```bash
pip install numpy librosa scipy
```

## Notes

- Expects stereo 16-bit WAV files (44.1kHz)
- `tracks/` and `mixes/` directories are gitignored - create them manually
- Edit `TRACKS` list in `dj_mix.py` to set your track paths
