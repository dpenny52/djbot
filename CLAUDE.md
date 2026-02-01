# DJ Mix Bot

Creates beatmatched DJ mixes from two tracks.

## Key Learnings

### Tempo Detection
- Librosa's default `beat_track` rounds tempo, hiding small BPM differences
- Use `hop_length=128` (not default 512) for ~3ms precision instead of ~12ms
- Calculate actual BPM from `60 / median(beat_intervals)`, not librosa's returned tempo

### Time-Stretching
- `librosa.effects.time_stretch(audio, rate)`: rate < 1 slows down, rate > 1 speeds up
- To match tempos: `stretch_rate = target_bpm / source_bpm`
- After stretching, scale beat times: `new_beat_times = old_beat_times / stretch_rate`

### Beat Alignment
- Don't re-detect beats after stretching (inconsistent results)
- Create mathematical beat grid: `first_beat + np.arange(n) * (60/bpm)`
- Align track 2's first beat to land on track 1's beat at blend point

### Crossfade
- Equal-power crossfade prevents volume dip: `fade_out = sqrt(1-t)`, `fade_in = sqrt(t)`

## Usage

```bash
source venv/bin/activate && python dj_mix.py
```

## Dependencies

```bash
pip install numpy librosa
```
