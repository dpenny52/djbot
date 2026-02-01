# DJ Mix Bot

Automatically creates beatmatched DJ mixes from two audio tracks.

## Features

- Precise tempo detection using librosa
- Automatic time-stretching to match tempos
- Mathematical beat grid alignment
- Equal-power crossfade for smooth transitions

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy librosa
```

## Usage

1. Place your tracks in the `tracks/` directory
2. Edit `dj_mix.py` to set your track paths and mix parameters
3. Run the script:

```bash
source venv/bin/activate
python dj_mix.py
```

Output will be saved to `mixes/`.

## Configuration

Edit these parameters in `dj_mix.py`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `TRACK_1` | First track path | - |
| `TRACK_2` | Second track path | - |
| `BLEND_START_SECONDS` | When to start the transition | 96.0 |
| `CROSSFADE_SECONDS` | Length of the crossfade | 30.0 |

## How It Works

1. Detects tempo and first beat of each track
2. Time-stretches track 2 to match track 1's BPM
3. Aligns beat grids mathematically
4. Applies equal-power crossfade during transition
