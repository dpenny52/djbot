asdfasdfdsafasdfdsasan viuahv yuihciewayehbuiaevhi

# DJ Mix Bot

Automatically creates beatmatched DJ mixes from multiple audio tracks.

## Features

- Precise tempo detection using librosa with high-resolution hop length
- Automatic time-stretching to match tempos
- Handles alternating beat patterns (8th note detection)
- Multi-track mixing with seamless transitions
- Equal-power crossfade for smooth blends

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy librosa
```

## Usage

1. Create `tracks/` and `mixes/` directories
2. Place your WAV files in `tracks/`
3. Edit `dj_mix.py` to set your track list and parameters
4. Run the script:

```bash
source venv/bin/activate
python dj_mix.py
```

Output will be saved to `mixes/`.

## Configuration

Edit these parameters in `dj_mix.py`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `TRACKS` | List of track paths to mix | - |
| `BLEND_BEFORE_END_SECONDS` | How early before track end to start transition | 60.0 |
| `CROSSFADE_SECONDS` | Length of the crossfade | 30.0 |

## How It Works

1. Analyzes each track for tempo and beat positions
2. Detects alternating beat patterns (when librosa finds 8th notes)
3. Time-stretches all tracks to match the first track's tempo
4. Finds optimal beat alignment for each transition
5. Applies equal-power crossfade during transitions
6. Outputs blend timestamps for QA

## File Format

- Stereo 16-bit WAV files at 44.1kHz
