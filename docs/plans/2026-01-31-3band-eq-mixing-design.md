# 3-Band EQ Mixing Design

## Overview

Replace simple crossfade with professional 3-band EQ mixing for cleaner transitions.

## EQ Bands

Using classic DJ mixer crossover frequencies with Linkwitz-Riley filters (24dB/octave):

| Band | Frequency Range | Content |
|------|-----------------|---------|
| Low  | 0-250Hz         | Kick, sub bass |
| Mid  | 250-2500Hz      | Melodies, vocals, synths |
| High | 2500Hz+         | Hi-hats, cymbals, air |

## Phrase-Based Timing

Transitions structured around musical phrases:
- **Bar length:** 4 beats (`4 * master_interval`)
- **8-bar phrase:** `8 * bar_length`
- **16-bar phrase:** `16 * bar_length`

## Per-Band Automation

### Bass (Low)
- **Rule:** Never play from both tracks simultaneously
- **Swap point:** 16-bar boundary (middle of transition)
- **Outgoing:** 100% until swap, then 0%
- **Incoming:** 0% until swap, then 100%

### Highs
- **Rule:** Very rarely overlap
- **Swap point:** 8-bar boundary, offset from bass swaps
- **Outgoing:** 100% until swap (8 bars in), then 0%
- **Incoming:** 0% until swap, then 100%

### Mids
- **Rule:** Often play together, gradual blend
- **Outgoing:** 100% at start → fades to 30% → hard cut to 0% over 2 beats
- **Incoming:** 0% at start → cuts in to 30% over 4 beats → rises to 100%

## Hard Cut at End

- Snap to 16-bar boundary near end of transition
- Outgoing track fades all bands to zero over 1-2 bars
- Quick but not jarring

## Implementation

Changes to `dj_mix.py`:

1. **Add scipy dependency** for butterworth filters

2. **`create_bandpass_filter(lowcut, highcut, sr, order=4)`**
   - Creates Linkwitz-Riley filter coefficients

3. **`apply_filter(audio, sos)`**
   - Applies filter to stereo audio

4. **`split_to_bands(audio, sr)`**
   - Returns `{'low': array, 'mid': array, 'high': array}`

5. **`calculate_phrase_points(blend_start, master_interval, transition_duration)`**
   - Returns dict with:
     - `bass_swap`: 16-bar boundary
     - `high_swap`: 8-bar boundary (offset)
     - `hard_cut_start`: 16-bar boundary near end
     - `hard_cut_end`: 1-2 bars after hard_cut_start

6. **`mix_with_eq(outgoing, incoming, blend_start, blend_duration, sr, master_interval)`**
   - Split both tracks to bands
   - Apply per-band gain envelopes
   - Sum bands and return mixed audio

7. **Update main crossfade loop (lines 285-309)**
   - Replace with call to `mix_with_eq()`

## Dependencies

Add to requirements:
```
scipy
```
