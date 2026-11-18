# GTA VI launch-day Twitch forecaster

An entry for make.fun's **gta6-twitch-ccu-forecast-01**: forecast, for each
hour 2–24 of Grand Theft Auto VI's release day (2026-11-19, UTC), five
quantiles (5/25/50/75/95%) of the concurrent viewers in Twitch's
"Grand Theft Auto VI" category, given everything recorded up to the end of
hour 1. Scoring is mean pinball loss on the log scale; a prize requires
beating both house entries (a flat persistence baseline and a one-shot AI
model forecast).

## The model

A launch day has a *shape*, and the shape depends on the launch's scale.
The biggest launches are front-loaded — hour 1 is at or near the day's peak
and the day sags from there — while small launches ignite late and can end
the day a hundred times their first hour. So the program learns the shape
from the ~40 past launches shipped inside every case file, weighted toward
launches of a similar size:

1. For each comparable, take its 24 release-day hourly counts (matched by
   timestamp against its `day_start`) and compute the log-ratio curve
   `log(viewers_h / viewers_1)` for h = 2..24. Comparables without a full
   release day, or with a first hour under 50 viewers, are skipped; the
   target category's own non-launch history (listed as a comparable in the
   final) is excluded by name.
2. Weight each comparable by `exp(-|log(its hour 1) - log(this hour 1)|)` —
   a launch whose first hour is e× this one's counts about 0.37 as much.
3. The median forecast for hour h is `p1 * exp(weighted median of the
   log-ratios)`. The other quantiles use the weighted dispersion of the
   ratios at that hour: the wider of the measured deviation (capped at
   e^±3 around the median) and a random-walk floor of
   `z_tau * 0.15 * sqrt(h - 1)`, so the band never claims more certainty
   late in the day than the house baseline's own spread.
4. Momentum (final only): when the case's `history` holds fine-grained
   reads inside hour 1, the mean of the last 15 minutes is compared with
   hour 1's print; the log difference, capped at ±0.30, lifts (or lowers)
   the early hours and decays with a 6-hour time constant. With hourly
   history the adjustment switches itself off.
5. Guard rails: quantiles are clamped to [1, 1e9], forced non-decreasing,
   and the output object is shape-checked before it is written; if the
   comparables are unusable the program falls back to a persistence-shaped
   forecast with a wider spread rather than failing the case.

## Validation

Leave-one-out over the practice comparables (37 usable; each in turn the
target with its hour-1 print as the only anchor, the other comparables as
the base), scored with the contest's pinball loss:

- **This program: 0.428** — persistence baseline: 0.930 (54% lower loss).
- On launches with a first hour of 100k+ viewers (this launch's regime):
  **0.225 against the baseline's 0.261**.

## Running it

Node 22, standard library only, no network. The program reads every `*.in`
file in `$MAKEFUN_INPUT_DIR` (default `/input`) and writes one `<id>.out`
per case to `$MAKEFUN_OUTPUT_DIR` (default `/output`):

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

It is deterministic: two runs on the same case write identical bytes, and
a case takes well under a second (limit is 8 s).
