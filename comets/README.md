# Comets: finding moving dots near the Sun in SOHO images

SOHO's LASCO coronagraphs block the Sun's disk so the faint corona shows up,
and so does anything passing close to the Sun. More than 5,000 comets have
been found in these images, nearly all by volunteers in NASA's
[Sungrazer Project](https://sungrazer.nrl.navy.mil/). Most are small
**Kreutz sungrazers**, fragments of one giant comet that broke up centuries
ago, diving toward the Sun and usually evaporating.

This folder automates the first pass of what a Sungrazer hunter does by eye:
flip through frames and look for a faint fuzzy dot moving steadily,
differently from the stars.

Each candidate card shows a zoom on the track (a small circle at the
object's predicted position in every frame, plus an inset of the whole
field) and close-ups from the background-subtracted frames with a
4-pixel circle drawn to scale. A real mover sits inside the circle in every
close-up; a star blend or a chance line-up of noise wanders out of it.

```
soho.py       get LASCO C2/C3 frames (Helioviewer API, cached)
track.py      remove the background, find spots, link straight-line tracks, reject stars
inject.py     injection-recovery: add fake comets, count what comes back
scan.py       scan recent frames, check Sungrazer reports, write queue cards
tests/        offline tests on synthetic frames
```

```bash
cd comets
python scan.py                               # last 24 hours of C3
python scan.py --start 2026-10-23T00:00 --hours 10
```

## The method

1. **Remove what doesn't move.** Each frame minus the median of frames
   48 minutes to 2 hours away. The corona and glare cancel, and so do the
   slowly drifting stars, mostly. The nearest frames are skipped
   on purpose: a slow comet hasn't moved far in 12–36 minutes, so they'd
   subtract the comet itself. Injection tests showed this doubled
   recovery of slow movers.
2. **Find spots.** Smooth a little (comets are fuzzy blobs a few pixels
   wide), divide by the local noise, keep local peaks above 4.5σ, and
   measure each position to a fraction of a pixel.
3. **Link into tracks.** Within any 8 consecutive frames (~1.6 h), two
   spots propose a velocity. Predict where the object should be in the
   other frames, and keep it if at least 6 of the 8 agree within 1.5 px.
   Noise and cosmic rays don't line up like that.
4. **Reject the sky.** SOHO orbits the Sun with Earth, so every star
   drifts across C3 at about 2.7 px/hour. The most common track velocity
   *is* the star drift; anything moving with it (within its measurement
   error) is a star. Anything barely moving is static.
5. **Score and check.**
   - Score: significance, track quality, and whether the object heads toward
     the Sun.
   - Check: compare against recent Sungrazer reports (rotated to their
     north-up image frame), so already-reported comets are labelled.

## Validation

**A real comet.** On 23 Oct 2026, Chuck Fields reported a Kreutz comet in
C3 (`report_cf_20261023080539`), moving at (−9.1, −7.2) px/hour. Running
`scan.py` over the same 10 hours found it independently:
- velocity (−8.3, −7.5) px/hour, aimed straight at the Sun (sunward = 0.98)
- 7 frames, 5σ
- matched to that report automatically

Of 311 tracks in those frames, 304 were stars and 2 were static. Five
"movers" remained, including the comet. The other four are the kind of
thing a person rules out in seconds (e.g. two close stars blending).

**Injected fakes** (40 fakes into real C3 frames, a first pass):

| Fake brightness (× noise) | ≤3 | 3–4 | 4–5 | 5–6 | 6–8 | 8–13 |
|---|---|---|---|---|---|---|
| Recovered | 0% | 67% | 17% | 100% | 80% | 91% |

There were no false movers in the same frames without injections. A larger
injection run is still to do.

## Limits and next steps

- **C2 lags.** Helioviewer's C2 frames run about a month behind; C3 is
  near real time. Real-time C2 (and C3) is on `soho.nascom.nasa.gov`, which
  this environment can't reach yet.
- **CCOR-1.** NOAA's new coronagraph on GOES-19 ("G1" in Sungrazer
  reports) is producing many recent discoveries and is less picked over.
  It's worth adding as a second camera.
- **Positions for reporting** are converted to the Sungrazer north-up frame
  approximately (good to ~25 px). Before submitting a report, measure the
  object on the project's own images with their tool.
- **Bright planets and their bleed streaks** are masked out, so a comet
  passing right next to one is missed.
