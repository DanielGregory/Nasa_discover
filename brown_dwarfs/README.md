# Brown dwarfs: cold, nearby, moving and very red in WISE

Brown dwarfs are too small to fuse hydrogen, so they cool and fade forever.
The coldest (T and Y types, at a few hundred °C) are nearly invisible to
optical telescopes but glow at the wavelengths NASA's WISE satellite sees.
Volunteers in [Backyard Worlds](https://www.zooniverse.org/projects/marckuchner/backyard-worlds-planet-9)
have found thousands of them by flipping between WISE images taken years apart.

```
search.py     query CatWISE2020 (NASA IRSA) for red, moving, clean sources; Gaia check; cards
flipbook.py   unWISE images from 2010-11 and 2010-2020 side by side, with a difference image
```

```bash
cd brown_dwarfs
python search.py --tiles 5                  # 5 random 10x10 degree patches
python search.py --ra 128 138 --dec -10 0   # a specific patch
```

## Method

1. **Colour.** Methane in cold atmospheres absorbs WISE's W1 band (3.4 µm)
   but not W2 (4.6 µm), so W1 − W2 > 1.5 picks mid/late T and Y dwarfs.
   Ordinary stars sit near 0.
2. **Motion.** Nearby objects drift across the sky. We require proper
   motion > 0.3″/yr, measured at ≥ 5σ in CatWISE2020 (2010-2018 baseline).
3. **Clean data.** No unWISE artifact flags (halos, diffraction spikes,
   ghosts) and W2 fainter than 8 mag: saturated stars fake a red colour.
4. **Merge split entries.** A very fast mover can be catalogued several
   times along its path (WISE 0855−0714 appears four times).
5. **Gaia check.** Cold brown dwarfs are too faint for Gaia's optical
   camera. A bright Gaia counterpart means a warmer dwarf or a known star,
   so it's ranked lower.
6. **Flipbook.** The card shows the old and new WISE images and their
   difference, with an arrow for the catalogue motion, so you can check
   that a dot really moved along it.

## Validation

WISE 0855−0714, the coldest known brown dwarf (moving 8″/yr,
W1 − W2 = 3.9), comes out **#1** in its 10°×10° patch. Its flipbook shows a
dot in 2010–11 and a 30-pixel streak in the 2010–2020 combined image, plus
the clear blue/red pair in the difference image.

## Honest expectations

- **Most bright red movers are already known.** The CatWISE team and
  Backyard Worlds have mined this catalogue thoroughly. New finds
  tend to be fainter than our W2 S/N > 10 cut, in crowded regions,
  or have catalogue motions spoiled by blending. Lowering the S/N cut and
  searching the Galactic plane are where to push next.
- **"Is it known?" is still manual.** SIMBAD and VizieR, which list known
  brown dwarfs, aren't reachable from this environment. Each card links to
  SIMBAD at the position. Allow `simbad.cds.unistra.fr` to automate it.
- Report anything promising through Backyard Worlds' own tools, so it's
  checked against their known-object lists and credited to you.
