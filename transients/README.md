# Transients: fast-fading new sources in ZTF (kilonova-like)

A kilonova is the short-lived glow from radioactive debris after two
neutron stars merge (AT2017gfo, in 2017, came with gravitational waves). The
Zwicky Transient Facility (ZTF) scans the northern sky every few nights and
issues alerts for everything that changes. Brokers like
[ALeRCE](https://alerce.online) classify those alerts in real time.

```
fast_fade.py   recent supernova-like ZTF objects -> fade rate per band -> cards for fast faders
tests/         the fade-rate logic on a kilonova-like curve, a supernova, and a noisy case
```

```bash
cd transients
python fast_fade.py                     # objects first seen in the last 7 days
python fast_fade.py --days 30 --max-objects 400
```

## Method

1. Ask ALeRCE for objects first seen in the last N days that its image
   classifier calls supernova-like (a real new source, not a variable star,
   AGN, asteroid or artifact), with at least 3 detections.
2. In each band, fit a straight line to the magnitudes from peak up to
   10 days later: the **fade rate** in mag/day.
3. Flag fades faster than **0.3 mag/day** at ≥ 3σ. Supernovae fade
   ~0.01–0.1 mag/day; AT2017gfo faded ~0.5–1 mag/day and reddened quickly.
4. Rank by fade rate, significance, redness and youth.

## Validation and first run

- SN 2023ixf (ZTF23aaklqou), a normal nearby supernova, fades
  0.032 mag/day and is correctly *not* flagged.
- A synthetic AT2017gfo-like light curve (0.8–1.1 mag/day, reddening) is
  flagged (`tests/`).
- No real kilonova is in ZTF's archive to test against, so treat the
  thresholds as reasonable rather than proven.
- First run (30 days, 100 objects): median fade 0.05 mag/day, as expected
  for supernovae, and **no significant fast faders**. That's the normal
  result: kilonovae are rare and mostly found by following
  gravitational-wave alerts.

## Next steps

- Trigger on gravitational-wave alerts (LIGO/Virgo/KAGRA public alerts via
  GCN) and search only the sky region and days after each event.
- Use host-galaxy distance to check the peak brightness is kilonova-like.
