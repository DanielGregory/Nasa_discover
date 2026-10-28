# NASA Discover

Learning-first searches of public NASA data for new worlds and other
moving or changing things in the sky, plus a phone-friendly queue where
a person reviews the most promising candidates.

| Folder | Hunts for | Data | Status |
|---|---|---|---|
| [`planets/`](planets/) | transiting exoplanets | TESS light curves (MAST), TIC, NASA Exoplanet Archive, ExoFOP | ✅ validated on known planets and 900 injections |
| `comets/` | comets near the Sun | SOHO LASCO coronagraph images | planned |
| `brown_dwarfs/` | nearby cold brown dwarfs | WISE / unWISE, CatWISE | planned |
| `transients/` | fast-fading transients (kilonova-like) | ZTF alert brokers | planned |
| [`feed/`](feed/) | n/a: the review app | the **Discovery Queue** page (claude.ai) | ✅ live |

How it fits together:

1. Each search finds candidates and vets them against what is already known.
2. Survivors become cards in the Discovery Queue (`feed/candidates.py` defines
   the card format).
3. You mark each card **Worth a look / Unsure / Not real** on your phone.
4. Claude follows up on the promising ones and drafts reports. You submit them
   from your own accounts (ExoFOP, Sungrazer, Backyard Worlds), so the credit
   is yours.

Ground rule for every search: prove it on known objects and injected fakes
before trusting it on anything new.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest            # offline tests for all searches
```
