# Gas Meter Reader — UH Mānoa Energy & Sustainability VIP

Reads an analog gas meter from a photograph: find the dials, read each needle,
correct the digits, log a value.

Team π-oneers, Fall 2026. Carried forward from Fall 2025 (Mark Davis) and
Spring 2026 (Kristine Loquias).

## Provenance

The Spring 2026 code was only ever distributed as a PDF printout. The modules
here were reconstructed from that printout on 2026-10-27: indentation and line
breaks were rebuilt, logic is unchanged. They import cleanly but have not yet
been run end to end — treat them as a starting point, not a working baseline.

`classifier_infer.py` was never in the printout. See its docstring.

## Layout

| File | What it does |
| --- | --- |
| `main.py` | Entry point. Capture or load an image, detect, read, print a reading. |
| `detection_main.py` | Finds the dials. Currently calls a hosted Roboflow workflow. |
| `reading_ensemble.py` | Reads a needle using ray / contour / line, combined by weighted vote. |
| `reading_main.py` | Older standalone reader. Same geometry, no ensemble module. |
| `ensemble_utils.py` | Angle maths, the weighted vote, and Rule 4. |
| `classifier_infer.py` | Stub. The fourth ensemble method, deliberately disabled. |

## Known issues

- **Dial count is not fixed.** `EXPECTED_GAUGES = 5` is hardcoded in `main.py`.
  Hale Kahawai has 4 reading dials; Campus Center has 5. This has to become a
  per-meter setting.
- **Detection needs the internet.** `detection_main.py` posts to a hosted
  Roboflow workflow. A locally trained YOLOv11 model now matches it at 99.5%
  mAP and runs on the Pi, so this should be swapped for local inference.
- **The classifier is disabled.** It was a car-colour demo model weighted
  higher than the three geometric methods.
- **Not yet run end to end.** Reconstructed from a printout; expect fixes.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp secret.example.py secret.py   # then fill in, if using hosted detection
python main.py
```

Never commit `secret.py`. The key printed in the Spring 2026 PDF is public and
should be rotated.
