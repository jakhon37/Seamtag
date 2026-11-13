# Seamtag's mark

Seamtag's mark is a seam: an S that parts photo tiles on paper from cloth in the app's blue. A line of stitches runs beside it on the cloth side, and a white label, the tag in the name, comes out from under its edge. Two editing handles sit on the seam.

## Files

| File | Use |
|---|---|
| `seamtag-mark.svg` | The mark: from 64 px in the app, from 48 px in the installers' icons |
| `seamtag-mark-small.svg` | The mark below that: the two sides and the shadow, with no detail, label or handles, and a firmer rim |
| `../../resources/icon.png` | Window icon: the mark at 944 of 1024 px |
| `../../resources/icon.icns`, `icon.ico`, `icons/` | The installers' icons for macOS, Windows and Linux, with the small mark at 32 px and under, and at 16 and 32 points on a Mac |

`make.py` draws both SVGs from Seamline's mark in [`docs/brand`](../brand) and copies them into `src/renderer/src/assets`; `icons.cjs` makes the icons from them:

```bash
python3 docs/mark/make.py && node docs/mark/icons.cjs
```

## Colours

Paper (`#f3f5f9`) on the photo side, with the tiles in greys at about a third of their strength, and the app's blue (`#4f8cff`) for the cloth, shaded from `#5c97ff` to `#3f7ae8`. The handles are ringed in `#14161a`, and the rim is a faint grey (`#8a93a3`).
