# Colormaps

VAPOR's built-in colormaps, as VAPOR 3.11 ships them (`share/palettes` in
NCAR/VAPOR), grouped the way its Load Built-In Colormap menu groups them:
Diverging, Highlighting and Sequential. Each `.tf3` file holds a few control
points (HSV and a position from 0 to 1) and how to interpolate between them;
`visualization/vaporColormaps.py` turns them into full colormaps with a port
of VAPOR's own interpolation, including Kenneth Moreland's diverging (Msh)
interpolation. The lowercase ones are VAPOR's versions of cmocean's maps
(Thyng et al., 2016).

The `.tf3` files are copied unchanged from VAPOR, under its BSD 3-Clause
license (`LICENSE-VAPOR.txt`).

Three of them are Kenneth Moreland's maps, and for those his published
256-colour tables are used instead (the `.csv` beside each `.tf3`):
Kindlmann, KindlmannExtended and BlackBody. VAPOR's control points only
approximate them, and blended in HSV (or Msh, for BlackBody) their
lightness dips and kinks between points, which shows as bands; his tables
rise evenly in lightness from end to end. From
https://www.kennethmoreland.com/color-advice/, where he releases his
original colour maps as public domain (CC0) and notes that no one claims
intellectual property on the Kindlmann or black-body colours.
