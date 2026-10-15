"""
Interpolate CAMulator output from its hybrid model levels to pressure levels.

CAMulator's state is on CAM6's 32 hybrid sigma-pressure levels (u0k..u31k,
0 = model top), which line up with no other model's output. This rewrites a
CAMulator run's NetCDF with the pressure-level names the other models use
(u500, t850, q700, z500, ...), so it can be viewed and compared like them.

Per column, the pressure of level k is hyam[k] * P0 + hybm[k] * sp, and of
the interfaces between levels hyai + hybi * sp (hyai is already in Pa in
CAMulator's file). Fields
are interpolated linearly in log pressure. Geopotential isn't a CAMulator
output; it is integrated up from the surface geopotential with the
hypsometric equation, from the virtual temperature of each layer, as CAM
does. Between the lowest model level and the ground, and below the ground,
temperature and geopotential follow a 6.5 K/km lapse rate from the lowest
level, and u, v and q keep their lowest-level values, close to ECMWF's
extrapolation for ERA5. Nothing is ever above the top level, which sits at
about 3.6 hPa.

q is CAMulator's total water (vapour plus cloud condensate), not specific
humidity. Outside cloud the two are the same.

    import camulatorPressure
    camulatorPressure.convert("run_model_levels.nc", "run.nc",
                              ["u500", "t850", "z500", "t2m"])
"""

import re

import netCDF4
import numpy as np
import xarray as xr

from earth2studio.models.px.camulator import _PHYSICS_FILE, CAMulator

N_LEVELS = 32
P0 = 1.0e5               # Pa; hyam is relative to it
RD = 287.04              # J/kg/K, dry air
RV = 461.5               # J/kg/K, water vapour
G = 9.80616              # m/s^2, CAM's gravity
LAPSE = 0.0065           # K/m, below the lowest model level
ALPHA = RD * LAPSE / G

# Pressure-level base name -> CAMulator model-level base name.
BASES = {"u": "u", "v": "v", "t": "t", "q": "qtot"}

_PL_RE = re.compile(r"^([uvtqz])(\d+)$")


def model_level_inputs(variables):
    """CAMulator variables a run has to save so that `variables`, a mix of
    pressure-level names (u500, z850) and CAMulator's own names, can be
    produced by convert()."""
    needed = []
    pressure = False
    for v in variables:
        m = _PL_RE.match(v)
        if m is None:
            needed.append(v)
            continue
        pressure = True
        base = m.group(1)
        if base in BASES:
            needed += [f"{BASES[base]}{k}k" for k in range(N_LEVELS)]
    if pressure:
        # Level pressures need sp; geopotential and the extrapolation
        # below the lowest level need t and qtot.
        needed += ["sp"] + [f"t{k}k" for k in range(N_LEVELS)] \
            + [f"qtot{k}k" for k in range(N_LEVELS)]
    return list(dict.fromkeys(needed))


def convert(src, dst, variables):
    """Write `variables` to `dst`, read from or interpolated out of the
    CAMulator output `src`. Same dimensions and coordinates as `src`."""
    coeffs, phis = _physics()
    plevels = {}
    passthrough = []
    for v in variables:
        m = _PL_RE.match(v)
        if m is None:
            passthrough.append(v)
        else:
            plevels.setdefault(m.group(1), []).append(int(m.group(2)))

    with netCDF4.Dataset(src) as fin, netCDF4.Dataset(dst, "w") as fout:
        for name, dim in fin.dimensions.items():
            fout.createDimension(name, None if dim.isunlimited() else len(dim))
        for name in fin.dimensions:
            if name in fin.variables:
                _copy_var(fin, fout, name)
        fout.setncatts({k: fin.getncattr(k) for k in fin.ncattrs()})

        missing = [v for v in passthrough if v not in fin.variables]
        if missing:
            raise KeyError(f"{src} has no {missing}")
        for v in passthrough:
            _copy_var(fin, fout, v)

        dims = fin.variables["sp"].dimensions if plevels else None
        out = {}
        for base, levels in plevels.items():
            for p in levels:
                out[f"{base}{p}"] = fout.createVariable(
                    f"{base}{p}", "f4", dims, zlib=True, complevel=1)

        if not plevels:
            return
        lat = np.asarray(fin.variables["lat"][:])
        if lat[0] > lat[-1]:
            phis = phis[::-1]           # physics file runs south to north
        n_time, n_lead = (len(fin.dimensions[d]) for d in dims[:2])
        for i in range(n_time):
            for j in range(n_lead):
                fields = _interpolate(fin, i, j, plevels, coeffs, phis)
                for name, field in fields.items():
                    out[name][i, j] = field


def _interpolate(fin, i, j, plevels, coeffs, phis):
    """Pressure-level fields at time i, lead time j, keyed by output name."""
    def column(base):
        return np.stack([np.asarray(fin.variables[f"{base}{k}k"][i, j], dtype=np.float64)
                         for k in range(N_LEVELS)])

    hyam, hybm, hyai, hybi = (c[:, None, None] for c in coeffs)
    sp = np.asarray(fin.variables["sp"][i, j], dtype=np.float64)
    p = hyam * P0 + hybm * sp               # midpoints, (32, lat, lon)
    p_int = hyai + hybi * sp                # interfaces, (33, lat, lon); hyai is in Pa
    lnp = np.log(p)
    t = column("t")
    tv = t * (1.0 + (RV / RD - 1.0) * column("qtot"))

    # Geopotential at each level's midpoint, integrated up from the
    # surface one layer at a time, as CAM's geopotential_t does.
    phi = np.empty_like(p)
    below = phis.astype(np.float64)
    for k in range(N_LEVELS - 1, -1, -1):
        phi[k] = below + RD * tv[k] * np.log(p_int[k + 1] / p[k])
        below = below + RD * tv[k] * np.log(p_int[k + 1] / p_int[k])

    out = {}
    for base, levels in plevels.items():
        if base == "z":
            field = phi
        else:
            field = t if base == "t" else column(BASES[base])
        for target in levels:
            out[f"{base}{target}"] = _at_pressure(
                field, base, 100.0 * target, p, lnp, t, phi).astype(np.float32)
    return out


def _at_pressure(field, base, target, p, lnp, t, phi):
    """`field` (32, lat, lon) at pressure `target` (Pa), per column."""
    lnt = np.log(target)
    # Index of the last level above the target; p increases with k.
    k = np.clip((p < target).sum(axis=0) - 1, 0, N_LEVELS - 2)
    lo = np.take_along_axis(field, k[None], 0)[0]
    hi = np.take_along_axis(field, k[None] + 1, 0)[0]
    lnp0 = np.take_along_axis(lnp, k[None], 0)[0]
    lnp1 = np.take_along_axis(lnp, k[None] + 1, 0)[0]
    w = np.clip((lnt - lnp0) / (lnp1 - lnp0), 0.0, 1.0)
    value = lo + w * (hi - lo)

    # Below the lowest model level.
    under = target > p[-1]
    if under.any():
        ratio = (target / p[-1]) ** ALPHA
        if base == "t":
            extra = t[-1] * ratio
        elif base == "z":
            extra = phi[-1] - G * t[-1] / LAPSE * (ratio - 1.0)
        else:
            extra = field[-1]
        value = np.where(under, extra, value)
    return value


def _physics():
    ds = xr.open_dataset(CAMulator.load_default_package().resolve(_PHYSICS_FILE))
    coeffs = tuple(ds[c].values.astype(np.float64)
                   for c in ("hyam", "hybm", "hyai", "hybi"))
    return coeffs, ds["PHIS"].values.astype(np.float64)


def _copy_var(fin, fout, name):
    var = fin.variables[name]
    kwargs = {}
    if var.dimensions and name not in fin.dimensions:
        kwargs = dict(zlib=True, complevel=1)
    new = fout.createVariable(name, var.datatype, var.dimensions, **kwargs)
    new.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
    new[:] = var[:]
