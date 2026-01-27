from __future__ import annotations
import os

import numpy as np
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union
from mockfactory import Catalog

ABACUS_HF_ROOT = Path("/dvs_ro/cfs/cdirs/desi/mocks/cai/abacus_HF")

_ZSNAPS_COMMON = {
    "BGS-21.35": (0.300,),
    "LRG": (0.500, 0.725, 0.950),
    "ELG": (0.950, 1.175, 1.475),
}
_ZSNAPS_QSO = {
    "v1": (1.400,),
    "variations": (1.400,),
    "v2": (0.950, 1.250, 1.550, 1.850),
}

# v2 flavors (your update)
_FLAVORS_V2 = {
    "BGS-21.35": ("base", "base_B", "base_dv", "base_B_dv"),
    "LRG": ("base", "base_B", "base_dv", "base_B_dv"),
    "ELG": ("base_conf_nfwexp",),
    "QSO": ("base",),
}

# variations flavors (your update)
_FLAVORS_VAR = {
    "BGS-21.35": ("base", "base_A", "base_B", "base_dv", "base_A_dv", "base_B_dv"),
    "LRG": ("base", "base_dv", "base_B_dv", "base_A_dv"),
    "ELG": ("base_conf", "base_conf_nfwexp"),
    "QSO": ("base", "base_dv"),
}

_NREAL = {"v1": 25, "v2": 25, "variations": 6}

_ZSNAP2ZRANGE = {
    0.300: (0.1, 0.4),
    0.500: (0.4, 0.6),
    0.725: (0.6, 0.8),
    0.950: (0.8, 1.1),
    1.175: (1.1, 1.3),
    1.475: (1.3, 1.6),
    1.250: (1.1, 1.4),
    1.550: (1.4, 1.7),
    1.850: (1.7, 2.1),
    1.400: (0.8, 2.1),
}

def _canon_version(version: str) -> str:
    v = str(version).strip().lower()
    if v in {"v1", "dr2_v1", "dr2_v1.0", "dr2-v1.0", "v1.0"}:
        return "v1"
    if v in {"v2", "dr2_v2", "dr2_v2.0", "dr2-v2.0", "v2.0"}:
        return "v2"
    if v in {"var", "vars", "variation", "variations"}:
        return "variations"
    raise ValueError(f"Unknown version={version!r}. Expected one of: v1, v2, variations.")


def _canon_tracer(tracer: str) -> str:
    t = str(tracer).strip()
    aliases = {
        "bgs-21.35": "BGS-21.35",
        "bgs_21.35": "BGS-21.35",
        "lrg": "LRG",
        "elg": "ELG",
        "qso": "QSO",
    }
    return aliases.get(t.lower(), t)


def _canon_zsnap(zsnap: Union[float, int, str]) -> float:
    if isinstance(zsnap, str):
        s = zsnap.strip().lower()
        if s.startswith("z"):
            s = s[1:]
        z = float(s)
    else:
        z = float(zsnap)
    return z


def _sznap_path_tag(z: float) -> str:
    return f"{z:.3f}".replace(".", "p")


def _allowed_zsnaps(version: str, tracer: str) -> Sequence[float]:
    if tracer == "QSO":
        return _ZSNAPS_QSO[version]
    return _ZSNAPS_COMMON[tracer]


def _allowed_flavors(version: str, tracer: str) -> Optional[Sequence[str]]:
    if version == "v1":
        return None
    if version == "v2":
        return _FLAVORS_V2[tracer]
    return _FLAVORS_VAR[tracer]


def _validate(version: str, tracer: str, z: float, flavor: Optional[str], imock: int) -> None:
    allowed_tracers = set(_ZSNAPS_COMMON) | {"QSO"}
    if tracer not in allowed_tracers:
        raise ValueError(f"Unknown tracer={tracer!r}. Expected one of: {sorted(allowed_tracers)}")

    zsnaps = _allowed_zsnaps(version, tracer)
    if all(abs(z - zz) > 1e-6 for zz in zsnaps):
        raise ValueError(
            f"zsnap={z} not allowed for (version={version}, tracer={tracer}). Allowed: {list(zsnaps)}"
        )

    nreal = _NREAL[version]
    if not (0 <= int(imock) < nreal):
        raise ValueError(f"imock={imock} out of range for version={version}: [0, {nreal-1}]")

    if version == "v1":
        if flavor is not None:
            raise ValueError("v1 paths have no flavor; pass flavor=None.")
    else:
        allowed = _allowed_flavors(version, tracer)
        assert allowed is not None
        if flavor is None:
            flavor = allowed[0]
        if flavor not in allowed:
            raise ValueError(
                f"flavor={flavor!r} not allowed for (version={version}, tracer={tracer}). "
                f"Allowed: {list(allowed)}"
            )


def abacus_hf_mock_path(
    *,
    version: str,
    tracer: str,
    zsnap: Union[float, int, str],
    imock: int,
    flavor: Optional[str] = None,
    cbox: str = "c000",  # only used for variations
    root: Union[str, Path] = ABACUS_HF_ROOT,
    tracer_dir: Optional[str] = None,
    tracer_tag: Optional[str] = None,
) -> Path:

    v = _canon_version(version)
    t = _canon_tracer(tracer)
    z = _canon_zsnap(zsnap)

    tdir = _canon_tracer(tracer_dir) if tracer_dir is not None else t
    ttag = _canon_tracer(tracer_tag) if tracer_tag is not None else t

    _validate(v, t, z, flavor, imock)

    s = _sznap_path_tag(z)
    root = Path(root)

    if v == "v1":
        return (
            root
            / "DR2_v1.0"
            / f"AbacusSummit_base_c000_ph{int(imock):03d}"
            / "Boxes"
            / tdir
            / f"abacus_HF_{ttag}_{s}_DR2_v1.0_AbacusSummit_base_c000_ph{int(imock):03d}_clustering.dat.fits"
        )

    if v == "v2":
        if flavor is None:
            flavor = _FLAVORS_V2[t][0]
        return (
            root
            / "DR2_v2.0"
            / f"AbacusSummit_base_c000_ph{int(imock):03d}"
            / "Boxes"
            / tdir
            / f"abacus_HF_{ttag}_{s}_DR2_v2.0_AbacusSummit_base_c000_ph{int(imock):03d}_{flavor}_clustering.dat.h5"
        )

    if flavor is None:
        flavor = _FLAVORS_VAR[t][0]
    return (
        root
        / "variations"
        / f"AbacusSummit_base_{cbox}_ph{int(imock):03d}"
        / "Boxes"
        / tdir
        / f"abacus_HF_{ttag}_{s}_variations_AbacusSummit_base_{cbox}_ph{int(imock):03d}_{flavor}_clustering.dat.h5"
    )


def abacus_hf_mock_paths(
    *,
    version: str,
    tracer: str,
    zsnap: Union[float, int, str],
    flavor: Optional[str] = None,
    cbox: str = "c000",
    imocks: Optional[Iterable[int]] = None,
    **kwargs,
) -> list[Path]:
    
    v = _canon_version(version)
    n = _NREAL[v]
    ims = list(range(n)) if imocks is None else list(imocks)
    return [
        abacus_hf_mock_path(
            version=v,
            tracer=tracer,
            zsnap=zsnap,
            imock=i,
            flavor=flavor,
            cbox=cbox,
            **kwargs,
        )
        for i in ims
    ]

def zsnap_to_zrange(zsnap: Union[float, int, str], tol: float = 2e-3) -> Tuple[float, float]:
    """Map AbacusHF zsnap to (zmin, zmax) for cubic-box measurement filenames."""
    if isinstance(zsnap, str):
        s = zsnap.strip().lower()
        if s.startswith("z"):
            s = s[1:]
        z = float(s)
    else:
        z = float(zsnap)

    keys = list(_ZSNAP2ZRANGE.keys())
    kbest = min(keys, key=lambda k: abs(k - z))
    if abs(kbest - z) > tol:
        raise ValueError(f"zsnap={z} not recognized (tol={tol}). Known: {sorted(set(keys))}")
    return _ZSNAP2ZRANGE[kbest]


def get_hf_stats_fn(
    stats_dir: Union[str, Path] = Path(os.getenv("SCRATCH", ".")) / "measurements",
    *,
    version: Optional[str] = None,
    kind: str = "mesh2_spectrum",
    tracer: str,
    zsnap: Optional[Union[float, int, str]] = None,
    zrange: Optional[Tuple[float, float]] = None,
    region: str,                    # cbox
    weight: str,                    # HOD flavor
    imock: Union[int, str, None] = None,
    extra: str = "",                # los
    ext: str = "h5",
    battrs: Optional[Iterable[str]] = None,
    basis: Optional[str] = None,
    imock_glob_max: int = 1000,
) -> Union[Path, list[Path]]:

    stats_dir = Path(stats_dir)
    if version:
        stats_dir = stats_dir / str(version)

    if imock == "*":
        out = []
        for i in range(imock_glob_max):
            fn = get_hf_stats_fn(
                stats_dir=stats_dir,
                version=None,  # already applied
                kind=kind,
                tracer=tracer,
                zsnap=zsnap,
                zrange=zrange,
                region=region,
                weight=weight,
                imock=i,
                extra=extra,
                ext=ext,
                battrs=battrs,
                basis=basis,
                imock_glob_max=imock_glob_max,
            )
            if os.path.exists(fn):
                out.append(Path(fn))
        return out

    if zrange is None:
        if zsnap is None:
            raise ValueError("Provide either zsnap or zrange.")
        zrange = zsnap_to_zrange(zsnap)

    ztag = f"z{zrange[0]:.1f}-{zrange[1]:.1f}"
    ztag = f"_{ztag}" if ztag else ""

    extra_s = f"_{extra}" if extra else ""
    imock_s = f"_{imock}" if imock is not None else ""

    corr_type = "smu"
    if battrs is not None:
        corr_type = "".join(list(battrs))

    kind_norm = {
        "mesh2_spectrum": "mesh2_spectrum_poles",
        "particle2_correlation": f"particle2_correlation_{corr_type}",
    }.get(kind, kind)

    if "mesh3" in kind_norm:
        basis_s = f"_{basis}" if basis else ""
        kind_norm = f"mesh3_spectrum{basis_s}_poles"

    basename = f"{kind_norm}_{tracer}{ztag}_{region}_{weight}{extra_s}{imock_s}.{ext}"
    return stats_dir / basename


def get_clustering_positions_weights(fn, los="z", **kwargs):
    import numpy as np
    from mpi4py import MPI

    mpicomm = MPI.COMM_WORLD
    mpiroot = 0

    catalog = None
    boxsize, scalev = None, None

    if mpicomm.rank == mpiroot:
        kwargs = dict()
        catalog = Catalog.read(fn, mpicomm=MPI.COMM_SELF, **kwargs)
        catalog.get(catalog.columns())
        boxsize = catalog.header.get("BOXSIZE", 2000.0)
        scalev = catalog.header.get("VELZ2KMS", None)

    boxsize, scalev = mpicomm.bcast((boxsize, scalev), root=mpiroot)

    if scalev is None:
        from cosmoprimo.fiducial import DESI
        cosmo = DESI()
        a = 1.0 / (1.0 + zsnap)
        E = cosmo.efunc(zsnap)
        scalev = 100.0 * a * E

    if mpicomm.size > 1:
        catalog = Catalog.scatter(catalog, mpicomm=mpicomm, mpiroot=mpiroot)

    for name in catalog.columns():
        catalog[name.upper()] = catalog[name]

    positions = np.column_stack([catalog["X"], catalog["Y"], catalog["Z"]])

    try:
        vsmear = catalog["VSMEAR"]
    except Exception:
        vsmear = np.zeros_like(catalog["VX"])

    velocities = (
        np.column_stack([catalog["VX"] + vsmear, catalog["VY"] + vsmear, catalog["VZ"] + vsmear])
        / scalev
    )

    vlos = los
    if isinstance(los, str):
        vlos = [0.0] * 3
        vlos["xyz".index(los)] = 1.0
    vlos = np.array(vlos)

    positions = positions + np.sum(velocities * vlos, axis=-1)[..., None] * vlos[None, :]
    positions = (positions + boxsize / 2.0) % boxsize - boxsize / 2.0

    return positions, np.ones_like(positions[..., 0])

