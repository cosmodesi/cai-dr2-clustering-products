"""
salloc -N 1 -C gpu -t 02:00:00 --gpus 4 --qos interactive --account desi_g
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
srun -n 4 python run_abacushf_cubic.py
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import time
import logging
import itertools
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Optional
from clustering_statistics.tools_abacushf_cubic import (
    abacus_hf_mock_path,
    get_hf_stats_fn,
    get_clustering_positions_weights
)
from mockfactory import setup_logging

logger = logging.getLogger('run_abacushf_cubic')

def compute_jaxpower_mesh2_spectrum(output_fn, get_data, get_shifted=None, ells=(0, 2, 4), los='z', cache=None, **attrs):
    import jax
    from jaxpower import (ParticleField, FKPField, compute_box2_normalization, compute_fkp2_shotnoise, BinMesh2SpectrumPoles, get_mesh_attrs, compute_mesh2_spectrum)
    mattrs = get_mesh_attrs(boxcenter=0., **attrs)
    data = ParticleField(*get_data(), attrs=mattrs, exchange=True, backend='jax')
    if cache is None: cache = {}
    bin = cache.get('bin_mesh2_spectrum', None)
    if bin is None: bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.001}, ells=ells)
    cache.setdefault('bin_mesh2_spectrum', bin)
    norm = compute_box2_normalization(data, bin=bin)
    wsum_data1 = data.sum()
    if get_shifted is not None:
        data = FKPField(data, ParticleField(*get_shifted(), attrs=mattrs, exchange=True, backend='jax'))
    num_shotnoise = compute_fkp2_shotnoise(data, bin=bin)
    mesh = data.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    mesh = mesh - mesh.mean()
    del data
    jitted_compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'], donate_argnums=[0])
    spectrum = jitted_compute_mesh2_spectrum(mesh, bin=bin, los=los).clone(norm=norm, num_shotnoise=num_shotnoise)
    mattrs = {name: mattrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']}
    spectrum = spectrum.clone(attrs=dict(los=los, wsum_data1=wsum_data1, **mattrs))
    if output_fn is not None and jax.process_index() == 0:
        logger.info(f'Writing to {output_fn}')
        spectrum.write(output_fn)
    return spectrum

def compute_jaxpower_mesh3_spectrum(output_fn, get_data, get_shifted=None, basis='scoccimarro', ells=[0, 2], los='z', mask_edges=None, cache=None, **attrs):
    import jax
    from jaxpower import (ParticleField, FKPField, compute_box3_normalization, compute_fkp3_shotnoise, BinMesh3SpectrumPoles, get_mesh_attrs, compute_mesh3_spectrum)
    mattrs = get_mesh_attrs(boxcenter=0., **attrs)
    data = ParticleField(*get_data(), attrs=mattrs, exchange=True, backend='jax')
    edges = {'step': 0.01 if 'scoccimarro' in basis else 0.005}
    if cache is None: cache = {}
    bin = cache.get(f'bin_mesh3_spectrum_{basis}', None)
    if bin is None: bin = BinMesh3SpectrumPoles(mattrs, edges=edges, basis=basis, ells=ells, buffer_size=16, mask_edges=mask_edges)
    cache.setdefault(f'bin_mesh3_spectrum_{basis}', bin)
    norm = compute_box3_normalization(data, bin=bin)
    if get_shifted is not None:
        data = FKPField(data, ParticleField(*get_shifted(), attrs=mattrs, exchange=True, backend='jax'))
    kw = dict(resampler='tsc', interlacing=3, compensate=True)
    num_shotnoise = compute_fkp3_shotnoise(data, los=los, bin=bin, **kw)
    mesh = data.paint(**kw, out='real')
    mesh = mesh - mesh.mean()
    spectrum = compute_mesh3_spectrum(mesh, los=los, bin=bin)
    spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)
    if output_fn is not None and jax.process_index() == 0:
        logger.info(f'Writing to {output_fn}')
        spectrum.write(output_fn)
    return spectrum

@dataclass(frozen=True)
class Task:
    version: str
    tracer: str
    zsnap: float
    imock: int
    los: str
    cbox: str = "c000"
    flavor: Optional[str] = None


def flavors_for(version: str, tracer: str) -> tuple[Optional[str], ...]:
    if version == "v1":
        return (None,)
    if version == "v2":
        return FLAVORS_V2[tracer]
    return FLAVORS_VAR[tracer]


def iter_tasks(
    tracers: Iterable[tuple[str, float]],
    versions: Iterable[str],
    los_list: Iterable[str],
    cbox_list: Iterable[str],
    imocks: Optional[Iterable[int]] = None,
) -> list[Task]:
    out: list[Task] = []
    for version in versions:
        nreal = NREAL[version]
        ims = list(range(nreal)) if imocks is None else list(imocks)
        for tracer, zsnap in tracers:
            for los in los_list:
                for cbox in (cbox_list if version == "variations" else ["c000"]):
                    for flavor in flavors_for(version, tracer):
                        for imock in ims:
                            out.append(Task(version, tracer, float(zsnap), int(imock), los, cbox, flavor))
    return out

def _maybe_skip(fn: Path, overwrite: bool) -> bool:
    return (not overwrite) and fn.exists()


def run_task(task: Task, todo: set[str], spectrum_args: dict, overwrite: bool = False):
    import jax
    from jaxpower.mesh import create_sharding_mesh
    data_fn = abacus_hf_mock_path(
        version=task.version,
        tracer=task.tracer,
        zsnap=task.zsnap,
        imock=task.imock,
        flavor=task.flavor,
        cbox=task.cbox,
    )

    cache = {}
    get_data = lambda: get_clustering_positions_weights(
        data_fn, 
        los=task.los,
    )


    stats_common = dict(
        stats_dir=STATS_DIR,
        version=task.version,
        tracer=task.tracer,
        zsnap=task.zsnap,
        region=task.cbox,
        weight=(task.flavor or "base"),
        extra=f"los{task.los}",
        imock=task.imock,
        ext="h5",
    )


    if "mesh2_spectrum" in todo:
        out_fn = get_hf_stats_fn(**stats_common, kind="mesh2_spectrum")
        if not _maybe_skip(out_fn, overwrite):
            with create_sharding_mesh():
                compute_jaxpower_mesh2_spectrum(
                    out_fn, get_data, get_shifted=GET_SHIFTED, cache=cache, **spectrum_args
                )
                jax.clear_caches()

    if "mesh3_spectrum_scoccimarro" in todo:
        bargs = spectrum_args | dict(basis="scoccimarro", ells=[0, 2], cellsize=8)
        out_fn = get_hf_stats_fn(**stats_common, kind="mesh3_spectrum", basis=bargs["basis"])
        if not _maybe_skip(out_fn, overwrite):
            with create_sharding_mesh():
                compute_jaxpower_mesh3_spectrum(
                    out_fn, get_data, get_shifted=GET_SHIFTED, cache=cache, **bargs
                )
                jax.clear_caches()

    if "mesh3_spectrum_sugiyama" in todo:
        bargs = spectrum_args | dict(
            basis="sugiyama-diagonal",
            ells=[(0, 0, 0), (2, 0, 2)],
            cellsize=6.25,
        )
        out_fn = get_hf_stats_fn(**stats_common, kind="mesh3_spectrum", basis=bargs["basis"])
        if not _maybe_skip(out_fn, overwrite):
            with create_sharding_mesh():
                compute_jaxpower_mesh3_spectrum(
                    out_fn, get_data, get_shifted=GET_SHIFTED, cache=cache, los=task.los, **bargs
                )
                jax.clear_caches()


# ---------------- config ----------------

TRACER_ZSNAPS = [
    ("LRG", 0.5),
    ("LRG", 0.725),
    ("LRG", 0.95),
    ("ELG", 0.95),
    ("ELG", 1.175),
    ("ELG", 1.475),
    ("QSO", 1.4),
    ("BGS-21.35", 0.3),
]

VERSIONS = ["v2"]          # ["v1", "v2", "variations"]
LOS_LIST = ["z"]           # ["x","y","z"]
CBOX_LIST = ["c000"]       # only used for variations

TODO = {
    "mesh2_spectrum",
    "mesh3_spectrum_scoccimarro",
    "mesh3_spectrum_sugiyama",
}

# ---------------- end config ----------------

SPECTRUM_ARGS = dict(boxsize=2000.,ells=(0, 2, 4), cellsize=5.)   # for mesh2
GET_SHIFTED = None
OVERWRITE = True

NREAL = {"v1": 25, "v2": 25, "variations": 6}

FLAVORS_V2 = {
    "BGS-21.35": ("base", "base_B", "base_dv", "base_B_dv"),
    "LRG": ("base", "base_B", "base_dv", "base_B_dv"),
    "ELG": ("base_conf_nfwexp",),
    "QSO": ("base",),
}
FLAVORS_VAR = {
    "BGS-21.35": ("base", "base_A", "base_B", "base_dv", "base_A_dv", "base_B_dv"),
    "LRG": ("base", "base_dv", "base_B_dv", "base_A_dv"),
    "ELG": ("base_conf", "base_conf_nfwexp"),
    "QSO": ("base", "base_dv"),
}

STATS_DIR = Path(os.getenv("SCRATCH", ".")) / "measurements_abacushf"

def main():
    setup_logging()
    import jax
    from jax import config
    config.update('jax_enable_x64', True)
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'
    jax.distributed.initialize()

    tasks = iter_tasks(TRACER_ZSNAPS, VERSIONS, LOS_LIST, CBOX_LIST)

    for t in tasks:
        run_task(t, TODO, SPECTRUM_ARGS, overwrite=OVERWRITE)


if __name__ == '__main__':
    main()