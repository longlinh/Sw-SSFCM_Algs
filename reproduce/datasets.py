"""Hyperspectral scene loading, label sampling and synthetic data generation.

This module defines the eight benchmark scenes used in the paper, a loader that
turns each one into the flat ``(N, B)`` representation expected by SeFCM and
Sw-SSFCM, and a fully synthetic scene generator so that the whole pipeline can
be exercised without downloading any external data.

Pixel ordering is row-major throughout: ``X[i]`` corresponds to image position
``(i // width, i % width)``, which is what ``SwSSFCM.fit`` assumes.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Scene registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetSpec:
    """Static description of one benchmark scene.

    Attributes
    ----------
    name : human-readable scene name as used in the paper.
    files : mapping ``role -> path relative to the data root``.
    keys : mapping ``role -> variable name inside the MATLAB file``.
    n_clusters : number of ground-truth classes C.
    shape : expected ``(height, width, bands)``, used as an integrity check.
    sha256 : mapping ``role -> SHA-256 of the reference copy of the file``.
    source : official distribution page.
    access : ``"open"`` for direct download, ``"registration"`` when the
        provider requires a (free) account before the archive can be fetched.
    """

    name: str
    files: dict
    keys: dict
    n_clusters: int
    shape: tuple
    sha256: dict
    source: str
    access: str


SPECS: dict = {
    "indian_pines": DatasetSpec(
        name="Indian Pines",
        files={
            "cube": "Indian_Pines/Indian_pines_corrected.mat",
            "gt": "Indian_Pines/Indian_pines_gt.mat",
        },
        keys={"cube": "indian_pines_corrected", "gt": "indian_pines_gt"},
        n_clusters=16,
        shape=(145, 145, 200),
        sha256={
            "cube": "ec2f8808710919d566f70f0d4aa885aae1ddfd42b734aba71c5e12ca65450939",
            "gt": "65c4687a8ab04f6da4789799bc3bc4f6e88bccac3ed6a2e6ae367e5e6b9e429c",
        },
        source="https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes",
        access="open",
    ),
    "pavia_university": DatasetSpec(
        name="Pavia University",
        files={
            "cube": "Pavia_University/PaviaU.mat",
            "gt": "Pavia_University/PaviaU_gt.mat",
        },
        keys={"cube": "paviaU", "gt": "paviaU_gt"},
        n_clusters=9,
        shape=(610, 340, 103),
        sha256={
            "cube": "28447fa87f7a5797845e9a189c0da85e23b1d06a4ba7361e5ff44efbf834d2fb",
            "gt": "23f6a426928f9b32984adffe659e29f554f9fb6c93b5a107528d308d5087a829",
        },
        source="https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes",
        access="open",
    ),
    "salinas": DatasetSpec(
        name="Salinas",
        files={
            "cube": "Salinas/Salinas_corrected.mat",
            "gt": "Salinas/Salinas_gt.mat",
        },
        keys={"cube": "salinas_corrected", "gt": "salinas_gt"},
        n_clusters=16,
        shape=(512, 217, 204),
        sha256={
            "cube": "5ec1c0d22f56d18ecd336f8e35735863c0f160682e04e0c18ef3f89a3334d87d",
            "gt": "ecfab4d31ef5553f097943235d8ea502038eb4a2067b2ad10b33e37c949955e2",
        },
        source="https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes",
        access="open",
    ),
    "botswana": DatasetSpec(
        name="Botswana",
        files={"cube": "Botswana/Botswana.mat", "gt": "Botswana/Botswana_gt.mat"},
        keys={"cube": "Botswana", "gt": "Botswana_gt"},
        n_clusters=14,
        shape=(1476, 256, 145),
        sha256={
            "cube": "f1603903c844cdc2980550b0180688e8e1a72d4292595d1120e1dec2a80a91c7",
            "gt": "668394905e10e629c16584bfd02b0f533b96d6ba18a63274a94ff3a77126a887",
        },
        source="https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes",
        access="open",
    ),
    "ksc": DatasetSpec(
        name="Kennedy Space Center",
        files={
            "cube": "Kennedy_Space_Center/KSC.mat",
            "gt": "Kennedy_Space_Center/KSC_gt.mat",
        },
        keys={"cube": "KSC", "gt": "KSC_gt"},
        n_clusters=13,
        shape=(512, 614, 176),
        sha256={
            "cube": "b1ad011cfdb65c853e4f9f6108ca4774467d87f90a5c23b74ff3a2984a3b4786",
            "gt": "a1d6ab9293691006bd4d9742d1a1e1c141b1aaa5fbc5fa128b33c1d09038510b",
        },
        source="https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes",
        access="open",
    ),
    "pavia_centre": DatasetSpec(
        name="Pavia Centre",
        files={"cube": "Pavia_Centre/Pavia.mat", "gt": "Pavia_Centre/Pavia_gt.mat"},
        keys={"cube": "pavia", "gt": "pavia_gt"},
        n_clusters=9,
        shape=(1096, 715, 102),
        sha256={
            "cube": "b60341da323fd271cd97dd6d7a69514d385ccd7fdfc5a60c4ff5994b047f152e",
            "gt": "7eb54ab81b404dd3ae572e6d0ec211595029e5a847bec2cfb8f9e72d07066b69",
        },
        source="https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes",
        access="open",
    ),
    "houston2013": DatasetSpec(
        name="Houston 2013",
        # Single file holding cube ``input`` plus the official DFC train/test
        # masks ``TR``/``TE``; ground truth is their union.
        files={"cube": "Houston/Houston.mat"},
        keys={"cube": "input", "train_mask": "TR", "test_mask": "TE"},
        n_clusters=15,
        shape=(349, 1905, 144),
        sha256={
            "cube": "554663a0f33bbfba28d6921317b7b9c7698747587144a9778848ced5d7419eab"
        },
        source="https://machinelearning.ee.uh.edu/2013-ieee-grss-data-fusion-contest/",
        access="registration",
    ),
    "whuhi_longkou": DatasetSpec(
        name="WHU-Hi-LongKou",
        files={
            "cube": "WHU-Hi-LongKou/WHU_Hi_LongKou.mat",
            "gt": "WHU-Hi-LongKou/WHU_Hi_LongKou_gt.mat",
        },
        keys={"cube": "WHU_Hi_LongKou", "gt": "WHU_Hi_LongKou_gt"},
        n_clusters=9,
        shape=(550, 400, 270),
        sha256={
            "cube": "4ef90f2222128245b8fc5c9c6003dfc37f43569991190eb32fcc8a3753430634",
            "gt": "b5a3634dc31a907ea6e60192ad7cd2b7946453ec57526ea7724941d8c1078178",
        },
        source="http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm",
        access="registration",
    ),
}

DATASET_KEYS = list(SPECS)


# ---------------------------------------------------------------------------
# Scene container
# ---------------------------------------------------------------------------
@dataclass
class Scene:
    """A loaded scene, flattened to the shape the estimators consume.

    Attributes
    ----------
    key, name : scene identifier and display name.
    X : ndarray (N, B), band-wise standardized reflectance, row-major pixels.
    y_true : ndarray (N,), 0-indexed class per pixel, ``-1`` for background.
    valid_mask : ndarray (N,) of bool, ``True`` where ground truth exists.
    height, width : image dimensions, with ``height * width == N``.
    n_clusters : number of classes C.
    """

    key: str
    name: str
    X: np.ndarray
    y_true: np.ndarray
    valid_mask: np.ndarray
    height: int
    width: int
    n_clusters: int

    @property
    def n_bands(self) -> int:
        return self.X.shape[1]

    @property
    def n_pixels(self) -> int:
        return self.X.shape[0]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def standardize(X: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance scaling per band (population standard deviation).

    Equivalent to ``sklearn.preprocessing.StandardScaler().fit_transform(X)``.
    """
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    return (X - mean) / std


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    """Stream a file through SHA-256 and return the hex digest."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mat(path: Path, key: str) -> np.ndarray:
    from scipy.io import loadmat

    content = loadmat(str(path))
    if key not in content:
        available = [k for k in content if not k.startswith("__")]
        raise KeyError(f"{path.name}: variable '{key}' not found (has {available})")
    return np.asarray(content[key])


def load_scene(key: str, data_root: str | Path) -> Scene:
    """Load one benchmark scene from ``data_root``.

    Ground-truth value 0 means "unlabelled background" in every distribution
    used here, so classes are shifted to the 0-indexed convention and
    background pixels receive ``-1``.
    """
    if key not in SPECS:
        raise KeyError(f"unknown dataset '{key}'; available: {DATASET_KEYS}")
    spec = SPECS[key]
    root = Path(data_root)

    for role, rel in spec.files.items():
        if not (root / rel).is_file():
            raise FileNotFoundError(
                f"{spec.name}: missing {role} file '{root / rel}'. "
                f"Run 'python reproduce/download_data.py --dataset {key}' first."
            )

    cube = _load_mat(root / spec.files["cube"], spec.keys["cube"]).astype(np.float64)
    if cube.ndim != 3:
        raise ValueError(f"{spec.name}: expected a 3-D cube, got shape {cube.shape}")
    if cube.shape != spec.shape:
        raise ValueError(
            f"{spec.name}: expected cube shape {spec.shape}, got {cube.shape}"
        )
    height, width, bands = cube.shape

    if key == "houston2013":
        train = _load_mat(root / spec.files["cube"], spec.keys["train_mask"])
        test = _load_mat(root / spec.files["cube"], spec.keys["test_mask"])
        gt = np.where(train > 0, train, test).astype(int).reshape(-1)
    else:
        gt = _load_mat(root / spec.files["gt"], spec.keys["gt"]).astype(int).reshape(-1)

    valid_mask = gt > 0
    y_true = np.full(gt.shape, -1, dtype=int)
    y_true[valid_mask] = gt[valid_mask] - 1

    n_classes = int(np.unique(y_true[valid_mask]).size)
    if n_classes != spec.n_clusters:
        raise ValueError(
            f"{spec.name}: expected {spec.n_clusters} classes, found {n_classes}"
        )

    return Scene(
        key=key,
        name=spec.name,
        X=standardize(cube.reshape(height * width, bands)),
        y_true=y_true,
        valid_mask=valid_mask,
        height=height,
        width=width,
        n_clusters=spec.n_clusters,
    )


def stratified_labels(
    y_true: np.ndarray,
    valid_mask: np.ndarray,
    n_per_class: int,
    seed: int = 42,
) -> np.ndarray:
    """Draw at most ``n_per_class`` labelled pixels per class, without replacement.

    Returns the partial-label vector expected by ``fit``: class index for the
    selected pixels, ``-1`` everywhere else. Classes smaller than
    ``n_per_class`` contribute all of their pixels.
    """
    rng = np.random.default_rng(seed)
    selected = []
    for cls in np.unique(y_true[valid_mask]):
        idx = np.where(y_true == cls)[0]
        selected.extend(rng.choice(idx, size=min(n_per_class, len(idx)), replace=False))
    y_partial = np.full_like(y_true, -1)
    y_partial[np.asarray(selected, dtype=int)] = y_true[np.asarray(selected, dtype=int)]
    return y_partial


# ---------------------------------------------------------------------------
# Synthetic scene
# ---------------------------------------------------------------------------
def make_synthetic_scene(
    height: int = 64,
    width: int = 64,
    n_bands: int = 30,
    n_clusters: int = 6,
    noise: float = 2.0,
    background_fraction: float = 0.15,
    seed: int = 42,
) -> Scene:
    """Build a synthetic scene with the statistical structure of a real HSI.

    Class regions are produced by a Voronoi tessellation of random seed points,
    which yields the spatially contiguous patches that the spatial term of
    Sw-SSFCM exploits. Every class gets a smooth, class-specific spectral
    signature; pixels are that signature plus Gaussian noise. A random subset of
    pixels is marked as background (``y_true == -1``) to mimic the partial
    ground-truth coverage of real benchmark scenes.

    The default noise level puts the scene in the same regime as the real
    benchmarks: hard enough that unguided FCM performs poorly, easy enough that
    guidance and spatial weighting each give a visible gain.

    The result is a drop-in replacement for ``load_scene`` output, so the whole
    reproduction pipeline can run without any external download.
    """
    rng = np.random.default_rng(seed)

    # Voronoi class map: each pixel takes the class of its nearest seed point.
    seeds_yx = rng.integers(0, [height, width], size=(n_clusters * 3, 2))
    seed_class = np.arange(len(seeds_yx)) % n_clusters
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    grid = np.stack([yy.ravel(), xx.ravel()], axis=1)
    dist = (
        (grid[:, None, 0] - seeds_yx[None, :, 0]) ** 2
        + (grid[:, None, 1] - seeds_yx[None, :, 1]) ** 2
    )
    labels = seed_class[dist.argmin(axis=1)]

    # Smooth per-class spectra: a Gaussian absorption profile over the bands.
    band_axis = np.linspace(0.0, 1.0, n_bands)[None, :]
    centre = np.linspace(0.15, 0.85, n_clusters)[:, None]
    amplitude = rng.uniform(1.5, 3.0, size=(n_clusters, 1))
    offset = rng.uniform(-0.5, 0.5, size=(n_clusters, 1))
    signatures = offset + amplitude * np.exp(-((band_axis - centre) ** 2) / 0.02)

    X = signatures[labels] + rng.normal(0.0, noise, size=(height * width, n_bands))

    y_true = labels.astype(int)
    background = rng.random(height * width) < background_fraction
    y_true[background] = -1
    valid_mask = y_true >= 0

    return Scene(
        key="synthetic",
        name="Synthetic HSI",
        X=standardize(X),
        y_true=y_true,
        valid_mask=valid_mask,
        height=height,
        width=width,
        n_clusters=n_clusters,
    )
