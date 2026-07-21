#!/usr/bin/env python3
"""Fetch and verify the hyperspectral scenes used in the paper.

Six of the eight scenes (Indian Pines, Salinas, Pavia University, Pavia Centre,
Kennedy Space Center, Botswana) are redistributed openly by the Computational
Intelligence Group of the University of the Basque Country and are downloaded
automatically. Houston 2013 and WHU-Hi-LongKou are only released after a free
registration with their respective providers, so the script prints the exact
manual steps and target filenames for those two.

Every file is checked against the SHA-256 of the copy used to produce the
published numbers, so a successful ``--verify`` run means the local data is
bit-identical to ours.

Usage
-----
    python reproduce/download_data.py --data-root ~/data/HSI
    python reproduce/download_data.py --data-root ~/data/HSI --dataset salinas
    python reproduce/download_data.py --data-root ~/data/HSI --verify
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reproduce.datasets import DATASET_KEYS, SPECS, sha256_of  # noqa: E402

# Direct file locations on the University of the Basque Country wiki. The
# authoritative index is DatasetSpec.source; these are the file links it points
# at. If the layout ever changes, download manually from the source page and
# keep the filenames listed in DatasetSpec.files.
DOWNLOAD_URLS = {
    "indian_pines": {
        "cube": "https://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat",
        "gt": "https://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat",
    },
    "pavia_university": {
        "cube": "https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat",
        "gt": "https://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat",
    },
    "salinas": {
        "cube": "https://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat",
        "gt": "https://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat",
    },
    "botswana": {
        "cube": "https://www.ehu.eus/ccwintco/uploads/7/72/Botswana.mat",
        "gt": "https://www.ehu.eus/ccwintco/uploads/5/58/Botswana_gt.mat",
    },
    "ksc": {
        "cube": "https://www.ehu.eus/ccwintco/uploads/2/26/KSC.mat",
        "gt": "https://www.ehu.eus/ccwintco/uploads/a/a6/KSC_gt.mat",
    },
    "pavia_centre": {
        "cube": "https://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat",
        "gt": "https://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat",
    },
}

# Provider-specific instructions for the two registration-gated scenes.
MANUAL_STEPS = {
    "houston2013": (
        "Request access to the 2013 IEEE GRSS Data Fusion Contest data at\n"
        "  https://machinelearning.ee.uh.edu/2013-ieee-grss-data-fusion-contest/\n"
        "The archive ships the hyperspectral cube and the official train/test\n"
        "masks. Save them as a single MATLAB file holding the variables\n"
        "'input' (349x1905x144), 'TR' and 'TE' (both 349x1905)."
    ),
    "whuhi_longkou": (
        "Download WHU-Hi-LongKou from the RSIDEA group at Wuhan University:\n"
        "  http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm\n"
        "The archive already contains 'WHU_Hi_LongKou.mat' (550x400x270) and\n"
        "'WHU_Hi_LongKou_gt.mat' (550x400)."
    ),
}

USER_AGENT = "Mozilla/5.0 (compatible; Sw-SSFCM-reproduction/1.0)"


def download(url: str, destination: Path) -> None:
    """Download ``url`` to ``destination`` atomically via a .part temp file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response, open(temp, "wb") as out:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    temp.replace(destination)


def check_file(path: Path, expected_sha: str) -> str:
    """Return ``"ok"``, ``"missing"`` or ``"checksum-mismatch"``."""
    if not path.is_file():
        return "missing"
    return "ok" if sha256_of(path) == expected_sha else "checksum-mismatch"


def report(key: str, role: str, path: Path, status: str) -> None:
    marker = {"ok": "OK      ", "missing": "MISSING ", "checksum-mismatch": "MISMATCH"}
    print(f"  [{marker[status]}] {key}/{role}: {path}")


def handle_dataset(key: str, data_root: Path, verify_only: bool) -> bool:
    """Ensure one scene is present and valid. Returns True when it is."""
    spec = SPECS[key]
    print(f"\n{spec.name} ({key}) -- {spec.access} access")
    print(f"  source: {spec.source}")

    complete = True
    for role, relative in spec.files.items():
        path = data_root / relative
        expected = spec.sha256[role]
        status = check_file(path, expected)

        if status == "missing" and not verify_only:
            url = DOWNLOAD_URLS.get(key, {}).get(role)
            if url is None:
                print(f"  [MANUAL  ] {key}/{role} must be obtained by hand:")
                print("    " + MANUAL_STEPS[key].replace("\n", "\n    "))
                print(f"    then place it at: {path}")
                complete = False
                continue
            print(f"  downloading {url}")
            try:
                download(url, path)
            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
                print(f"  [FAILED  ] {exc}")
                print(f"    download manually from {spec.source} and save as {path}")
                complete = False
                continue
            status = check_file(path, expected)

        report(key, role, path, status)
        if status != "ok":
            if status == "checksum-mismatch":
                print(f"    expected SHA-256 {expected}")
                print(f"    actual   SHA-256 {sha256_of(path)}")
            complete = False

    return complete


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-root", required=True, help="directory that will hold the scenes"
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=DATASET_KEYS,
        choices=DATASET_KEYS,
        help="subset of scenes (default: all eight)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="only check what is already on disk, download nothing",
    )
    args = parser.parse_args(argv)

    data_root = Path(args.data_root).expanduser()
    print(f"Data root: {data_root}")

    missing = [
        key for key in args.dataset if not handle_dataset(key, data_root, args.verify)
    ]

    print("\n" + "=" * 70)
    if missing:
        print(f"Incomplete: {', '.join(missing)}")
        print("The remaining scenes are ready to use.")
        sys.exit(1)
    print(f"All requested scenes verified: {', '.join(args.dataset)}")


if __name__ == "__main__":
    main()
