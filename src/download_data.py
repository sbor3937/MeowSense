"""Download the CatMeows dataset from Zenodo.

CatMeows (Ludovico, Ntalampiras, Presti, Cannas, Battini & Mattiello, 2020) is
hosted on Zenodo as record 4008297 and licensed CC BY 4.0. It is *not*
redistributed in this repository -- this script fetches it from the original
source so that attribution and licensing stay with the authors.

The record contains two archives:

* ``dataset.zip`` (~8.9 MB) -- the 440 mono 8 kHz WAV vocalizations. Required.
* ``extras.zip``  (~4.1 MB) -- supplementary material. Optional.

Usage::

    python src/download_data.py                  # -> data/raw/
    python src/download_data.py --with-extras    # also fetch extras.zip
    python src/download_data.py --force          # re-download and overwrite

The download is idempotent: if the expected number of WAV files is already
present, the script exits early instead of re-fetching.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ZENODO_RECORD_ID = "4008297"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
EXPECTED_N_WAV = 440

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"


def _fetch_record_metadata(timeout: int = 60) -> dict:
    """Fetch the Zenodo record's JSON metadata.

    Args:
        timeout: Socket timeout in seconds.

    Returns:
        The decoded Zenodo record.

    Raises:
        RuntimeError: If the record cannot be retrieved or parsed.
    """
    try:
        with urllib.request.urlopen(ZENODO_API_URL, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - surface any network/parse failure alike
        raise RuntimeError(
            f"Could not fetch Zenodo record {ZENODO_RECORD_ID}: {exc}\n"
            f"Check your connection, or download manually from "
            f"https://doi.org/10.5281/zenodo.{ZENODO_RECORD_ID}"
        ) from exc


def _file_links(record: dict) -> dict[str, str]:
    """Map filename -> download URL for every file in the record."""
    return {f["key"]: f["links"]["self"] for f in record.get("files", [])}


def _md5(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute the MD5 of a file, reading it in chunks."""
    digest = hashlib.md5()  # noqa: S324 - integrity check only, not security
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path, timeout: int = 300) -> None:
    """Stream ``url`` to ``dest``, reporting progress on a single line.

    Downloads to a ``.part`` file first and renames on success, so an
    interrupted run cannot leave a truncated archive that looks complete.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    with urllib.request.urlopen(url, timeout=timeout) as response:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        with tmp.open("wb") as fh:
            while chunk := response.read(1 << 16):
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100.0 * downloaded / total
                    print(
                        f"\r  {dest.name}: {downloaded / 1e6:6.2f} / "
                        f"{total / 1e6:6.2f} MB ({pct:5.1f}%)",
                        end="",
                        flush=True,
                    )
    print()
    tmp.replace(dest)


def _extract_wavs(archive: Path, raw_dir: Path) -> int:
    """Extract every ``.wav`` in ``archive`` into a flat ``raw_dir``.

    The archive nests WAVs inside a subdirectory; we flatten them because all
    metadata lives in the filename itself (see :mod:`features`), so the
    directory structure carries no information.

    Args:
        archive: Path to the downloaded ``.zip``.
        raw_dir: Destination directory.

    Returns:
        Number of WAV files written.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            if not member.lower().endswith(".wav"):
                continue
            name = Path(member).name
            # Guard against path traversal in a malformed archive.
            if not name or name.startswith("."):
                continue
            with zf.open(member) as src, (raw_dir / name).open("wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1

    return count


def download_catmeows(
    data_dir: Path = DEFAULT_DATA_DIR,
    with_extras: bool = False,
    force: bool = False,
) -> Path:
    """Download and extract CatMeows.

    Args:
        data_dir: Root data directory. WAVs land in ``data_dir / "raw"`` and
            archives are cached in ``data_dir / "archives"``.
        with_extras: Also download the optional ``extras.zip``.
        force: Re-download and re-extract even if the data is already present.

    Returns:
        The directory containing the extracted WAV files.

    Raises:
        RuntimeError: If the record cannot be fetched or ``dataset.zip`` is
            missing from it.
    """
    raw_dir = data_dir / "raw"
    archive_dir = data_dir / "archives"

    existing = list(raw_dir.glob("*.wav")) if raw_dir.exists() else []
    if existing and not force:
        print(f"Found {len(existing)} WAV files already in {raw_dir} -- skipping.")
        print("Pass --force to re-download.")
        return raw_dir

    print(f"Querying Zenodo record {ZENODO_RECORD_ID} ...")
    record = _fetch_record_metadata()
    links = _file_links(record)

    print(f"  title:   {record.get('title', '?')}")
    print(f"  doi:     {record.get('doi', '?')}")
    print(f"  license: {record.get('metadata', {}).get('license', {}).get('id', '?')}")

    wanted = ["dataset.zip"] + (["extras.zip"] if with_extras else [])
    for name in wanted:
        if name not in links:
            raise RuntimeError(
                f"{name!r} is not present in Zenodo record {ZENODO_RECORD_ID}. "
                f"Available: {sorted(links)}"
            )

    for name in wanted:
        archive = archive_dir / name
        if archive.exists() and not force:
            print(f"Using cached {archive}")
        else:
            print(f"Downloading {name} ...")
            _download(links[name], archive)
        print(f"  md5: {_md5(archive)}")

    print("Extracting WAV files ...")
    n_wav = _extract_wavs(archive_dir / "dataset.zip", raw_dir)
    print(f"  extracted {n_wav} WAV files to {raw_dir}")

    if n_wav != EXPECTED_N_WAV:
        print(
            f"  WARNING: expected {EXPECTED_N_WAV} WAV files but got {n_wav}. "
            "The upstream record may have changed.",
            file=sys.stderr,
        )

    return raw_dir


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download the CatMeows dataset (Zenodo record 4008297).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Root data directory (default: ./data).",
    )
    parser.add_argument(
        "--with-extras",
        action="store_true",
        help="Also download the optional extras.zip.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the data is already present.",
    )
    args = parser.parse_args()

    try:
        raw_dir = download_catmeows(
            data_dir=args.data_dir, with_extras=args.with_extras, force=args.force
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"\nDone. Dataset ready at {raw_dir}")
    print("Next: python src/train_baseline.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
