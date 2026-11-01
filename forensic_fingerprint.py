import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import imagehash
from PIL import Image

from config import FORENSIC_LEDGER


def calculate_sha256(file_path: str | os.PathLike[str]) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    sha256_hash = hashlib.sha256()
    with file_path.open("rb") as file:
        for byte_block in iter(lambda: file.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def calculate_phash(image: Image.Image) -> str:
    """Return the perceptual hash of a PIL image."""
    return str(imagehash.phash(image))


def generate_digital_fingerprint(file_path):
    """Generate cryptographic and perceptual fingerprints for forensic traceability."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    crypto_hash = calculate_sha256(file_path)

    try:
        with Image.open(file_path) as pil_img:
            p_hash = calculate_phash(pil_img)
    except Exception:
        p_hash = "N/A (Video / Non-Image File)"

    return {
        "file_name": os.path.basename(file_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sha256": crypto_hash,
        "phash": p_hash,
    }


def record_forensic_log(metadata_record, log_file=FORENSIC_LEDGER):
    """Append a forensic fingerprint to the integrity ledger."""
    ledger = []
    log_path = Path(log_file)
    if log_path.exists():
        try:
            with log_path.open("r", encoding="utf-8") as file:
                ledger = json.load(file)
        except Exception:
            ledger = []

    ledger.append(metadata_record)
    with log_path.open("w", encoding="utf-8") as file:
        json.dump(ledger, file, indent=4)
    print(f"Recorded forensic fingerprint into: {log_path}")


def main():
    record_forensic_log(generate_digital_fingerprint(__file__))


if __name__ == "__main__":
    main()