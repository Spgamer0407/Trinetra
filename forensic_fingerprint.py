import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

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


def generate_digital_fingerprint(file_path: str | os.PathLike[str], extra_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Generate cryptographic and perceptual fingerprints for forensic traceability."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    crypto_hash = calculate_sha256(file_path)

    try:
        with Image.open(file_path) as pil_img:
            p_hash = calculate_phash(pil_img)
    except Exception:
        p_hash = "N/A (Video / Non-Image File)"

    record = {
        "file_name": file_path.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sha256": crypto_hash,
        "phash": p_hash,
    }

    if extra_metadata:
        record.update(extra_metadata)

    return record


def record_forensic_log(metadata_record: Dict[str, Any], log_file: str | os.PathLike[str] = FORENSIC_LEDGER, allow_duplicates: bool = False) -> bool:
    """Append a forensic fingerprint to the integrity ledger."""
    log_path = Path(log_file)
    ledger = []
    if log_path.exists():
        try:
            with log_path.open("r", encoding="utf-8") as file:
                ledger = json.load(file)
                if not isinstance(ledger, list):
                    ledger = []
        except Exception:
            ledger = []

    # Check for existing SHA-256 to prevent duplicate entries if not allowed
    if not allow_duplicates:
        for item in ledger:
            if item.get("sha256") == metadata_record.get("sha256"):
                print(f"[Ledger] Record with SHA-256 '{metadata_record.get('sha256')}' already exists in ledger.")
                return False

    ledger.append(metadata_record)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as file:
        json.dump(ledger, file, indent=4)
    print(f"Recorded forensic fingerprint into: {log_path}")
    return True


def verify_integrity(
    file_path: str | os.PathLike[str],
    log_file: str | os.PathLike[str] = FORENSIC_LEDGER,
    exact_threshold: int = 0,
    perceptual_threshold: int = 10,
) -> Dict[str, Any]:
    """
    Verify the forensic integrity and authenticity of a file against the ledger.
    1. Checks for bit-level exact match using SHA-256.
    2. Checks for perceptual similarity using pHash Hamming distance.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    current_fingerprint = generate_digital_fingerprint(file_path)
    current_sha256 = current_fingerprint["sha256"]
    current_phash = current_fingerprint["phash"]

    log_path = Path(log_file)
    if not log_path.exists():
        return {
            "verified": False,
            "status": "LEDGER_NOT_FOUND",
            "message": f"Ledger file not found at '{log_path}'. No reference records available.",
            "query_fingerprint": current_fingerprint,
            "matched_record": None,
            "hamming_distance": None,
        }

    try:
        with log_path.open("r", encoding="utf-8") as file:
            ledger = json.load(file)
            if not isinstance(ledger, list):
                ledger = []
    except Exception as exc:
        return {
            "verified": False,
            "status": "LEDGER_READ_ERROR",
            "message": f"Could not read ledger: {exc}",
            "query_fingerprint": current_fingerprint,
            "matched_record": None,
            "hamming_distance": None,
        }

    # Step 1: Check for exact SHA-256 match
    for record in ledger:
        if record.get("sha256") == current_sha256:
            return {
                "verified": True,
                "status": "VERIFIED_EXACT_MATCH",
                "message": (
                    f"Verified! Exact cryptographic SHA-256 match found in ledger "
                    f"(registered as '{record.get('file_name')}' on {record.get('timestamp')}). "
                    "Content is 100% unaltered (bit-level integrity preserved)."
                ),
                "query_fingerprint": current_fingerprint,
                "matched_record": record,
                "hamming_distance": 0,
            }

    # Step 2: If image, check for perceptual match via pHash
    if current_phash != "N/A (Video / Non-Image File)":
        try:
            curr_hash_obj = imagehash.hex_to_hash(current_phash)
            best_match = None
            min_dist = float("inf")

            for record in ledger:
                rec_phash = record.get("phash")
                if rec_phash and rec_phash != "N/A (Video / Non-Image File)":
                    try:
                        rec_hash_obj = imagehash.hex_to_hash(rec_phash)
                        dist = curr_hash_obj - rec_hash_obj
                        if dist < min_dist:
                            min_dist = dist
                            best_match = record
                    except Exception:
                        continue

            if best_match is not None:
                if min_dist <= 5:
                    return {
                        "verified": True,
                        "status": "VERIFIED_PERCEPTUAL_MATCH",
                        "message": (
                            f"Verified with perceptual similarity! Hamming distance is {min_dist} (<= 5). "
                            f"Matches '{best_match.get('file_name')}'. Content is visually authentic "
                            "with minor format, compression, or resolution differences."
                        ),
                        "query_fingerprint": current_fingerprint,
                        "matched_record": best_match,
                        "hamming_distance": int(min_dist),
                    }
                elif min_dist <= perceptual_threshold:
                    return {
                        "verified": False,
                        "status": "SUSPICIOUS_SIMILARITY",
                        "message": (
                            f"Suspicious similarity detected. Hamming distance is {min_dist} "
                            f"(between 6 and {perceptual_threshold}) compared to '{best_match.get('file_name')}'. "
                            "Content shares visual structure but may have undergone localized tampering or edits."
                        ),
                        "query_fingerprint": current_fingerprint,
                        "matched_record": best_match,
                        "hamming_distance": int(min_dist),
                    }
        except Exception as exc:
            print(f"[Warning] Error during pHash comparison: {exc}")

    return {
        "verified": False,
        "status": "NOT_FOUND",
        "message": "No matching cryptographic or perceptual fingerprint found in the forensic ledger.",
        "query_fingerprint": current_fingerprint,
        "matched_record": None,
        "hamming_distance": None,
    }


def main():
    test_fp = generate_digital_fingerprint(__file__)
    record_forensic_log(test_fp)
    result = verify_integrity(__file__)
    print(f"Self-verification result: {result['status']} - {result['message']}")


if __name__ == "__main__":
    main()
