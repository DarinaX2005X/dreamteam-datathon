"""Fetch official EKB source text without browser automation or personal signatures.

This is a research/update command, never part of inference. Existing snapshots are
kept unless --refresh is supplied. Quotes are independently checked by
verify_sources.py before a changed catalogue is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
DOCUMENTS = {
    "S231": "136659",
    "S14": "139854",
    "S25-2026": "223567",
    "S78-2026": "227270",
    "S21-2024": "196550",
}


def fetch(name: str, code: str, checked_at: str, refresh: bool) -> dict:
    target = ROOT / "primary" / f"{name}.json"
    if target.exists() and not refresh:
        return json.loads(target.read_text(encoding="utf-8"))
    url = f"https://zan.gov.kz/api/documents/{code}/rus"
    request = Request(url, headers={"User-Agent": "Normative-source-verification/1.0"})
    with urlopen(request, timeout=30) as response:
        payload = response.read()
    data = json.loads(payload)
    # The API also returns the signer's certificate and personal identifiers.
    # They are irrelevant to a visual-rule catalogue and are not retained.
    snapshot = {
        "source_url": url,
        "checked_at": checked_at,
        "response_sha256": hashlib.sha256(payload).hexdigest(),
        "registry_code": data["ngr"],
        "source_version": data["version"]["id"],
        "edition_checked": data["versionDate"],
        "actual_version": data["actualVersion"],
        "title": data["metadata"]["title"]["rus"],
        "act": data["metadata"]["requisites"]["rus"],
        "initial_publication_date": data.get("initialPublicationDate"),
        "content": [
            {key: node[key] for key in ("id", "type", "level", "index", "text") if key in node}
            for node in data["content"]
        ],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return snapshot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checked-at", required=True, help="Verification date, YYYY-MM-DD")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    for name, code in DOCUMENTS.items():
        result = fetch(name, code, args.checked_at, args.refresh)
        print(f"{name}: {result['registry_code']}, edition={result['edition_checked']}, "
              f"actual={result['actual_version']}, nodes={len(result['content'])}")
