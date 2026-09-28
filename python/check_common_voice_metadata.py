# python/check_common_voice_metadata.py

from pathlib import Path
import csv
from collections import defaultdict, Counter

ROOT = Path("dataset/raw/negative/common_voice")

CORPUS_TSV = ROOT / "ss-corpus-en.tsv"
REPORTED_TSV = ROOT / "ss-reported-audios-en.tsv"


def read_tsv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


rows = read_tsv(CORPUS_TSV)
reported_rows = read_tsv(REPORTED_TSV)

print("=" * 70)
print("COMMON VOICE METADATA CHECK")
print("=" * 70)

print(f"Corpus metadata rows : {len(rows)}")
print(f"Reported rows        : {len(reported_rows)}")

# ---------------------------------------------------------------------
# Basic file matching
# ---------------------------------------------------------------------

audio_files = {
    p.name
    for p in ROOT.iterdir()
    if p.is_file() and p.suffix.lower() == ".mp3"
}

metadata_files = {
    row["audio_file"]
    for row in rows
    if row.get("audio_file")
}

print()
print("--- FILE MATCHING ---")
print(f"MP3 files on disk    : {len(audio_files)}")
print(f"Metadata audio files : {len(metadata_files)}")
print(f"Missing on disk      : {len(metadata_files - audio_files)}")
print(f"Unlisted MP3 files   : {len(audio_files - metadata_files)}")

# ---------------------------------------------------------------------
# Speakers
# ---------------------------------------------------------------------

speakers = defaultdict(list)

for row in rows:
    client = row.get("client_id", "").strip()
    if client:
        speakers[client].append(row)

print()
print("--- SPEAKERS ---")
print(f"Unique client IDs    : {len(speakers)}")
print(f"Total recordings     : {len(rows)}")

# ---------------------------------------------------------------------
# Original split distribution
# ---------------------------------------------------------------------

split_counts = Counter(
    row.get("split", "").strip()
    for row in rows
)

print()
print("--- ORIGINAL SPLITS ---")
for split, count in sorted(split_counts.items()):
    print(f"{split or '<empty>':10s}: {count}")

# ---------------------------------------------------------------------
# Does a speaker occur in multiple original splits?
# ---------------------------------------------------------------------

speaker_splits = defaultdict(set)

for row in rows:
    client = row.get("client_id", "").strip()
    split = row.get("split", "").strip()

    if client and split:
        speaker_splits[client].add(split)

cross_split_speakers = {
    client: splits
    for client, splits in speaker_splits.items()
    if len(splits) > 1
}

print()
print("--- SPEAKER / SPLIT CONSISTENCY ---")
print(f"Speakers in >1 split : {len(cross_split_speakers)}")

if cross_split_speakers:
    print()
    print("First 20 cross-split speakers:")
    for client, splits in list(cross_split_speakers.items())[:20]:
        print(f"  {client}: {sorted(splits)}")

# ---------------------------------------------------------------------
# Reported audio
# ---------------------------------------------------------------------

reported_files = {
    row.get("audio_file", "").strip()
    for row in reported_rows
    if row.get("audio_file", "").strip()
}

reported_present = reported_files & audio_files

print()
print("--- REPORTED AUDIO ---")
print(f"Reported audio entries : {len(reported_files)}")
print(f"Reported files present : {len(reported_present)}")
print(f"Unreported recordings  : {len(audio_files - reported_present)}")

# ---------------------------------------------------------------------
# Report reasons
# ---------------------------------------------------------------------

reasons = Counter(
    row.get("reason", "").strip() or "<empty>"
    for row in reported_rows
)

print()
print("--- REPORT REASONS ---")
for reason, count in reasons.most_common():
    print(f"{reason:30s}: {count}")

# ---------------------------------------------------------------------
# Usable speaker count after excluding reported files
# ---------------------------------------------------------------------

usable_rows = [
    row
    for row in rows
    if row.get("audio_file", "").strip() not in reported_files
]

usable_speakers = {
    row.get("client_id", "").strip()
    for row in usable_rows
    if row.get("client_id", "").strip()
}

print()
print("--- USABLE CORPUS ---")
print(f"Usable recordings : {len(usable_rows)}")
print(f"Usable speakers    : {len(usable_speakers)}")

# ---------------------------------------------------------------------
# Duration
# ---------------------------------------------------------------------

duration_ms = []

for row in usable_rows:
    try:
        duration_ms.append(float(row["duration_ms"]))
    except (ValueError, KeyError):
        pass

if duration_ms:
    total_hours = sum(duration_ms) / 1000 / 3600
    avg_sec = sum(duration_ms) / len(duration_ms) / 1000

    print(f"Usable duration    : {total_hours:.2f} hours")
    print(f"Average duration   : {avg_sec:.3f} sec")

print()
print("=" * 70)