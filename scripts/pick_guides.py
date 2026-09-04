"""Pick realistic CRISPR guides out of a genome file.

The Week 1-2 target list was written by hand, so most of its sites have
no PAM and every hit is correctly scored 'low'. To exercise the Week 3
scoring properly you want guides that a real design tool would return:
20 nt protospacers that are actually followed by an NGG PAM in the
reference, so their off-targets can genuinely be cut.

    python scripts/pick_guides.py --input data/genome.fasta \
        --output data/guides.csv --count 5

Each guide is taken from a different record and checked for the PAM
before it is written, so the resulting file is guaranteed to produce
scored, tiered hits.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parser import iter_fasta_records, load_sequence_dataset  # noqa: E402
from src.scoring import matches_motif  # noqa: E402

GUIDE_LENGTH = 20
VALID = set("ACGT")


def find_guide(sequence, guide_length=GUIDE_LENGTH, stride=7):
    """First window in `sequence` that is clean DNA followed by NGG."""
    limit = len(sequence) - guide_length - 3

    for start in range(0, max(0, limit), stride):
        guide = sequence[start:start + guide_length]
        pam = sequence[start + guide_length:start + guide_length + 3]

        if len(guide) < guide_length or len(pam) < 3:
            continue

        if not set(guide).issubset(VALID):
            continue

        if matches_motif(pam, "NGG"):
            return guide, pam, start

    return None


def main():
    parser = argparse.ArgumentParser(description="Pick PAM-adjacent guides")
    parser.add_argument("--input", default="data/genome.fasta")
    parser.add_argument("--output", default="data/guides.csv")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument(
        "--skip",
        type=int,
        default=37,
        help="Take a guide from every Nth record, so they are not neighbours",
    )

    args = parser.parse_args()

    guides = []
    seen = set()

    for index, (record_id, sequence) in enumerate(iter_fasta_records(args.input)):
        if index % args.skip:
            continue

        found = find_guide(sequence.upper())

        if not found:
            continue

        guide, pam, start = found

        if guide in seen:
            continue

        seen.add(guide)
        guides.append((guide, pam, record_id, start))

        if len(guides) >= args.count:
            break

    if not guides:
        raise SystemExit("No PAM-adjacent guides found in " + args.input)

    directory = os.path.dirname(args.output)

    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write("target,pam,source_record,source_position\n")

        for guide, pam, record_id, start in guides:
            handle.write(f"{guide},{pam},{record_id},{start}\n")

    print(f"Wrote {len(guides)} PAM-adjacent guides to {args.output}")

    for guide, pam, record_id, start in guides:
        print(f"  {guide} {pam}  ({record_id} @ {start})")


if __name__ == "__main__":
    main()
