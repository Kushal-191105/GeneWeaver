"""Biological scoring for CRISPR off-target hits.

Week 3 - Scoring Logic.

An alignment tells you *where* a guide RNA could bind. It does not tell
you whether that binding matters. Two hits with the same raw mismatch
count can differ by two orders of magnitude in how likely Cas9 is to
actually cut there, and the difference is almost entirely *where* the
mismatches sit relative to the PAM.

Three biological facts drive this module:

1. **PAM dependence.** SpCas9 cannot cut without a 5'-NGG-3' PAM
   immediately 3' of the 20 nt protospacer. No PAM, no cut - however
   good the alignment looks.
2. **PAM-proximal seed.** The ~12 nt closest to the PAM are the seed
   region. A single mismatch there usually abolishes cleavage; the same
   mismatch at the distal 5' end is often tolerated completely.
3. **Substitution identity.** rG:dT and rU:dG wobble pairs still stack,
   so transition mismatches (A<->G, C<->T) are far better tolerated than
   transversions.

The position weights are the Hsu et al. (2013) MIT specificity weights,
the same vector used by the MIT CRISPR design tool, and the aggregation
follows the published MIT off-target score:

    score = 100 * PROD(1 - W[m]) * 1 / (((19 - d) / 19) * 4 + 1) * 1 / n^2

where W[m] is the weight of each mismatched position, d is the mean
pairwise distance between mismatches, and n is the mismatch count. The
score runs 0-100 and reads as "how much of the on-target cutting
activity survives" - so a HIGH score is a DANGEROUS off-target.
"""

import numpy as np

# Hsu et al. (2013), positions 1..20 counted 5' -> 3' along the
# protospacer, i.e. index 19 is the base closest to the PAM. Higher
# weight = a mismatch there costs more activity.
HSU_POSITION_WEIGHTS = (
    0.000, 0.000, 0.014, 0.000, 0.000,
    0.395, 0.317, 0.000, 0.389, 0.079,
    0.445, 0.508, 0.613, 0.851, 0.732,
    0.828, 0.615, 0.804, 0.685, 0.583,
)

# Length of the PAM-proximal seed region, counted back from the 3' end.
SEED_LENGTH = 12

# Transitions (purine<->purine, pyrimidine<->pyrimidine) form wobble
# pairs that Cas9 tolerates; transversions distort the duplex much more.
TRANSITIONS = frozenset({("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")})

TRANSITION_TOLERANCE = 1.15   # transitions keep ~15% more activity
TRANSVERSION_TOLERANCE = 0.85  # transversions lose ~15% more

# Canonical SpCas9 PAM: any base, then GG.
PAM_MOTIF = "NGG"
PAM_LENGTH = len(PAM_MOTIF)

IUPAC = {
    "A": set("A"), "C": set("C"), "G": set("G"), "T": set("T"),
    "R": set("AG"), "Y": set("CT"), "S": set("GC"), "W": set("AT"),
    "K": set("GT"), "M": set("AC"), "B": set("CGT"), "D": set("AGT"),
    "H": set("ACT"), "V": set("ACG"), "N": set("ACGT"),
}

SEVERITY_TIERS = (
    # (minimum score, label, colour used by the TUI and the web UI)
    (50.0, "critical", "#ff3b30"),
    (20.0, "high", "#ff9500"),
    (5.0, "moderate", "#ffd60a"),
    (0.0, "low", "#34c759"),
)

SEVERITY_ORDER = {"critical": 3, "high": 2, "moderate": 1, "low": 0}


# ----------------------------------------------------------------------
# mismatch geometry
# ----------------------------------------------------------------------

def mismatch_positions(target, candidate):
    """0-based indices where `candidate` differs from `target`.

    Index 0 is the 5' (PAM-distal) end, so the last index is the base
    sitting right against the PAM.
    """
    return [
        index
        for index, (expected, found) in enumerate(zip(target, candidate))
        if expected != found
    ]


def seed_mismatches(positions, target_length):
    """Mismatches that fall inside the PAM-proximal seed region."""
    seed_start = max(0, target_length - SEED_LENGTH)

    return [position for position in positions if position >= seed_start]


def mean_pairwise_distance(positions):
    """Mean distance between mismatches, as the MIT score defines it.

    Clustered mismatches are more disruptive than spread-out ones, so
    the published formula folds this in. With fewer than two mismatches
    the term is defined as 19 (maximum spread, no penalty).
    """
    if len(positions) < 2:
        return 19.0

    return float(max(positions) - min(positions)) / (len(positions) - 1)


def position_weight(index, target_length):
    """Hsu weight for a mismatch, rescaled for non-20 nt guides.

    The published vector is defined for a 20 nt protospacer. A shorter
    or longer target is mapped onto the same 20-slot curve so the
    PAM-proximal end still carries the heavy weights.
    """
    if target_length <= 0:
        return 0.0

    if target_length == 20:
        return HSU_POSITION_WEIGHTS[index]

    scaled = int(round(index * 19.0 / (target_length - 1))) if target_length > 1 else 19

    return HSU_POSITION_WEIGHTS[min(19, max(0, scaled))]


def substitution_factor(expected, found):
    """Activity multiplier for one substitution's chemistry."""
    if (expected, found) in TRANSITIONS:
        return TRANSITION_TOLERANCE

    return TRANSVERSION_TOLERANCE


# ----------------------------------------------------------------------
# PAM handling
# ----------------------------------------------------------------------

def matches_motif(sequence, motif=PAM_MOTIF):
    """True when `sequence` satisfies an IUPAC motif such as NGG."""
    if len(sequence) != len(motif):
        return False

    for base, code in zip(sequence.upper(), motif.upper()):
        if base not in IUPAC.get(code, set()):
            return False

    return True


def extract_pam(genome, position, target_length, motif=PAM_MOTIF):
    """The bases immediately 3' of a hit, or '' when the record ends."""
    start = position + target_length
    pam = genome[start:start + len(motif)]

    return pam if len(pam) == len(motif) else ""


def pam_status(pam, motif=PAM_MOTIF):
    """'ngg' when the PAM is canonical, 'absent' when off the end, else 'none'."""
    if not pam:
        return "absent"

    return "ngg" if matches_motif(pam, motif) else "none"


# ----------------------------------------------------------------------
# the score
# ----------------------------------------------------------------------

def mit_score(target, candidate):
    """MIT (Hsu 2013) off-target score, 0-100.

    100 means the site is indistinguishable from the intended target;
    near 0 means Cas9 is very unlikely to cut there.
    """
    target = target.upper()
    candidate = candidate.upper()

    length = min(len(target), len(candidate))

    if length == 0:
        return 0.0

    positions = mismatch_positions(target[:length], candidate[:length])
    count = len(positions)

    if count == 0:
        return 100.0

    activity = 1.0

    for index in positions:
        weight = position_weight(index, length)
        factor = substitution_factor(target[index], candidate[index])

        # A tolerated substitution shaves the penalty, an intolerant one
        # deepens it, while a zero-weight position stays free either way.
        penalty = min(1.0, weight * (2.0 - factor))
        activity *= (1.0 - penalty)

    distance = mean_pairwise_distance(positions)
    distance_term = 1.0 / (((19.0 - distance) / 19.0) * 4.0 + 1.0)
    count_term = 1.0 / (count ** 2)

    return round(100.0 * activity * distance_term * count_term, 4)


def severity_for(score, pam):
    """Tier a hit by its score, with no-PAM hits capped at 'low'.

    A perfect 20/20 alignment without an NGG downstream cannot be cut by
    SpCas9, so it never earns a high tier no matter how the alignment
    scores.
    """
    if pam == "none":
        return "low"

    for minimum, label, _ in SEVERITY_TIERS:
        if score >= minimum:
            return label

    return "low"


def severity_colour(severity):
    for _, label, colour in SEVERITY_TIERS:
        if label == severity:
            return colour

    return "#8e8e93"


def score_match(match, genome=None, motif=PAM_MOTIF):
    """Annotate one alignment hit with its biological interpretation.

    `match` is a pipeline hit dict (sequence_id, target, position,
    sequence, mismatches). `genome` is the record the hit came from,
    used only to read the PAM sitting 3' of the site; pass None and the
    PAM is reported as 'absent'.

    Returns a new dict with the original keys plus:
        score, severity, mismatch_positions, seed_mismatches, pam,
        pam_status, alignment (per-base 'match'/'mismatch' string).
    """
    target = str(match["target"]).upper()
    found = str(match["sequence"]).upper()

    positions = mismatch_positions(target, found)
    length = len(target)

    if genome is not None:
        pam = extract_pam(genome, int(match["position"]), length, motif)
    else:
        pam = ""

    status = pam_status(pam, motif)
    score = mit_score(target, found)

    scored = dict(match)
    scored.update({
        "score": score,
        "severity": severity_for(score, status),
        "mismatch_positions": positions,
        "seed_mismatches": len(seed_mismatches(positions, length)),
        "pam": pam,
        "pam_status": status,
        "alignment": "".join(
            "." if index not in positions else "x" for index in range(length)
        ),
    })

    return scored


def score_matches(matches, sequences=None, motif=PAM_MOTIF):
    """Score a list of hits.

    `sequences` maps sequence_id -> full record text so the PAM can be
    read; omit it and every hit is scored without PAM context.
    """
    sequences = sequences or {}

    return [
        score_match(match, sequences.get(match.get("sequence_id")), motif)
        for match in matches
    ]


def rank_matches(matches, limit=None):
    """Most dangerous first: severity, then score, then fewest mismatches."""
    ranked = sorted(
        matches,
        key=lambda match: (
            SEVERITY_ORDER.get(match.get("severity", "low"), 0),
            match.get("score", 0.0),
            -int(match.get("mismatches", 0)),
        ),
        reverse=True,
    )

    return ranked[:limit] if limit else ranked


def severity_counts(matches):
    """How many hits landed in each tier, for the summary panels."""
    counts = {label: 0 for _, label, _ in SEVERITY_TIERS}

    for match in matches:
        label = match.get("severity", "low")
        counts[label] = counts.get(label, 0) + 1

    return counts


def scoring_matrix(target_length=20):
    """The position-weight curve itself, for display and for tests.

    Returns a list of dicts, one per protospacer position, describing
    how much activity a mismatch there destroys and whether it sits in
    the seed region.
    """
    seed_start = max(0, target_length - SEED_LENGTH)

    rows = []

    for index in range(target_length):
        weight = position_weight(index, target_length)

        rows.append({
            "position": index + 1,
            "distance_to_pam": target_length - index,
            "weight": round(weight, 3),
            "retained_activity": round(1.0 - weight, 3),
            "region": "seed" if index >= seed_start else "distal",
        })

    return rows


def weight_vector(target_length=20):
    """The same curve as a numpy array, for vectorized scoring."""
    return np.array(
        [position_weight(index, target_length) for index in range(target_length)],
        dtype=np.float64,
    )


def describe():
    """One-paragraph explanation used by the UIs' help panels."""
    return (
        "Hits are ranked with the MIT (Hsu 2013) specificity score: each "
        "mismatch is penalised by its distance to the PAM, transitions are "
        "tolerated more than transversions, clustered mismatches cost more "
        "than spread ones, and a site with no NGG PAM is capped at low "
        "severity because SpCas9 cannot cut it."
    )
