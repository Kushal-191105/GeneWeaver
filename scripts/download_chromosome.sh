set -euo pipefail

CHROMOSOME="${1:-21}"
RELEASE="${2:-115}"

BASE="https://ftp.ensembl.org/pub/release-${RELEASE}/fasta/homo_sapiens/dna"
FILE="Homo_sapiens.GRCh38.dna.chromosome.${CHROMOSOME}.fa.gz"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$(dirname "${SCRIPT_DIR}")/data"

mkdir -p "${DATA_DIR}"

echo "Downloading chromosome ${CHROMOSOME} (Ensembl release ${RELEASE})..."
curl -L --fail --progress-bar -o "${DATA_DIR}/${FILE}" "${BASE}/${FILE}"

echo
echo "Saved to data/${FILE}"
du -h "${DATA_DIR}/${FILE}" | cut -f1 | xargs echo "Size:"
echo
echo "Scan it with:"
echo "  python main.py --input data/${FILE} --limit 0 --mode gpu"
echo "  python -m src.dashboard --input data/${FILE} --limit 0"
