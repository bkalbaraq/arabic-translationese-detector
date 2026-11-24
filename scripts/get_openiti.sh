#!/usr/bin/env bash
# Download classical Arabic books from OpenITI (one GitHub repo per 25-year block
# of author death dates). Shallow clones, ~150–900 MB each.
#
#   bash scripts/get_openiti.sh            # default set below (~6–10 GB)
#   bash scripts/get_openiti.sh 0275 0325  # only these blocks
#
# Default blocks are chosen to cover the strongest prose eras and genres:
#   0150–0300  early adab & history (ابن المقفع، الجاحظ، ابن قتيبة، المبرد، البلاذري)
#   0325–0400  الطبري، القالي، أبو الفرج الأصفهاني، التنوخي
#   0425–0600  ابن حزم، الثعالبي، الحريري، ابن الجوزي
#   0650–0850  ابن الأثير، ابن خلكان، ابن عذاري، ابن خلدون، ابن حجر
set -euo pipefail
OUT="${OPENITI_DIR:-openiti}"
mkdir -p "$OUT"
BLOCKS=("$@")
if [ ${#BLOCKS[@]} -eq 0 ]; then
  BLOCKS=(0150 0175 0200 0225 0250 0275 0300 0325 0350 0375 0400 0425 0450 0475 0500 0550 0600 0650 0700 0725 0750 0800 0850)
fi
for b in "${BLOCKS[@]}"; do
  repo="${b}AH"
  dest="$OUT/$(echo "$repo" | tr 'A-Z' 'a-z')"
  if [ -d "$dest/.git" ]; then echo "have $repo"; continue; fi
  echo "cloning $repo ..."
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 "https://github.com/OpenITI/${repo}" "$dest" || echo "!! failed $repo"
done
echo "done. repos in $OUT/"
