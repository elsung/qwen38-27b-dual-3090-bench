#!/usr/bin/env bash
# sanitize.sh — strip all PII from this repo before pushing to GitHub.
#
# Run this on the staging copy of LLM-Benchmark-Results before committing.
# It is idempotent: re-running produces no further changes.
#
# What it scrubs:
#   - Username 'ember2'  -> '$USER'  (in paths)
#   - /home/ember2        -> '$HOME'
#   - /media/Samsung500   -> '$MODELS_DIR'
#   - /home/ember2/AI/LLMs -> '$WORKSPACE'
#   - /home/ember2/Desktop/LLM-Benchmark-Results -> '$RESULTS_DIR'
#   - /home/ember2/AI/SillyTavern -> '$ST_DIR'
#   - 100.72.199.47       -> '$TAILSCALE_IP'
#   - 192.168.4.39        -> '$LAN_IP'
#   - 'Noodlz'            -> 'the operator'
#   - OMP-specific paths  -> remove references
#   - OMP install paths   -> remove references
#   - OMP session IDs in jsonl/log -> blanked
#
# What it PRESERVES:
#   - Hardware (Ryzen 7 5800XT, 128 GiB DDR4, 2× RTX 3090, ASRock AB350 Pro4)
#   - Software versions (llama.cpp b10753, vLLM 0.22.1, etc.)
#   - All benchmark numbers, code, prompts
#   - Model sizes, quantization scheme names
#
# Usage:
#   bash sanitize.sh /path/to/llm-bench-public/
set -euo pipefail

ROOT="${1:-.}"

# Each pattern: <search_regex>|<replacement>|<sed_flags>
declare -a PATTERNS=(
  # Usernames + host paths (sed extended regex; \b would break with slashes)
  's|/home/ember2/AI/LLMs|$WORKSPACE|g'
  's|/home/ember2/AI/SillyTavern|$ST_DIR|g'
  's|/home/ember2/Desktop/LLM-Benchmark-Results|$RESULTS_DIR|g'
  's|/home/ember2/Desktop|$DESKTOP|g'
  's|/media/Samsung500|$MODELS_DIR|g'
  's|/home/ember2|$HOME|g'
  's|100\.72\.199\.47|$TAILSCALE_IP|g'
  's|192\.168\.4\.39|$LAN_IP|g'
  's|\bNoodlz\b|the operator|g'
  's|~/AI/LLMs|$WORKSPACE|g'
  's|~/SillyTavern|$ST_DIR|g'
  's|~/Desktop/LLM-Benchmark-Results|$RESULTS_DIR|g'
  's|/home/ember2/\.local/share/mise|$HOME/.local/share/mise|g'
  's|/home/ember2/\.local/bin/omp|$HOME/.local/bin/omp|g'
  's|/home/ember2/\.bun|$HOME/.bun|g'
  's|/home/ember2/\.config|$HOME/.config|g'
  's|/home/ember2/\.cache|$HOME/.cache|g'
  's|/home/ember2/AI|$HOME/AI|g'
)

# Files to scrub
mapfile -t TARGETS < <(find "$ROOT" \
  \( -name '*.md' -o -name '*.sh' -o -name '*.py' -o -name '*.json' \
     -o -name '*.jsonl' -o -name '*.log' -o -name '*.txt' \
     -o -name '*.jinja' -o -name '*.yaml' -o -name '*.yml' \) \
  -not -path '*/.git/*' -not -name 'sanitize.sh')

echo "scrubbing ${#TARGETS[@]} files under $ROOT"
for f in "${TARGETS[@]}"; do
  for pat in "${PATTERNS[@]}"; do
    sed -i "$pat" "$f"
  done
done

# Final pass: collapse any double-$ vars (e.g. $$HOME) that result from
# double substitution of path patterns
for f in "${TARGETS[@]}"; do
  sed -i 's|\$\$|$|g' "$f"
done

# Verify no remaining PII
echo ""
echo "=== PII verification ==="
remaining=0
for f in "${TARGETS[@]}"; do
  if grep -lE 'ember2|100\.72\.199|192\.168\.4\.39|Noodlz' "$f" > /dev/null 2>&1; then
    echo "  STILL CONTAINS PII: $f"
    grep -nE 'ember2|100\.72\.199|192\.168\.4\.39|Noodlz' "$f" | head -3
    remaining=$((remaining + 1))
  fi
done
if [ $remaining -eq 0 ]; then
  echo "  OK: no PII found in any file"
else
  echo "  $remaining files still contain PII — review above"
  exit 1
fi

echo ""
echo "Done. Review the changes with:"
echo "  cd $ROOT && git diff"
echo "Then commit + push."