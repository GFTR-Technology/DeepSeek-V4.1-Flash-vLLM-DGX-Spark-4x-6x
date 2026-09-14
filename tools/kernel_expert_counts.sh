#!/usr/bin/env bash
# Which expert counts will the fused routing kernel accept?
#
# The draft's expert count has to divide TP, so patch/dsv41_tp_pad pads it with
# dead experts. But the count is ALSO a template parameter of the routing
# kernel, which rejects anything it was not compiled for:
#
#   RuntimeError: topkGatingSoftplusSqrtKernelLauncher,
#   .../moe/topk_softplus_sqrt_kernels.cu:841, Unsupported expert number: 132
#
# So the pad target has to satisfy both. This prints the values the kernel
# actually dispatches on, and the smallest legal pad for the TP you name.
#
#   ./tools/kernel_expert_counts.sh          # TP from cluster.env
#   ./tools/kernel_expert_counts.sh 6
#   DSV41_VLLM_SRC=/src/vllm-dsv41 ./tools/kernel_expert_counts.sh 6
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

TP="${1:-}"
if [ -z "$TP" ]; then
  # shellcheck source=../scripts/lib.sh
  . "$REPO/scripts/lib.sh"
  require_cluster
  TP="$NNODES"
fi

KERNEL="moe/topk_softplus_sqrt_kernels.cu"
FOUND=""
for root in \
  "${DSV41_VLLM_SRC:-}" \
  /src/vllm-dsv41 \
  "$REPO/build/vllm" \
  "$REPO/build/vllm-dsv41" \
  "$HOME/vllm" \
  /workspace
do
  [ -n "$root" ] || continue
  hit="$(find "$root" -path "*$KERNEL" -type f 2>/dev/null | head -1)"
  [ -n "$hit" ] && { FOUND="$hit"; break; }
done

if [ -z "$FOUND" ]; then
  echo "Could not find $KERNEL."
  echo
  echo "It is in the full vllm checkout, not in build/vllm (which is only the"
  echo "Python tree). Point at it explicitly:"
  echo "  DSV41_VLLM_SRC=/path/to/vllm $0 $TP"
  echo
  echo "Or read it out of the image instead, which needs no checkout:"
  echo "  docker run --rm --entrypoint sh \"\$IMAGE\" -c \\"
  echo "    'grep -rn \"Unsupported expert number\" -B60 /workspace/csrc 2>/dev/null | head -80'"
  exit 1
fi

echo "kernel: $FOUND"
echo

# The launcher dispatches with a case/if per supported count. Pull the integer
# literals out of whatever form this build uses -- a switch on case labels, or a
# chain of `num_experts == N`.
counts="$(
  {
    grep -oE 'case[[:space:]]+([0-9]+)[[:space:]]*:' "$FOUND" | grep -oE '[0-9]+'
    grep -oE '(num_experts|n_experts|numExperts)[[:space:]]*==[[:space:]]*([0-9]+)' "$FOUND" \
      | grep -oE '[0-9]+$'
    grep -oE 'LAUNCH[A-Z_]*\([[:space:]]*([0-9]+)' "$FOUND" | grep -oE '[0-9]+'
  } 2>/dev/null | sort -n -u
)"

if [ -z "$counts" ]; then
  echo "No dispatch literals found. Read the launcher by hand:"
  echo "  grep -n 'Unsupported expert number' -B 60 '$FOUND'"
  exit 1
fi

echo "expert counts this kernel dispatches on:"
echo "$counts" | paste -sd' ' -
echo

echo "legal pad targets for the draft's 128 experts at TP=$TP:"
best=""
for n in $counts; do
  if [ "$n" -ge 128 ] && [ $((n % TP)) -eq 0 ]; then
    echo "  $n   (+$((n - 128)) dead experts per block)"
    [ -z "$best" ] && best="$n"
  fi
done
if [ -z "$best" ]; then
  echo "  none. Every supported count >= 128 fails to divide TP=$TP."
  echo "  Run this TP without the draft: DSV41_SPEC=none ./scripts/dsv41-serve.sh"
  exit 2
fi
echo
echo "smallest: DSV41_DRAFT_EXPERTS=$best DSV41_LANE=300k ./scripts/dsv41-serve.sh"
