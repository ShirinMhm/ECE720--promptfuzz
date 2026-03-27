#!/bin/bash
# budget_experiment.sh
#
# Runs three paired experiments for the budget-aware multi-fidelity extension:
#
#   1. BASELINE      — original full evaluation (no scheduler)
#   2. MULTIFIDELITY — staged evaluation with default params (20/50/100%)
#   3. AGGRESSIVE    — tighter stages (15/35/100%) with higher beta
#
# After all runs, get_metric.py can be used to compare bestASR, ESR, and
# coverage across the three conditions at the same query budget.
#
# Usage:
#   cd Experiment
#   bash budget_experiment.sh --openai_key sk-... [--model gpt-3.5-turbo-0125]
#
# Results land in:
#   Results/focus/hijacking/baseline_full/
#   Results/focus/hijacking/mf_default/
#   Results/focus/hijacking/mf_aggressive/

set -e

OPENAI_KEY=""
MODEL="gpt-3.5-turbo-0125"
MODE="hijacking"
PHASE="focus"
MAX_QUERY=2000       # fixed budget for fair comparison

# --- Parse args ---
while [[ $# -gt 0 ]]; do
  case $1 in
    --openai_key) OPENAI_KEY="$2"; shift 2 ;;
    --model)      MODEL="$2";      shift 2 ;;
    --mode)       MODE="$2";       shift 2 ;;
    --max_query)  MAX_QUERY="$2";  shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$OPENAI_KEY" ]]; then
  echo "Error: --openai_key required"
  exit 1
fi

COMMON_ARGS="--openai_key $OPENAI_KEY --model_path $MODEL \
             --mode $MODE --phase $PHASE --all_defenses \
             --max_query $MAX_QUERY --few_shot"

echo "============================================================"
echo " Budget-Aware PromptFuzz Experiments"
echo " Mode=$MODE  Phase=$PHASE  Budget=$MAX_QUERY queries"
echo "============================================================"

# ── 1. Baseline (full evaluation) ──────────────────────────────
echo ""
echo "[1/3] Running BASELINE (full evaluation)..."
python run.py $COMMON_ARGS \
  --result_prefix "baseline_full"

# ── 2. Multi-fidelity default (20% → 50% → 100%) ───────────────
echo ""
echo "[2/3] Running MULTIFIDELITY default (stages: 0.2 0.5 1.0)..."
python run.py $COMMON_ARGS \
  --multifidelity \
  --stage_fractions 0.2 0.5 1.0 \
  --budget_beta 1.0 \
  --top_k_fractions 0.5 0.5 \
  --promotion_threshold 0.0 \
  --result_prefix "mf_default"

# ── 3. Multi-fidelity aggressive (15% → 35% → 100%) ────────────
echo ""
echo "[3/3] Running MULTIFIDELITY aggressive (stages: 0.15 0.35 1.0, beta=2.0)..."
python run.py $COMMON_ARGS \
  --multifidelity \
  --stage_fractions 0.15 0.35 1.0 \
  --budget_beta 2.0 \
  --top_k_fractions 0.4 0.5 \
  --promotion_threshold 0.0 \
  --result_prefix "mf_aggressive"

echo ""
echo "============================================================"
echo " All runs complete.  Compute metrics with:"
echo "   python get_metric.py --phase $PHASE --mode $MODE"
echo "============================================================"
