#!/usr/bin/env bash
# Resubmit only the (language, seed) combinations that have no results.json.
#
#   bash scripts/recover_missing_runs.sh            # report only
#   bash scripts/recover_missing_runs.sh --submit   # report and resubmit
#
# The task runners have no skip logic: a resubmitted array task retrains from
# scratch. Recovery must therefore target individual missing runs rather than
# whole arrays, or completed work is silently redone.
#
# Preempted tasks frequently DO write their results before being killed, so
# presence of results.json — not the SLURM exit state — is the test.

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ROOT=results/multilingual_evaluation
SEEDS=(42 43 44 45 46)
SUBMIT=0
[[ "${1:-}" == "--submit" ]] && SUBMIT=1

SA_LANGS=(amh arq ary hau ibo kin pcm por swa tso twi yor)
NER_T=(masakhaner2:bam masakhaner2:bbj masakhaner2:ewe masakhaner2:fon
       masakhaner2:hau masakhaner2:ibo masakhaner2:kin masakhaner2:lug
       masakhaner2:luo masakhaner2:mos masakhaner2:nya masakhaner2:pcm
       masakhaner2:sna masakhaner2:swa masakhaner2:tsn masakhaner2:twi
       masakhaner2:wol masakhaner2:xho masakhaner2:yor masakhaner2:zul
       masakhaner_amh: tigrinya_ner:)
QA_DS=(amqa tigqa)

missing_sa=() missing_ner=() missing_qa=() missing_zs=()

for i in "${!SA_LANGS[@]}"; do
  for j in "${!SEEDS[@]}"; do
    [[ -f "$ROOT/sentiment/afrisenti-${SA_LANGS[$i]}-seed${SEEDS[$j]}/results.json" ]] \
      || missing_sa+=("$(( i * 5 + j ))")
  done
done

for i in "${!NER_T[@]}"; do
  t="${NER_T[$i]}"; ds="${t%%:*}"; cfg="${t##*:}"
  name="${ds}${cfg:+-$cfg}"
  for j in "${!SEEDS[@]}"; do
    [[ -f "$ROOT/ner/${name}-seed${SEEDS[$j]}/results.json" ]] \
      || missing_ner+=("$(( i * 5 + j ))")
  done
done

for i in "${!QA_DS[@]}"; do
  for j in "${!SEEDS[@]}"; do
    [[ -f "$ROOT/qa/${QA_DS[$i]}-seed${SEEDS[$j]}/results.json" ]] \
      || missing_qa+=("$(( i * 5 + j ))")
  done
done

for i in 0 1; do
  lang=$([[ $i -eq 0 ]] && echo orm || echo tir)
  for j in "${!SEEDS[@]}"; do
    [[ -f "$ROOT/sentiment_zeroshot/afrisenti-${lang}-from-amh-seed${SEEDS[$j]}/results.json" ]] \
      || missing_zs+=("$(( i * 5 + j ))")
  done
done

join() { local IFS=,; echo "$*"; }

echo "missing SA        : ${#missing_sa[@]}"
echo "missing NER       : ${#missing_ner[@]}"
echo "missing QA        : ${#missing_qa[@]}"
echo "missing zero-shot : ${#missing_zs[@]}"

total=$(( ${#missing_sa[@]} + ${#missing_ner[@]} + ${#missing_qa[@]} + ${#missing_zs[@]} ))
if [[ $total -eq 0 ]]; then
  echo "nothing to recover — every (language, seed) has a results.json"
  exit 0
fi

if [[ $SUBMIT -eq 0 ]]; then
  echo
  echo "re-run with --submit to resubmit these array indices:"
  [[ ${#missing_sa[@]}  -gt 0 ]] && echo "  SA        $(join "${missing_sa[@]}")"
  [[ ${#missing_ner[@]} -gt 0 ]] && echo "  NER       $(join "${missing_ner[@]}")"
  [[ ${#missing_qa[@]}  -gt 0 ]] && echo "  QA        $(join "${missing_qa[@]}")"
  [[ ${#missing_zs[@]}  -gt 0 ]] && echo "  zero-shot $(join "${missing_zs[@]}")"
  exit 0
fi

ids=()
[[ ${#missing_sa[@]}  -gt 0 ]] && ids+=("$(sbatch --parsable --array="$(join "${missing_sa[@]}")"  scripts/slurm_multilingual_sa.sh)")
[[ ${#missing_ner[@]} -gt 0 ]] && ids+=("$(sbatch --parsable --array="$(join "${missing_ner[@]}")" scripts/slurm_multilingual_ner.sh)")
[[ ${#missing_qa[@]}  -gt 0 ]] && ids+=("$(sbatch --parsable --array="$(join "${missing_qa[@]}")"  scripts/slurm_multilingual_qa.sh)")
[[ ${#missing_zs[@]}  -gt 0 ]] && ids+=("$(sbatch --parsable --array="$(join "${missing_zs[@]}")"  scripts/slurm_zeroshot_sa.sh)")

echo "submitted: ${ids[*]}"
echo
echo "chain aggregation behind them with:"
echo "  sbatch --dependency=afterany:$(join "${ids[@]}") scripts/slurm_aggregate_multilingual.sh"
