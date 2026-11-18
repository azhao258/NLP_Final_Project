set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python verify_assets.py
python e1/check_local.py
mkdir -p e1_results
mode=pilot
if [ "$#" -gt 0 ]; then mode="$1"; fi
case "$mode" in
 pilot) python -u e1/evaluate.py --model codi --limit 5 --output e1_results/codi_course_pilot --resume 2>&1 | tee -a e1_results/course_pilot.log ;;
 full) python -u e1/evaluate.py --model codi --output e1_results/codi_course_full --resume 2>&1 | tee -a e1_results/course_full.log ;;
 svamp) python -u e1/evaluate.py --model codi --datasets svamp --output e1_results/codi_svamp_course_test --resume 2>&1 | tee -a e1_results/svamp_course_test.log ;;
 *) echo "Usage: bash e1/run_e1.sh pilot|full|svamp"; exit 2 ;;
esac
