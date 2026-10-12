#!/usr/bin/env bash
# Run every example, capture status, timeout at 60s each.
set -u
cd "$(dirname "$0")"

RESULTS_DIR="run_logs"
mkdir -p "$RESULTS_DIR"
SUMMARY="$RESULTS_DIR/SUMMARY.txt"
> "$SUMMARY"

printf "%-40s %-8s %-10s %s\n" "EXAMPLE" "STATUS" "TIME" "NOTE" | tee -a "$SUMMARY"
printf "%-40s %-8s %-10s %s\n" "-------" "------" "----" "----" | tee -a "$SUMMARY"

for f in examples/*.py; do
    name=$(basename "$f")
    log="$RESULTS_DIR/${name%.py}.log"
    start=$(date +%s.%N)
    timeout 60 python3 "$f" > "$log" 2>&1
    rc=$?
    end=$(date +%s.%N)
    elapsed=$(printf "%.2f" "$(echo "$end - $start" | bc)")

    case $rc in
        0)   status="OK"     ; note=$(grep -iE "fidelity|concurrence|speedup|PASS|FAIL|OK|correct" "$log" | tail -1 | cut -c1-60) ;;
        124) status="TIMEOUT"; note="exceeded 60s" ;;
        *)   status="FAIL"   ; note=$(tail -1 "$log" | cut -c1-60) ;;
    esac

    printf "%-40s %-8s %-10s %s\n" "$name" "$status" "$elapsed" "$note" | tee -a "$SUMMARY"
done

echo
echo "Full logs in $RESULTS_DIR/"
