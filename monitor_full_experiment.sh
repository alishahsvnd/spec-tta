#!/bin/bash

# Monitor the full comparison experiment progress

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║              Full Comparison Experiment - Progress Monitor                  ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Find the most recent result directory
RESULT_DIR=$(ls -td results/FULL_COMPARISON_* 2>/dev/null | head -1)

if [ -z "$RESULT_DIR" ]; then
    echo "❌ No experiment in progress"
    exit 1
fi

echo "📁 Result Directory: $RESULT_DIR"
echo ""

# Count completed experiments
SPEC_COUNT=$(wc -l < "$RESULT_DIR/spec_tta_results.csv" 2>/dev/null || echo "1")
PETSA_COUNT=$(wc -l < "$RESULT_DIR/petsa_results.csv" 2>/dev/null || echo "1")
SPEC_COUNT=$((SPEC_COUNT - 1))  # Subtract header
PETSA_COUNT=$((PETSA_COUNT - 1))  # Subtract header

TOTAL=20
COMPLETED=$((SPEC_COUNT < PETSA_COUNT ? SPEC_COUNT : PETSA_COUNT))
PROGRESS=$((COMPLETED * 100 / TOTAL))

echo "📊 Progress: $COMPLETED / $TOTAL experiments ($PROGRESS%)"
echo ""

# Show current progress bars
echo "SPEC-TTA: "
printf "["
for i in $(seq 1 20); do
    if [ $i -le $SPEC_COUNT ]; then
        printf "█"
    else
        printf "░"
    fi
done
printf "] $SPEC_COUNT/20\n"

echo "PETSA:    "
printf "["
for i in $(seq 1 20); do
    if [ $i -le $PETSA_COUNT ]; then
        printf "█"
    else
        printf "░"
    fi
done
printf "] $PETSA_COUNT/20\n"
echo ""

# Show recent activity from main log
echo "📝 Recent Activity:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
tail -15 full_experiment.log 2>/dev/null || echo "No log file yet"
echo ""

# Show current results if any
if [ -f "$RESULT_DIR/spec_tta_results.csv" ] && [ $COMPLETED -gt 0 ]; then
    echo "📈 Current Results Summary:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    python3 - "$RESULT_DIR" << 'PYTHON_END'
import sys
import csv

result_dir = sys.argv[1]

try:
    spec_results = []
    petsa_results = []
    
    with open(f'{result_dir}/spec_tta_results.csv', 'r') as f:
        reader = csv.DictReader(f)
        spec_results = list(reader)
    
    with open(f'{result_dir}/petsa_results.csv', 'r') as f:
        reader = csv.DictReader(f)
        petsa_results = list(reader)
    
    if len(spec_results) > 0 and len(petsa_results) > 0:
        spec_mse_sum = sum(float(r['MSE']) for r in spec_results if r['MSE'])
        petsa_mse_sum = sum(float(r['MSE']) for r in petsa_results if r['MSE'])
        
        spec_avg = spec_mse_sum / len(spec_results) if spec_results else 0
        petsa_avg = petsa_mse_sum / len(petsa_results) if petsa_results else 0
        
        if petsa_avg > 0:
            improvement = ((petsa_avg - spec_avg) / petsa_avg * 100)
            print(f"Average SPEC-TTA MSE: {spec_avg:.4f}")
            print(f"Average PETSA MSE:    {petsa_avg:.4f}")
            print(f"Current Improvement:  {improvement:.1f}%")
        
        # Show last 3 completed
        print(f"\nLast 3 completed experiments:")
        for i, (spec, petsa) in enumerate(zip(spec_results[-3:], petsa_results[-3:])):
            if spec['MSE'] and petsa['MSE']:
                imp = ((float(petsa['MSE']) - float(spec['MSE'])) / float(petsa['MSE']) * 100)
                print(f"  {spec['Model']:<12} H={spec['Horizon']:<4} SPEC={float(spec['MSE']):.4f} PETSA={float(petsa['MSE']):.4f} ({imp:+.1f}%)")

except Exception as e:
    print(f"Error reading results: {e}")

PYTHON_END
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "💡 To monitor live: tail -f full_experiment.log"
echo "💡 To check detailed results: ls -lh $RESULT_DIR/"
echo ""
