#!/bin/bash
# Backup existing checkpoints before training from scratch

cd /home/alishah/PETSA || exit 1

BACKUP_DIR="./checkpoints_backup_$(date +%Y%m%d_%H%M%S)"
DATASET="ETTh1"
PRED_LEN=96

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        Backup Existing Checkpoints                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Backup Directory: $BACKUP_DIR"
echo ""

# Create backup directory
mkdir -p "$BACKUP_DIR"

BACKBONES=("iTransformer" "DLinear" "PatchTST" "MICN" "FreTS")

echo "📦 Backing up checkpoints..."
echo ""

BACKED_UP=0
for MODEL in "${BACKBONES[@]}"; do
    SRC="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}/"
    if [ -d "$SRC" ]; then
        DST="${BACKUP_DIR}/${MODEL}/${DATASET}_${PRED_LEN}/"
        echo "  • $MODEL: Backing up..."
        mkdir -p "$DST"
        cp -r "$SRC"* "$DST" 2>/dev/null
        if [ -f "${DST}/checkpoint_best.pth" ]; then
            SIZE=$(du -h "${DST}/checkpoint_best.pth" | cut -f1)
            echo "    ✅ Backed up (size: $SIZE)"
            ((BACKED_UP++))
        else
            echo "    ⚠️  No checkpoint found"
        fi
    else
        echo "  • $MODEL: No checkpoint directory found"
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Backup complete: $BACKED_UP checkpoints backed up"
echo "📁 Location: $BACKUP_DIR"
echo ""

# List backup contents
if [ $BACKED_UP -gt 0 ]; then
    echo "Backup contents:"
    du -sh "${BACKUP_DIR}"/*/* 2>/dev/null | sed 's/^/  /'
    echo ""
fi
