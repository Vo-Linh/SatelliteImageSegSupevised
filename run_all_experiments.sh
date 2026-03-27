#!/usr/bin/env bash
# ---------------------------------------------------------------
# Run All Supervised Segmentation Experiments
# Each framework x {before_fusion, after_fusion} = 12 experiments
#
# Usage:
#   bash run_all_experiments.sh          # single GPU
#   bash run_all_experiments.sh 4        # 4 GPUs distributed
# ---------------------------------------------------------------

GPUS=${1:-1}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# All configuration files
CONFIGS=(
    # DDRNet
    "configs/ddrnet/ddrnet_dapcn_before_fusion_cityscapes.py"
    "configs/ddrnet/ddrnet_dapcn_after_fusion_cityscapes.py"
    # PIDNet
    "configs/pidnet/pidnet_dapcn_before_fusion_cityscapes.py"
    "configs/pidnet/pidnet_dapcn_after_fusion_cityscapes.py"
    # KNet
    "configs/knet/knet_dapcn_before_fusion_cityscapes.py"
    "configs/knet/knet_dapcn_after_fusion_cityscapes.py"
    # SegFormer
    "configs/segformer/segformer_dapcn_before_fusion_cityscapes.py"
    "configs/segformer/segformer_dapcn_after_fusion_cityscapes.py"
    # SegMenter
    "configs/segmenter/segmenter_dapcn_before_fusion_cityscapes.py"
    "configs/segmenter/segmenter_dapcn_after_fusion_cityscapes.py"
    # UNetFormer
    "configs/unetformer/unetformer_dapcn_before_fusion_cityscapes.py"
    "configs/unetformer/unetformer_dapcn_after_fusion_cityscapes.py"
)

echo "============================================================"
echo "  Supervised Segmentation with Dynamic Anchor Module"
echo "  Total experiments: ${#CONFIGS[@]}"
echo "  GPUs: ${GPUS}"
echo "============================================================"

cd "$SCRIPT_DIR"

for cfg in "${CONFIGS[@]}"; do
    exp_name=$(basename "$cfg" .py)
    echo ""
    echo "------------------------------------------------------------"
    echo "  Starting: ${exp_name}"
    echo "  Config:   ${cfg}"
    echo "  Work dir: work_dirs/${exp_name}"
    echo "------------------------------------------------------------"

    if [ "$GPUS" -gt 1 ]; then
        bash tools/dist_train.sh "$cfg" "$GPUS" \
            --work-dir "work_dirs/${exp_name}"
    else
        python tools/train.py "$cfg" \
            --work-dir "work_dirs/${exp_name}"
    fi

    echo "  Finished: ${exp_name}"
done

echo ""
echo "============================================================"
echo "  All experiments completed!"
echo "============================================================"
