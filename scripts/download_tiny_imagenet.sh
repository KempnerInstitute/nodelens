#!/usr/bin/env bash
# Download and prepare Tiny-ImageNet-200 for ImageFolder loading.
#
# The validation set ships as a flat directory with a val_annotations.txt file.
# This script reorganizes val/ into class subfolders so torchvision.ImageFolder works.
#
# Usage:
#   bash scripts/download_tiny_imagenet.sh [DATA_DIR]
#   DATA_DIR defaults to ./data

set -euo pipefail

DATA_DIR="${1:-./data}"
TIN_DIR="${DATA_DIR}/tiny-imagenet-200"
ZIP_PATH="${DATA_DIR}/tiny-imagenet-200.zip"

if [[ -d "${TIN_DIR}/train" && -d "${TIN_DIR}/val" ]]; then
    # Check if val is already reorganized (has class subdirs)
    N_SUBDIRS=$(find "${TIN_DIR}/val" -mindepth 1 -maxdepth 1 -type d | wc -l)
    if [[ "${N_SUBDIRS}" -ge 200 ]]; then
        echo "[ok] Tiny-ImageNet already downloaded and organized at ${TIN_DIR}"
        exit 0
    fi
fi

mkdir -p "${DATA_DIR}"

# Download
if [[ ! -f "${ZIP_PATH}" ]]; then
    echo "[download] Fetching tiny-imagenet-200.zip (~237 MB)..."
    wget -q --show-progress -O "${ZIP_PATH}" "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
fi

# Extract
if [[ ! -d "${TIN_DIR}/train" ]]; then
    echo "[extract] Unzipping..."
    unzip -q "${ZIP_PATH}" -d "${DATA_DIR}"
fi

# Reorganize val/ into class subdirectories
VAL_DIR="${TIN_DIR}/val"
VAL_ANNOT="${VAL_DIR}/val_annotations.txt"

if [[ ! -f "${VAL_ANNOT}" ]]; then
    echo "[error] val_annotations.txt not found at ${VAL_ANNOT}"
    exit 1
fi

echo "[reorg] Reorganizing val/ into class subfolders..."
while IFS=$'\t' read -r fname cls _ _ _ _; do
    cls_dir="${VAL_DIR}/${cls}"
    mkdir -p "${cls_dir}"
    src="${VAL_DIR}/images/${fname}"
    if [[ -f "${src}" ]]; then
        mv "${src}" "${cls_dir}/${fname}"
    fi
done < "${VAL_ANNOT}"

# Clean up flat images dir if empty
rmdir "${VAL_DIR}/images" 2>/dev/null || true

N_CLASSES=$(find "${VAL_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l)
N_TRAIN=$(find "${TIN_DIR}/train" -name "*.JPEG" | wc -l)
N_VAL=$(find "${VAL_DIR}" -name "*.JPEG" | wc -l)

echo "[ok] Tiny-ImageNet ready at ${TIN_DIR}"
echo "     Classes: ${N_CLASSES}"
echo "     Train images: ${N_TRAIN}"
echo "     Val images: ${N_VAL}"
