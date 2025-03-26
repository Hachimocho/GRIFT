#!/bin/bash
# Script to regenerate quality CSVs in parallel by using split CSV files
# Usage: ./parallel_regenerate_quality_csvs.sh [train|val|test] [num_parts]

# Configuration
DATA_ROOT="/home/brg2890/major/datasets/ai-face"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="/home/brg2890/major/bryce_python_workspace/GraphWork/HyperGraph/logs/parallel_regenerate_${TIMESTAMP}"
SPLIT_DIR="${DATA_ROOT}/split_csvs_${TIMESTAMP}"
BACKUP_DIR="${LOG_DIR}/backups"
PYTHON_SCRIPT="/home/brg2890/major/bryce_python_workspace/GraphWork/HyperGraph/utils/additional_attributes.py"
SPLIT_SCRIPT="/home/brg2890/major/bryce_python_workspace/GraphWork/HyperGraph/split_csv.py"

# Default values
DATASET=${1:-"train"}
NUM_PARTS=${2:-2}

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$BACKUP_DIR"
mkdir -p "$SPLIT_DIR"

echo "==================================================="
echo "Parallel Quality CSV Regeneration"
echo "Dataset: $DATASET"
echo "Parts: $NUM_PARTS"
echo "Timestamp: $TIMESTAMP"
echo "Log Directory: $LOG_DIR"
echo "Split CSV Directory: $SPLIT_DIR"
echo "==================================================="

# Source CSV and target quality CSV
BASE_CSV="${DATASET}.csv"
QUALITY_CSV="${DATASET}_quality.csv"

# Create backup of original quality CSV if it exists
if [ -f "${DATA_ROOT}/${QUALITY_CSV}" ]; then
    echo "Backing up original ${QUALITY_CSV}..."
    cp "${DATA_ROOT}/${QUALITY_CSV}" "${BACKUP_DIR}/${QUALITY_CSV}"
fi

# Split the CSV file
echo "Splitting ${BASE_CSV} into ${NUM_PARTS} parts..."
python "$SPLIT_SCRIPT" --input "${DATA_ROOT}/${BASE_CSV}" --output-dir "$SPLIT_DIR" --num-parts "$NUM_PARTS"

# Process each part
for part_file in "$SPLIT_DIR"/${DATASET}_part*.csv; do
    part_name=$(basename "$part_file" .csv)
    part_quality_csv="${part_name}_quality.csv"
    
    echo "==============================================="
    echo "Processing: $part_file → $part_quality_csv"
    echo "==============================================="
    
    # Run the attribute generation script with optimizations
    echo "Running attribute generation..."
    python "$PYTHON_SCRIPT" \
        --data_root "$DATA_ROOT" \
        --metadata_path "$part_file" \
        --output_path "${SPLIT_DIR}/${part_quality_csv}" \
        --batch_size 64 \
        --disable_deepface \
        2>&1 | tee "${LOG_DIR}/${part_name}_log.txt"
    
    # Check if generation was successful
    if [ -f "${SPLIT_DIR}/${part_quality_csv}" ]; then
        echo "✅ Successfully generated ${part_quality_csv}"
    else
        echo "❌ Failed to generate ${part_quality_csv}"
    fi
done

echo "==================================================="
echo "All parts processed. Results are in ${SPLIT_DIR}"
echo "==================================================="
echo ""
echo "To merge the results back together:"
echo "python -c \"import pandas as pd; import glob; pd.concat([pd.read_csv(f) for f in glob.glob('${SPLIT_DIR}/${DATASET}_part*_quality.csv')]).to_csv('${DATA_ROOT}/${QUALITY_CSV}', index=False)\""
echo ""
echo "==================================================="
