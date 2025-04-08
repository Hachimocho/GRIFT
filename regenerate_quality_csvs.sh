#!/bin/bash

# Script to regenerate all quality attribute CSV files for the deepfake project
# This will process one file at a time to avoid memory issues

# Set the data root directory
DATA_ROOT="/home/brg2890/major/datasets/ai-face"

# Define the array of base CSV files and their corresponding quality outputs
declare -A FILE_MAPPINGS=(
    ["train_part2.csv"]="train_quality_part2.csv"
)
    # ["val.csv"]="val_quality.csv"
    # ["test.csv"]="test_quality.csv"
# Create a logs directory for this run
LOG_DIR="/home/brg2890/major/bryce_python_workspace/GraphWork/HyperGraph/logs/regenerate_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "Logs will be saved to: $LOG_DIR"

# Create a backup directory for original files
BACKUP_DIR="${LOG_DIR}/originals"
mkdir -p "$BACKUP_DIR"
echo "Creating backups of original files in: $BACKUP_DIR"

# Function to process a single CSV file
process_file() {
    local base_csv="$1"
    local quality_csv="$2"
    
    echo "==============================================="
    echo "Processing: $base_csv → $quality_csv"
    echo "==============================================="
    
    # Create a backup of the original quality CSV if it exists
    if [ -f "${DATA_ROOT}/${quality_csv}" ]; then
        echo "Backing up original ${quality_csv}..."
        cp "${DATA_ROOT}/${quality_csv}" "${BACKUP_DIR}/${quality_csv}"
    fi
    
    # Run the attribute generation script with optimizations
    echo "Running attribute generation..."
    python /home/brg2890/major/bryce_python_workspace/GraphWork/HyperGraph/utils/additional_attributes.py \
        --data_root "$DATA_ROOT" \
        --metadata_path "${base_csv}" \
        --output_path "${DATA_ROOT}/${quality_csv}" \
        --batch_size 64 \
        --disable_deepface \
        2>&1 | tee "${LOG_DIR}/${base_csv%.*}_log.txt"
    
    # Check if generation was successful
    if [ -f "${DATA_ROOT}/${quality_csv}" ]; then
        echo "✅ Successfully generated ${quality_csv}"
        
        # Compare the number of rows in the original vs. new file
        if [ -f "${BACKUP_DIR}/${quality_csv}" ]; then
            orig_count=$(wc -l < "${BACKUP_DIR}/${quality_csv}")
            new_count=$(wc -l < "${DATA_ROOT}/${quality_csv}")
            echo "Original file: $orig_count rows"
            echo "New file: $new_count rows"
        fi
    else
        echo "❌ Failed to generate ${quality_csv}"
    fi
    
    echo ""
}

# Process each file in the mapping
for base_csv in "${!FILE_MAPPINGS[@]}"; do
    quality_csv="${FILE_MAPPINGS[$base_csv]}"
    process_file "$base_csv" "$quality_csv"
done

echo "All quality CSV files have been regenerated."
echo "Log files and debug visualizations are available in: $LOG_DIR"
echo "Original files were backed up to: $BACKUP_DIR"
