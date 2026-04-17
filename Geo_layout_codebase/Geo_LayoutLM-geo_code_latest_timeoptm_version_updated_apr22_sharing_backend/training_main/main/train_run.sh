#!/bin/bash
# Extract ROOT_PATH from constants.py
ROOT_PATH=$(python3 -c "import constants; print(constants.root_path)")
# OUT_FILE_NAME=$(python3 -c "import constants; print(getattr(constants, 'out_file_name', 'train_output.log'))")


# Check if ROOT_PATH is set
if [ -z "$ROOT_PATH" ]; then
    echo "Error: ROOT_PATH is not set in constants.py"
    exit 1
fi

#start process

# python3 funsd_gv.py --root_path $ROOT_PATH
# echo "Data preprocessing is Done!! stage1"

# python3 preprocess/custom/prepare_data_final.py --root_path $ROOT_PATH
# echo "Data preprocessing is Done!! stage2"

# python3 preprocess/custom/preprocess_for_training.py --root_path $ROOT_PATH
# echo "Data preprocessing is Done!! stage3"


OUT_FILE_NAME=$(python3 -c "import constants; print(getattr(constants, 'out_file_name', 'train_output.log'))")

if [[ -z "$OUT_FILE_NAME" ]]; then
    echo "Warning: OUT_FILE_NAME is empty. Defaulting to nohup.out"
    OUT_FILE_NAME="nohup.out"
fi

echo "Training logs will be saved to: $OUT_FILE_NAME"
nohup python3 train.py > "$OUT_FILE_NAME" 2>&1 &
