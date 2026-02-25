#!/bin/bash

# Generate timestamp
timestamp=$(date +%Y%m%d_%H%M%S)

# debug flag
DEBUG_FLAG=""

for arg in "$@"; do
    if [[ "$arg" == "--debug" ]]; then
        DEBUG_FLAG="--debug"
        timestamp="${timestamp}_debug"
    fi
done

# Create a directory named after the timestamp
dir_name="outputs/$timestamp"
mkdir -p "$dir_name"

# Run the Python script, passing the timestamp as an argument
# Redirect output to a log file inside the timestamped directory
PYTHONPATH=$(pwd) python ./train.py "$dir_name" "$timestamp" $DEBUG_FLAG > "$dir_name/training_inference_output_$timestamp.log" 2>&1