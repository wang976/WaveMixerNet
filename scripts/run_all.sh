#!/bin/bash

# Set script directory - adjust path relative to the execution directory
SCRIPT_DIR="scripts/long_term_forecast"

# Check whether the script directory exists
if [ ! -d "$SCRIPT_DIR" ]; then
    echo "Error: Script directory $SCRIPT_DIR does not exist"
    exit 1
fi

# List of scripts to run in order
scripts=(
    "etth1.sh"
    "etth2.sh" 
    "ettm1.sh"
    "ettm2.sh"
    "weather.sh"
    "electricity.sh"
    "traffic.sh"
)

echo "Starting parameter tuning scripts..."
echo "Execution order: ETTh1 -> ETTh2 -> ETTm1 -> ETTm2 -> Weather -> Electricity"
echo "=========================================="
echo " "

# Loop through each script and execute
for script in "${scripts[@]}"; do
    script_path="$SCRIPT_DIR/$script"
    
    if [ -f "$script_path" ]; then
        echo "Running: $script"
        echo "Start time: $(date)"
        
    # Run the script and check the return code
        bash "$script_path"
        exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            echo "✓ $script completed"
        else
            echo "✗ $script failed (exit code: $exit_code)"
            echo "Continue with the next script? (y/n)"
            read -r response
            if [[ ! "$response" =~ ^[Yy]$ ]]; then
                echo "User chose to exit"
                exit $exit_code
            fi
        fi
        
        echo "End time: $(date)"
        echo "=========================================="
    else
        echo "Warning: Script file $script_path not found, skipping"
    fi
done

echo "All parameter tuning scripts finished!"
echo "Final end time: $(date)"