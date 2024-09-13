#!/bin/bash

# SSH tunnel parameters
SSH_USER="brg2890"
SSH_HOST="nsf-gpu.main.ad.rit.edu"
LOCAL_PORT=9998
REMOTE_PORT=9998

# Conda environment name
CONDA_ENV="DeepEARL"

# Python script to execute
PYTHON_SCRIPT="./AgentClient.py"

# Create SSH tunnel
ssh -f -N -L $LOCAL_PORT:localhost:$REMOTE_PORT $SSH_USER@$SSH_HOST

# Check if the tunnel was created successfully
if [ $? -eq 0 ]; then
    echo "SSH tunnel created successfully"
    
    # Initialize Conda
    eval "$(conda shell.bash hook)"
    
    # Activate Conda environment
    conda activate $CONDA_ENV
    
    # Check if Conda environment activation was successful
    if [ $? -eq 0 ]; then
        echo "Conda environment '$CONDA_ENV' activated successfully"
        
        # Execute Python script
        python $PYTHON_SCRIPT
        
        # Deactivate Conda environment
        conda deactivate
    else
        echo "Failed to activate Conda environment '$CONDA_ENV'"
    fi
    
    # Close the SSH tunnel after the Python script finishes
    tunnel_pid=$(pgrep -f "ssh.*$LOCAL_PORT:localhost:$REMOTE_PORT")
    if [ ! -z "$tunnel_pid" ]; then
        kill $tunnel_pid
        echo "SSH tunnel closed"
    fi
else
    echo "Failed to create SSH tunnel"
fi
