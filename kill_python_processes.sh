#!/bin/bash

# Name of the Python process to search for (you can replace this with your specific script name)
PROCESS_NAME="main.py"

# List all Python processes with the given name
echo "Listing all Python processes with the name: $PROCESS_NAME"
ps aux | grep "$PROCESS_NAME" | grep -v grep

# Killing all Python processes with the given name
echo "Killing all Python processes with the name: $PROCESS_NAME"
pkill -f "$PROCESS_NAME"

# Confirmation message
echo "All processes with the name $PROCESS_NAME have been killed."