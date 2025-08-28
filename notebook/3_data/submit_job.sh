#!/bin/bash


SCRIPT=$1

if [ -z "$SCRIPT" ]; then
    echo "Usage: $0 script.py"
    exit 1
fi

# Define ranges
RANGES=(
    "0 5000"
    "5000 10000"
    "10000 15000"
    "15000 20000"
    "20000 25000"
    "25000 30000"
    "30000 35000"
    "35000 40000"
)

# Loop over ranges
for range in "${RANGES[@]}"; do
    read START END <<< "$range"
    echo "Submitting $SCRIPT from $START to $END"
    xsubLarge mpirun -np 400 python "$SCRIPT" --start "$START" --end "$END"
    sleep 1.5
done

