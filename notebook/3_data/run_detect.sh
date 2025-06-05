#!/bin/bash

# Get short hostname (strip .local if present)
HOSTNAME=$(hostname | cut -d. -f1)

# Extract numeric part, e.g., 001 from hpcs001
NUM=$(echo "$HOSTNAME" | sed -E 's/^hpcs([0-9]+)$/\1/')

# Validate numeric part
if [[ -z "$NUM" ]]; then
    echo "Invalid hostname: $HOSTNAME"
    exit 1
fi

# Convert NUM to integer (strip leading zeros)
INDEX=$((10#$NUM))  # 10# forces base 10 to avoid octal

# Compute start and end
START=$(((INDEX - 1) * 1650))
END=$((START + 1650))

echo "Running on $HOSTNAME with START=$START and END=$END"
mpirun -np 10 ./make_catalog_detect.py --start "$START" --end "$END"
