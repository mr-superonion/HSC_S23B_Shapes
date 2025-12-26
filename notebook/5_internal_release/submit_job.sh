#!/bin/bash
# Usage: ./submit_gal_catalog.sh make_db_gal_catalog.py

if [ $# -lt 1 ]; then
    echo "Usage: $0 <script_name.py>"
    exit 1
fi

SCRIPT=$1

FIELDS=("spring1" "spring2" "spring3")

for field in "${FIELDS[@]}"; do
    echo "Submitting job for field: $field"
    xsubTiny python $SCRIPT --field "$field"
    sleep 10
done
