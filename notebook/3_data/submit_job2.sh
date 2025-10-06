#!/bin/bash

#!/bin/bash

SCRIPT=$1

if [ -z "$SCRIPT" ]; then
    echo "Usage: $0 script.py"
    exit 1
fi

START=5000
END=38000

STEP=1000   # chunk size

while [ $START -lt $END ]; do
    RANGE_END=$((START + STEP))
    echo "Submitting $SCRIPT from $START to $RANGE_END"
    xsubLarge mpirun -np 400 python "$SCRIPT" --start "$START" --end "$RANGE_END"
    sleep 1.5
    START=$RANGE_END
done
