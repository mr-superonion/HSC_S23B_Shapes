#!/usr/bin/env bash

# ---------------- PBS wrapper config ----------------
server="gfarm"
Nodes="nodes=ansys01-ib:ppn=20+ansys06-ib:ppn=20+ansys07-ib:ppn=20+ansys10-ib:ppn=20+ansys11-ib:ppn=20+ansys26-ib:ppn=20+ansys27-ib:ppn=20+ansys28-ib:ppn=20+ansys30-ib:ppn=20+ansys31-ib:ppn=20+ansys42-ib:ppn=20+ansys53-ib:ppn=20+ansys55-ib:ppn=20+ansys59-ib:ppn=20+ansys60-ib:ppn=20"

# ------------- Function: submit one PBS job ---------
submit_qsub() {
    # All args = command line to run inside the job
    # Use printf '%q' to preserve quoting safely
    local CMD
    CMD=$(printf '%q ' "$@")

    local datetime JobName
    datetime=$(date +%d%H%M%S)
    JobName="xlarge-$datetime"

    qsub -V \
      -l "$Nodes" -d "$PWD" -q large \
      -N "$JobName" \
      -o "$PWD/xlarge-$server-out-$datetime" \
      -e "$PWD/xlarge-$server-error-$datetime" \
      -l pmem=22gb \
      -l walltime=10:00:00 \
      -m abe <<EOF
#!/usr/bin/env bash
source /gpfs02/work/xiangchong.li/ana/setupIm.sh
cd "$PWD"
mpirun -np 300 $CMD
EOF
}

# ----------------- Main submission logic -----------------

SCRIPT=$1
NJOBS=${2:-8}   # number of jobs: 1 or 8 (default 8)

if [ -z "$SCRIPT" ]; then
    echo "Usage: $0 script.py [njobs]"
    echo "  njobs: 1 or 8 (default 8)"
    exit 1
fi

# Full range of tracts_fdfc_v2_final.fits
TOTAL=36686

if [ "$NJOBS" = "1" ]; then
    RANGES=("0 $TOTAL")
elif [ "$NJOBS" = "8" ]; then
    RANGES=(
        "0 5000"
        "5000 10000"
        "10000 15000"
        "15000 20000"
        "20000 25000"
        "25000 30000"
        "30000 35000"
        "35000 $TOTAL"
    )
else
    echo "Unsupported njobs=$NJOBS (use 1 or 8)"
    exit 1
fi

# Loop over ranges
for range in "${RANGES[@]}"; do
    read -r START END <<< "$range"
    echo "Submitting $SCRIPT from $START to $END"
    submit_qsub python "$SCRIPT" --start "$START" --end "$END" --band z
    sleep 1.5
done
