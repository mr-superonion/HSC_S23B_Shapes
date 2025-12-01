#!/usr/bin/env bash

# ---------------- PBS wrapper config ----------------
server="gfarm"
Nodes="nodes=ansys01-ib:ppn=20+ansys02-ib:ppn=20+ansys03-ib:ppn=20+ansys05-ib:ppn=20+ansys06-ib:ppn=20+ansys07-ib:ppn=20+ansys09-ib:ppn=20+ansys10-ib:ppn=20+ansys11-ib:ppn=20+ansys12-ib:ppn=20+ansys13-ib:ppn=20+ansys15-ib:ppn=20+ansys16-ib:ppn=20+ansys17-ib:ppn=20+ansys18-ib:ppn=20+ansys19-ib:ppn=20+ansys20-ib:ppn=20+ansys25-ib:ppn=20+ansys26-ib:ppn=20+ansys27-ib:ppn=20"

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
mpirun -np 400 $CMD
EOF
}

# ----------------- Main submission logic -----------------

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
    read -r START END <<< "$range"
    echo "Submitting $SCRIPT from $START to $END"
    submit_qsub python "$SCRIPT" --start "$START" --end "$END"
    sleep 1.5
done
