#!/bin/bash
# Usage: ./submit_divide_job.sh divide_tracts.py [--suffix _multiband]

if [ $# -lt 1 ]; then
    echo "Usage: $0 <script_name.py> [extra_args...]"
    exit 1
fi

SCRIPT=$1
shift
EXTRA_ARGS="$@"

# Good nodes: gpfs02 mounted, /work mounted, hydra_pmi_proxy exists
GOOD_NODES=("ansys06-ib" "ansys07-ib" "ansys10-ib" "ansys11-ib" "ansys26-ib" "ansys27-ib" "ansys28-ib" "ansys30-ib" "ansys31-ib" "ansys42-ib" "ansys53-ib" "ansys55-ib" "ansys59-ib" "ansys60-ib")

FIELDS=("spring1" "spring2" "spring3" "autumn1" "autumn2" "hectomap")

for i in "${!FIELDS[@]}"; do
    field="${FIELDS[$i]}"
    node="${GOOD_NODES[$i]}"
    Nodes="nodes=${node}:ppn=1"
    echo "Submitting job for field: $field on $node"
    datetime=$(date +%d%H%M%S)
    JobName="xdiv-$datetime"
    qsub -V \
      -l "$Nodes" -d "$PWD" -q mini \
      -N "$JobName" \
      -o "$PWD/xdiv-out-$datetime" \
      -e "$PWD/xdiv-error-$datetime" \
      -l walltime=900:00:00 <<EOF
#!/usr/bin/env bash
source /gpfs02/work/xiangchong.li/ana/setupIm.sh
cd "$PWD"
python $SCRIPT --field "$field" $EXTRA_ARGS
EOF
    sleep 2.5
done
