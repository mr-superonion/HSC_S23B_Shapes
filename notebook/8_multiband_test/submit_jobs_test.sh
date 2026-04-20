#!/bin/bash
# Usage: ./submit_jobs_test.sh [emax]
#   emax: max |e| for selection cut (default 0.3)

EMAX=${1:-0.3}

# Good nodes: gpfs02 mounted, /work mounted, hydra_pmi_proxy exists
Nodes="nodes=ansys06-ib:ppn=20+ansys07-ib:ppn=20+ansys10-ib:ppn=20+ansys11-ib:ppn=20+ansys26-ib:ppn=20+ansys27-ib:ppn=20+ansys28-ib:ppn=20+ansys30-ib:ppn=20+ansys31-ib:ppn=20+ansys42-ib:ppn=20+ansys53-ib:ppn=20+ansys55-ib:ppn=20+ansys59-ib:ppn=20+ansys60-ib:ppn=20"

submit_qsub() {
    local CMD
    CMD=$(printf '%q ' "$@")

    local datetime JobName
    datetime=$(date +%d%H%M%S)
    JobName="xlarge-$datetime"

    qsub -V \
      -l "$Nodes" -d "$PWD" -q large \
      -N "$JobName" \
      -o "$PWD/xlarge-mb-out-$datetime" \
      -e "$PWD/xlarge-mb-error-$datetime" \
      -l pmem=22gb \
      -l walltime=10:00:00 \
      -m abe <<EOF
#!/usr/bin/env bash
source /gpfs02/work/xiangchong.li/ana/setupIm.sh
cd "$PWD"
mpirun -np 280 $CMD
EOF
}

echo "Submitting with emax=$EMAX"

submit_qsub python ./test_1_bin_multiband.py --start 0 --end 1000 --emax "$EMAX"
sleep 2.0
submit_qsub python ./test_2_tancross1_multiband.py --start 0 --end 1000 --emax "$EMAX"
sleep 2.0
submit_qsub python ./test_2_tancross2_gaia_multiband.py --start 0 --end 1000 --emax "$EMAX"
sleep 2.0
submit_qsub python ./test_2_tancross3_gglens_multiband.py --start 0 --end 1000 --emax "$EMAX"
sleep 2.0
submit_qsub python ./test_3_galstar_multiband.py --start 0 --end 1000 --emax "$EMAX"
