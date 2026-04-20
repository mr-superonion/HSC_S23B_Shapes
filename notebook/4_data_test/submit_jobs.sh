#!/bin/bash

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
      -o "$PWD/xlarge-out-$datetime" \
      -e "$PWD/xlarge-error-$datetime" \
      -l pmem=22gb \
      -l walltime=10:00:00 \
      -m abe <<EOF
#!/usr/bin/env bash
source /gpfs02/work/xiangchong.li/ana/setupIm.sh
cd "$PWD"
mpirun -np 280 $CMD
EOF
}

submit_qsub python ./test_1_bin.py --start 0 --end 1000
sleep 2.0
submit_qsub python ./test_2_tancross1.py --start 0 --end 1000
sleep 2.0
submit_qsub python ./test_2_tancross2_gaia.py --start 0 --end 1000
sleep 2.0
submit_qsub python ./test_2_tancross3_gglens.py --start 0 --end 1000
sleep 2.0
submit_qsub python ./test_3_galstar.py --start 0 --end 1000
