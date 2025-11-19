#!/usr/bin/env bash

server="gfarm"

Nodes="nodes=ansys01-ib:ppn=20+ansys02-ib:ppn=20+ansys03-ib:ppn=20+ansys05-ib:ppn=20+ansys06-ib:ppn=20+ansys07-ib:ppn=20+ansys09-ib:ppn=20+ansys10-ib:ppn=20+ansys11-ib:ppn=20+ansys12-ib:ppn=20+ansys13-ib:ppn=20+ansys15-ib:ppn=20+ansys16-ib:ppn=20+ansys17-ib:ppn=20+ansys18-ib:ppn=20+ansys19-ib:ppn=20+ansys20-ib:ppn=20+ansys21-ib:ppn=20+ansys22-ib:ppn=20+ansys23-ib:ppn=20"

# Timestamped job name
datetime=$(date +%d%H%M%S)
JobName="xlarge-$datetime"

# Command you want to run inside the job
CMD="$@"

# Submit to PBS/Torque
qsub -V \
  -l "$Nodes" -d "$PWD" -q large \
  -N "$JobName" \
  -o "$PWD/xlarge-$server-out-$datetime" \
  -e "$PWD/xlarge-$server-error-$datetime" \
  -l walltime=10:00:00 \
  -m abe <<EOF
#!/usr/bin/env bash
cd "$PWD"
$CMD
EOF
