#!/bin/bash

shape=grains
path="`dirname \"$0\"`"
hpc="no"

dir_mesh=/home/davood/projects/periDEM-dev-private/grain-data

#--------------------------------------------------
tstep=5e-4
Gscale=1.0
grav=1e4
base="examples_output/test"

# Get timestamp
timestamp=$(date +"%Y%m%d-%H%M")

# Find the next run number
count=$(ls -1 "$base" 2>/dev/null | grep -E '^run[0-9]+' | sed 's/^run\([0-9]\+\).*/\1/' | sort -n | tail -1)
count=$((count+1))

# Build run name with counter + N + rate + timestamp
#run_name="run${count}-msh${mf}-stptim${tstep}-Gscale${Gscale}-Grav-${grav}-container_z-${container}-${timestamp}"
run_name="run${count}"
dir="$base/$run_name"

#-------------------------------------------------


#-----------------copyimg some config files in dir------
config=$dir/main.conf
sfile=$dir/setup.h5
logfile=$dir/output.log
cp $path/base.conf $config
cp $path/setup.py $dir/



#------------------------Generate setup----------------------------
echo "we are here"
python3 -u $path/setup.py --shape=$shape --setup_file $dir/setup.h5  --msh_path $dir_mesh
cp setup.png $dir/
#----------------------MPIRUN-----------------------------



#mpirun -n 10 --use-hwthread-cpus bin/simulate3d -c $path/base.conf -i $dir/setup.h5 -o $dir




#---------------Genrate plots----------------

#python3 plot3d_timestep.py --all_dir $dir --dotsize 30 --lc=50
