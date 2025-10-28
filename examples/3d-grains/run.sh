#!/bin/bash

shape=grains
path="`dirname \"$0\"`"
hpc="no"

N=1
rate=2e-8
timestamp=$(date +"%Y%m%d-%H%M")
run_name="comp-N${N}-steptime${rate}-${timestamp}"
dir="examples_output/3d-grains/test/$run_name"
mkdir -p "$dir"


echo "$dir"
dir_mesh=/home/davood/projects/periDEM-dev-private/grain-data


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
