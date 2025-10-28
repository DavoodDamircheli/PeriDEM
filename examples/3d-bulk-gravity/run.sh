#!/bin/bash

path="`dirname \"$0\"`"
hpc="no"
# 'hollow_sphere', 'plus3d', cube, sphere
shape='sphere'
if [ "$hpc" = "yes" ]
then
	output_path="/work/$USER/XXXX/$dirname"
else
	output_path="examples_output/3d-gravity"
fi

echo "dir is the output path for results"
dir="$output_path/$shape/N100"


#dir=/home/davood/projects/containers/cont-examples/examples_output/3d-gravity/cube/N1
echo "$dir"
#dir_mesh=/home/davood/projects/containers/cont-examples/examples_output/meshes



#-----------------copyimg some config files in dir------
config=$dir/main.conf
sfile=$dir/setup.h5
logfile=$dir/output.log
cp $path/base.conf $config
cp $path/setup.py $dir/
mkdir -p  $dir



#------------------------Generate setup----------------------------
echo "we are here"
#python3 -u $path/setup.py --shape=$shape --setup_file $dir/setup.h5  --msh_path $dir_mesh  >>$logfile 
python3 -u $path/setup.py --shape=$shape --setup_file $dir/setup.h5     

#----------------------MPIRUN-----------------------------



#mpirun -n 10 bin/simulate3d -c $path/base.conf -i $dir/setup.h5 -o $dir




#---------------Genrate plots----------------

#python3 plot3d_timestep.py --all_dir $dir --dotsize 30 --lc=50
#python3 plot3d_timestep.py --all_dir $dir --dotsize 90 --lc=50







