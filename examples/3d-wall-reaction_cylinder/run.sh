#!/bin/bash
hpc=no

# n=56
n=2

# shape=plus
shape=plus_coarse
# shape=sphere

#######################################################################
# path

this_path=$(dirname "$0")
this_dir=$(basename "$this_path")

if [ "$hpc" = "yes" ]
then
    path="/work/$USER/3d-peri-wheel-output/nofrac_s-${shape}_small_3d-2x2"
else
    path="$HOME/peri-wheel/examples_output/$this_dir/nofrac_s-${shape}_small_3d-2x2"
fi

mkdir $path -p

config="$this_path/base.conf"

# #######################################################################
# #generate setup
#
gmsh meshdata/3d/3d_${shape}_small.geo -3
#
python3 $this_path/setup.py --shape=${shape}_small_3d --setup_file=$path/setup.h5 --plot
# python3 $this_path/setup.py --shape=${shape}_small_3d --setup_file=$path/setup.h5

#######################################################################
# run

if [ "$hpc" = "yes" ]
then
    export NPROCS=`wc -l $PBS_NODEFILE |gawk '//{print $1}'`
    echo "NPROCS=$NPROCS"
    mpirun -machinefile $PBS_NODEFILE -np $NPROCS bin/simulate3d -c $config  -i $path/setup.h5 -o $path
else
    # bin/simulate3d -c $config -i $path/setup.h5 -o $path
    mpirun -n $n bin/simulate3d -c $config -i $path/setup.h5 -o $path
fi

# ######################################################################
# #Plot
#
# lc=41

# Get the largest indices
last=$(ls $path/tc_*.h5 | tail -1)
last=${last##*/} # strip the path
last="${last%.*}" # strip the extension
lc=$(echo $last | awk -F '_' '{ print $2 }')

echo 'lc:'
echo $lc
# #
# #
# python3 plot3d_timestep.py --all_dir=$path --dotsize 20 --lc=19
# #
#
python3 plot3d_timestep.py --all_dir=$path --dotsize 10 --lc=$lc --motion --motion_az_stepsize=-0 --motion_angle_stepsize=0 --alpha=0.8 --view_az=0 --view_angle=0

# python3 $this_path/get_timestep_wall_reaction.py --path=$path --lc=$lc

# # ind='5,10,15,20,25,30,35,40,45,50,55'
# # ind='0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120'
# ind='2,7,12,17,22,27,32,37,42,47,52,57,62,67,72,77,82,87,92,97,102,107,112,117,122'
#
# # ind=0
# # for ind in 1 3 5 7 9 11 13 17 19 21 23 25
# # do
#     python3 plot3d_timestep-selective.py --all_dir=$path --dotsize 30 --lc=41 --motion --motion_az_stepsize=-0.5 --motion_angle_stepsize=2 --alpha=1 --ind=$ind --img_prefix=P_bottom_ --camera_rad=5e-3 --nocolorbar  #--serial
#    # python3 plot3d_timestep-selective.py --all_dir=$path --dotsize 30 --lc=41 --motion --motion_az_stepsize=-0.5 --motion_angle_stepsize=2 --alpha=1 --ind=$ind --img_prefix=P_$ind --nowall --camera_rad=5e-3 --nocolorbar  #--serial
    # --colormap=Greys
# done


# open3d
# python3 plot3d_timestep_o3d.py --path=$path --dotsize 30 --fc=1 --lc=80 --plot_surface --serial --no_scatter --camera_json=examples/3d-bulk-compress/camera-plus-4-compress.json
# python3 plot3d_timestep_o3d.py --path=$path --dotsize 30 --fc=1 --lc=80 --plot_surface --serial --no_scatter
# python3 plot3d_timestep_o3d.py --path=$path --dotsize 30 --fc=61 --lc=61 --plot_surface --serial --no_scatter --vis
