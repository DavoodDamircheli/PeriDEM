#!/bin/bash
path="`dirname \"$0\"`"

non_hpc_cores=4

frac="nofrac"

plot="--noplot --novis"

#--------------------------------------------------
tstep=5e-4
Gscale=1.0
grav=1e4
#base="examples_output"
base="examples_output/sub_cylinder_ottawa"

# Get timestamp
timestamp=$(date +"%Y%m%d-%H%M")

# Find the next run number
count=$(ls -1 "$base" 2>/dev/null | grep -E '^run[0-9]+' | sed 's/^run\([0-9]\+\).*/\1/' | sort -n | tail -1)
count=$((count+1))

run_name="run${count}-qs"
dir="$base/sphere_ottawa_S0_e-9_24"
#-------------------------------------------------
# create subdirectory

config=$dir/main.conf
sfile=$dir/setup.h5
logfile=$dir/output.log
echo "logfile: $logfile"

create_env(){
    mkdir -p $dir
    #clear logfile
    echo '' > $logfile
    echo $(cd $path; git show -s) > $logfile
    echo "arglist: $arglist" >> $logfile

    cp $path/base.conf $config
    cp $path/setup.py  $dir/

}


gen_setup(){
    echo "Generating experiment setup"
    echo $path
   echo $dir
    #python3 $path/setup.py  --pin_to bottom --novis --path "$dir"    >> $logfile
    python3 $path/setup.py   --setup_file $dir/setup.h5 >> $logfile

    cp setup.png $dir/
}


# run code
run()
{
    echo 'running MPI'

	NP=$(nproc)
        #echo "Running with $NP cores..."
        #mpirun -n $(nproc)   --oversubscribe bin/simulate3d -c $config -o $dir -i $sfile  >> $logfile
        mpirun -n 1   bin/simulate3d -c $config -o $dir -i $sfile  >> $logfile

}

gen_plot(){
    echo 'Generating plots'

    # dotsize=0.5
    dotsize=3
    alpha=0.5
    bond_linewidth=0.1
    fcval=24

    # qq=force
    qq=damage
    # cmap='Greys_r'
    cmap='Greys'
    # cmap='viridis'
    # python3 plot3d_timestep.py --data_dir $dir --img_dir $dir --setup_file $sfile --dotsize $dotsize --quantity $qq --nocolorbar --colormap $cmap #--plot_bonds --bond_linewidth ${bond_linewidth} --fc $fcval
   #python3 plot3d_timestep.py --all_dir $dir --dotsize 39  --setup_file $sfile  --fc 17 --lc 18

#---------------------------------------

#-----------------------------------------





python3 plot3d_timestep.py --all_dir $dir --dotsize 25
    echo "$dir"
    sxiv $dir/*.png &
}

gen_vid(){
    echo 'Generating video'
    ./gen_vid.sh $dir
}

create_env
gen_setup
run
gen_plot
# # # # # # # # # # # # # # # gen_vid
# # # extract
# multiplot
#
#gen_vid
