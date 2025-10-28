#!/bin/bash
path="`dirname \"$0\"`"

hpc="no"
#hpc="yes"
non_hpc_cores=4

frac="nofrac"

plot="--noplot --novis"

# input from command line arg
arglist=""
# while getopts af:ph:r:L:x:h:y:t:R:G:K:n:v: flag
while getopts a:c:f:G:h:K:L:pr:R:t:m:n:v::x:y: flag
do
    case "${flag}" in
        a)
	    acc="--acc_val ${OPTARG}"
	    arglist="${arglist}_${flag}-${OPTARG}";;
        c)
	    contact_rad_factor="--contact_rad_factor ${OPTARG}"
	    arglist="${arglist}_${flag}-${OPTARG}";;
	f)
	    frac="frac";;
        G)
	    G_scale="--G_scale ${OPTARG}"
	    arglist="${arglist}_${flag}-${OPTARG}";;
        h)
	    mshfac="--meshsize_factor ${OPTARG}"
	    arglist="${arglist}_${flag}-${OPTARG}";;
        K)
	    K_scale="--K_scale ${OPTARG}"
	    arglist="${arglist}_${flag}-${OPTARG}";;
        L)
	    L="--L ${OPTARG}"
	    arglist="${arglist}_${flag}-${OPTARG}";;
        m)
	    delta_factor="--delta_factor ${OPTARG}"
	    arglist="${arglist}_${flag}-${OPTARG}";;
        n)
	    non_hpc_cores=${OPTARG};;
        p)
	    plot="";;
        r)
	    particle_rad="--particle_rad ${OPTARG}"
	    arglist="${arglist}_${flag}-${OPTARG}";;
        R)
	    rho_scale="--rho_scale ${OPTARG}"
	    arglist="${arglist}_${flag}-${OPTARG}";;
        t)
	    wallh_ratio="--wallh_ratio ${OPTARG}"
	    arglist="${arglist}_${flag}-${OPTARG}";;
        v)
	    vel="--vel_val ${OPTARG}"
	    arglist="${arglist}_${flag}-${OPTARG}";;
        x)
	    nx="--nx ${OPTARG}"
	    arglist="${arglist}_${flag}-${OPTARG}";;
        y)
	    ny="--ny ${OPTARG}"
	    arglist="${arglist}_${flag}-${OPTARG}";;
    esac
done


#resume="yes"
resume="no"

gen_setup="yes"
#gen_setup="no"






#--------------------------------------------------
tstep=5e-4
Gscale=1.0
grav=1e4
base="examples_output/new_setup_ottawa"

# Get timestamp
timestamp=$(date +"%Y%m%d-%H%M")

# Find the next run number
count=$(ls -1 "$base" 2>/dev/null | grep -E '^run[0-9]+' | sed 's/^run\([0-9]\+\).*/\1/' | sort -n | tail -1)
count=$((count+1))

# Build run name with counter + N + rate + timestamp
#run_name="run${count}-msh${mf}-stptim${tstep}-Gscale${Gscale}-Grav-${grav}-container_z-${container}-${timestamp}"
run_name="run${count}-meshsize-10-shape${shape}-tstep-${tstep}"
dir="$base/$run_name"

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

    if [ "$resume" = "yes" ]
    then
	## get the last index
	last=$(ls $dir/tc_*.h5 | tail -1) # Get the largest indices
	last=${last##*/} # strip the path
	last="${last%.*}" # strip the extension
	last=$(echo $last | awk -F '_' '{ print $2 }')

	echo "do_resume = 1" >> $config
	echo "resume_ind = $last" >> $config
	echo "wall_resume = 1" >> $config
    else
	echo ''
    fi
}


gen_setup(){
    echo "Generating experiment setup"
    echo $path
   echo $dir
    python3 $path/setup.py --positions_in_mm --pin_to bottom --novis --path "$dir"    >> $logfile
    cp setup.png $dir/
}


# run code
run()
{
    echo 'running'

    if [ "$hpc" = "yes" ]
    then
	echo 'On hpc'
	export NPROCS=`wc -l $PBS_NODEFILE |gawk '//{print $1}'`
	echo "NPROCS="
	echo $NPROCS
	mpirun -machinefile $PBS_NODEFILE -np $NPROCS bin/simulate3d -c $config -o $dir -i $sfile  >> $logfile
    else
	echo "On non-hpc with ${non_hpc_cores} cores"
	mpirun -n ${non_hpc_cores} bin/simulate3d -c $config -o $dir -i $sfile  >> $logfile
    fi
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
    python3 plot3d_timestep.py --all_dir $dir --dotsize $dotsize --alpha $alpha
    echo "$dir"
    sxiv $dir/*.png &
}

extract(){
    echo 'extracting data'
    python3 $path/extract.py --data_dir $dir

    # two directories to keep quantities.h5 data for each particle
    newdir=$dir/P_A
    mkdir $newdir
    python3 $path/extract.py --data_dir $dir --wheel_ind 0 --output_file $newdir/quantities.h5

    newdir=$dir/P_B
    mkdir $newdir
    python3 $path/extract.py --data_dir $dir --wheel_ind 1 --output_file $newdir/quantities.h5
}

multiplot(){
    echo 'plotting data'
    # quantity=vel_y
    quantity=vel_z

    for quantity in vel_x vel_y vel_z
    do
	python3 $path/multiplot.py -p $dir --subdirs P_A P_B --quantity $quantity --img_dir $dir
    done
}

gen_vid(){
    echo 'Generating video'
    ./gen_vid.sh $dir
}

create_env
gen_setup
#run
#gen_plot
# extract
# multiplot
#
# gen_vid
