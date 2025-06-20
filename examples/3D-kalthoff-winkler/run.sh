#!/bin/bash
path="`dirname \"$0\"`"
#hpc="no"
hpc="yes"

export PATH=/opt/mpich/bin:$PATH

non_hpc_cores=4

# input from command line arg
#shape=$1
#theta=$2
#P_meshsize=$3
frac="nofrac"
#delta_frac=$1
#echo "delta_frac = $delta_frac"
:<<'COMMENT'

COMMENT
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
    plot="--plot";;
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

gen_setup="no"
#gen_setup="no"

dirname="kalthoff3d"

if [ "$hpc" = "yes" ]
then
data_output_loc="/work/$USER/periHpc-output/$dirname"
else
data_output_loc="examples_output/$dirname"
fi

#plot=""
plot="--plot"


echo "----------------------------------------------------"
# paths
dir="$data_output_loc/kalthof3d_hpc_h10"
#dir=$data_output_loc/${frac}_${arglist}
config=$dir/main.conf
sfile=$dir/setup.h5
logfile=$dir/output.log
##
mkdir -p $dir
echo "logfile: $logfile"
#clear logfile
echo '' > $logfile

# copy config file
cp $path/base.conf $config
echo "the base.cofig copied in the main.cong"
#cp $path/setup.py $dir/
cp $path/setup.py $dir/

echo "----------------------------------------------------"

#!/bin/bash


echo "----------------------------------------------------"
gen_setup()
{
    echo " gen_setup: Generating experiment setup"
    echo "python3 $path/setup.py --setup_file $sfile $hopper_ratio $particle_rad $nx $ny $L $wallh_ratio $vel $acc $rho_scale $K_scale $G_scale $mshfac $delta_factor $contact_rad_factor $plot" >> $logfile

    python3 $path/setup.py --setup_file $sfile 
    cp setup.png $dir/

}

echo "----------------------------------------------------"
# run code
#$PBS_NODEFILE>pbsNotefile.dat
echo 'this is PBS_NODEFILE'
echo $PBS_NODEFILE
run()
{

    echo 'cpp is running'
    if [ "$hpc" = "yes" ]
    then
        echo 'On hpc'
	echo 'this is PBS_NODEFILE'
	echo $PBS_NODEFILE
	#export NPROCS=`wc -l $PBS_NODEFILE | gawk '//{print $1}'`
        #export NPROCS=`wc -l $PBS_NODEFILE | gawk '{print $1}'`
        #NPROCS=`cat $PBS_NODEFILE | wc -l`
        NPROCS=$(cat $PBS_NODEFILE 2>&1 | wc -l)
        echo "NPROCS=" 
        echo $NPROCS
        #mpirun -machinefile $PBS_NODEFILE -np $NPROCS bin/simulate3d -c $config -o $dir -i $sfile  >> $logfile
        #mpirun -machinefile $PBS_NODEFILE --np 4 bin/simulate3d -c $config -o $dir -i $sfile  >> $logfile
        mpirun -machinefile $PBS_NODEFILE -np $NPROCS hostname
    else
        echo "On non-hpc with ${non_hpc_cores} cores"
        mpirun -n ${non_hpc_cores} bin/simulate3d -c $config -o $dir -i $sfile  >> $logfile
    fi
}


echo "----------------------------------------------------"
gen_plot(){

    echo 'gen_plot:Generating plots'
    dotsize=0.5
    bond_linewidth=0.1
    fcval=24

    # qq=force
    qq=damage
    # cmap='Greys_r'
    #cmap='Greys'
    cmap='viridis'
    python3 plot3d_timestep.py --data_dir $dir --img_dir $dir --setup_file $sfile --dotsize 9 --view_az 5 --view_angle 90  
    #python3 plot3d_timestep.py --all_dir $dir --img_dir $dir --setup_file $sfile --dotsize 5 
    #python3 plot3d_timestep.py --all_dir $dir  
    echo "$dir"
    sxiv $dir/*.png &
}

echo "----------------------------------------------------"

extract(){
    echo 'extracting data'
    python3  $path/extract.py --data_dir $dir

    # two directories to keep quantities.h5 data for each particle
    newdir=$dir/P_A
    mkdir $newdir
    python3 $path/extract.py --data_dir $dir --wheel_ind 0 --output_file $newdir/quantities.h5

    newdir=$dir/P_B
    mkdir $newdir
    python3 $path/extract.py --data_dir $dir --wheel_ind 1 --output_file $newdir/quantities.h5
}


echo "----------------------------------------------------"
multiplot(){
    # quantity=vel_y

    echo 'multiPlot:  plotting data'
    quantity=vel_z

    for quantity in vel_x vel_y vel_z
    do
       python3 $path/multiplot.py -p $dir --subdirs P_A P_B --quantity $quantity --img_dir $dir
    done
}

echo "----------------------------------------------------"

gen_vid(){
    echo 'Generating video'
    ./gen_vid.sh $dir
}
#creat_env
gen_setup

run
gen_plot
#multiplot
#gen_vid
:<<'COMMENT'
COMMENT
