#!/bin/bash
shape=hollow_sphere
path="`dirname \"$0\"`"


#----------------out put result path----------

output_path="/work/$USER/periHpc-output/$dirname"
echo "dir is the output path for results"


#dir="/work/davdam/periHpc-output/3d-gravity/hollow_sphere/N600-3"
dir="/work/davdam/periHpc-output/3d-gravity/new_setup_hollow_sphere/N1000"

#-------data directory
dir_mesh=/app/perigrain-oct-2024/grain-data



#-----------------copyimg some config files in dir------
config=$dir/main.conf
sfile=$dir/setup.h5
logfile=$dir/output.log
cp $path/base.conf $config
cp $path/setup.py $dir/
mkdir -p  $dir


#---------------update_files-configuring singularity

PATH_IMG2=/app/perigrain-oct-2024/

PATH_files11=/app/perigrain-oct-2024/exp_dict.py
PATH_files12=/home/davdam/containers/update_files/exp_dict.py

PATH_files21=/app/perigrain-oct-2024/shape_dict.py
PATH_files22=/home/davdam/containers/update_files/shape_dict.py

PATH_files31=/app/perigrain-oct-2024/arrangements.py
PATH_files32=/home/davdam/containers/update_files/arrangements.py



#------------outputs----------------------
PATH_OUT=/work/davdam/periHpc-output/

PATH_MSH1=/app/perigrain-oct-2024/meshdata/
PATH_MSH2=/work/davdam/meshdata/
PATH_MSH_G=/work/davdam/grain-data
PATH_LOC=/home/davdam/perihpcExamples/3d-grain-slurm-tested-oct17/
PATH_IMG=/app/perigrain-oct-2024/examples/hpcExample/myexample/

IMG=/home/davdam/containers/mpi-412-Feb-20-2025.sif
IMG1=/home/davdam/containers/ 

PATH_h5py1=/app/perigrain-oct-2024/data/hdf5
PATH_h5py2=/work/davdam/data/hdf5

PATH_h5py21=/app/perigrain-oct-2024/output/hdf5
PATH_h5py22=/work/davdam/output/hdf5
PATH_DATA1=/app/perigrain-oct-2024/data
PATH_DATA2=/work/davdam/data

PATH1=/app/perigrain-oct-2024/output
PATH2=$dir

PATH_SIM3D=/app/perigrain-oct-2024/bin/simulate3d 

#---------------------------------------------
myPython() {
    singularity exec \
        -B "$PATH_OUT:$PATH_OUT:rw,$PATH_MSH2:$PATH_MSH1,$PATH_MSH_G:$PATH_MSH_G,$PATH_LOC:$PATH_IMG,$PATH_DATA2:$PATH_DATA1:rw,$PATH_h5py2:$PATH_h5py1:rw,$PATH_h5py22:$PATH_h5py21:rw,$PATH2:$PATH1:rw,$PATH_files12:$PATH_files11:rw,$PATH_files22:$PATH_files21:rw,$PATH_files32:$PATH_files31:rw" \
        "$IMG" \
        bash -c "cd $PATH_IMG2; python3  $*"
}


myMPIsingularity() {
      srun -n "$SLURM_NTASKS" --mpi=pmi2 \
        singularity exec  -B "$PATH_OUT:$PATH_OUT:rw,$PATH_MSH2:$PATH_MSH1,$PATH_LOC:$PATH_IMG:rw,$PATH2:$PATH1:rw"  "$IMG" \
        bash -c "cd $PATH_IMG2; $PATH_SIM3D -c $config -o $dir -i $sfile" >> "$logfile" 2>&1
}

echo "----------------Generate setup-------------------------"



myPython $path/setup.py --shape=$shape --setup_file $dir/setup.h5  --msh_path $dir_mesh   >> $logfile 
 echo "Generating setup is done!!"
echo "----------------------MPIRUN-----------------------------"



echo 'On hpc'
echo 'this is SLURM_NODELIST'
echo "$SLURM_NODELIST"
NPROCS=$SLURM_NTASKS
echo "NPROCS="
echo "$NPROCS"

echo "this is dir before MPI"
echo "$dir"

myMPIsingularity
echo "MPI is used!!"
echo "time loop is finished!!!!!!!!!!!!"
echo "$dir"


echo "---------------Genrate plots----------------------"

myPython plot3d_timestep.py --all_dir $dir --dotsize 5 

