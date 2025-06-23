#define EIGEN_DONT_PARALLELIZE
#include <chrono> 
using namespace std::chrono; 
# include <string>
# include <iostream>
#include "Eigen/Dense"
//#include "../lib/eigen-3.2.10/Eigen/Dense"
//#include "../lib/eigen-3.2.10/Eigen/Core"
//#include "../lib/eigen-3.2.10/Eigen/IterativeLinearSolvers"
//#include "../lib/eigen-3.2.10/Eigen/Sparse"
using namespace Eigen;

// redirect output to file with freopen()
#include <cstdio>

#include<omp.h>

#include<mpi.h>

// for getopt
#include <unistd.h>

//#include "read/varload.h"

//#include "particle/particle.h"
//#include "particle/particle2.h"
//#include "read/load_mesh.h"

//#include "read/load_csv.h"
#include "read/load_hdf5.h"
#include "read/read_config.h"

#include "particle/timeloop.h"


//#include "particle/nbdarr.h"

using namespace std;

int main(int argc, char *argv[]){
    const unsigned dim = 2;
    string config_file = "config/main.conf";

    //Eigen::initParallel();

    // Redirect stdout to file
    //freopen( "output.log", "w", stdout );
    //freopen( "error.log", "w", stderr );

    // Allow nested parallel computation
    omp_set_nested(1);

    //// doesn't work here. Need to print this within #omp parallel
    //std::cout << "Available threads: " <<  omp_get_num_threads() << std::endl;


    // MPI initialization 
    int numprocessors, rank; 
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocessors);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    char name[50]; int count;
    MPI_Get_processor_name(name, &count);     

    /////////////////////////////////
    // All output from rank 0 only
    // Killing all outputs from other ranks
    std::ofstream sink("/dev/null");
    if (rank != 0) {
	// Mute standard output
	std::cout.rdbuf(sink.rdbuf());
	// Optionally mute standard error
	std::cerr.rdbuf(sink.rdbuf());      
    }
    ////////////////////////////////////

    std::cout << "MPI total processors " << numprocessors << std::endl;

    // Load config file
    auto CFGV = ConfigVal();

    // get command line options
    int opt;
    // colon after an option means it has a parameter
    // put ':' in the starting of the string so that program can distinguish between '?' and ':' 
    while((opt = getopt(argc, argv, "c:i:o:")) != -1) 
    { 
	switch(opt) 
	{ 
	    case 'c': 
		config_file = optarg;
		break; 
	    case 'i': 
		// input setup filename
		CFGV.setup_filename = optarg;
		break; 
	    case 'o': 
		// output data directory
		CFGV.output_dir = optarg;
		break; 
	    case ':': 
		printf("option needs a value\n"); 
		break; 
	} 
    } 

    CFGV.read_file(config_file);

    // load from hdf5 files
    //auto PArr = load_particles<dim>();
    //Contact CN = load_contact();
    //RectWall<dim> Wall = load_wall<dim>();
    auto PArr = load_particles<dim>(CFGV);
    Contact CN = load_contact(CFGV);
    RectWall<dim> Wall = load_wall<dim>(CFGV);

    // default value if not set
    Timeloop TL(100);

    //CFGV.print();
    //CFGV.apply<dim>(TL, CN, Wall);
    TL.apply_config(CFGV);
    CN.apply_config(CFGV);
    Wall.apply_config(CFGV);

    std::cout << "extforce_gradient" << TL.gradient_extforce << std::endl;

    // print info
    //CN.print();
    //Wall.print();
    //PArr[0].print();
	
    // Debug
    //std::cout << "Debug: making particles breakable so that the contact force utilizes all the nodes, not just the boundary nodes" << std::endl;
    //std::cout << "Debug: Note that breaking bonds depends only on: TL.enable_fracture." << std::endl;

    for (unsigned i = 0; i < PArr.size(); i++) {
	PArr[i].break_bonds = 1;
    }
auto start = system_clock::now(); 
    //std::cout << "run_timeloop: My rank number is " << rank  << std::endl;
    run_timeloop<dim> (PArr, TL, CN, Wall, CFGV);

    // more compact code but much slower
    //run_timeloop_compact<dim> (PArr, TL, CN, Wall);
auto stop = system_clock::now(); 

    // Close MPI
    MPI_Finalize();

    // are nanoseconds, microseconds, milliseconds, seconds, minutes, hours
    auto duration = duration_cast<seconds>(stop - start); 
    // To get the value of duration use the count(), member function on the duration object 
    cout << "Runtime: " << duration.count() << "s" << endl; 

    return 0;
}
