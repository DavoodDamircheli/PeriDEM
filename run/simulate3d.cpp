#define EIGEN_DONT_PARALLELIZE
#include <chrono>
using namespace std::chrono;
#include "Eigen/Dense"
#include <iostream>
#include <string>
// #include "../lib/eigen-3.2.10/Eigen/Dense"
// #include "../lib/eigen-3.2.10/Eigen/Core"
// #include "../lib/eigen-3.2.10/Eigen/IterativeLinearSolvers"
// #include "../lib/eigen-3.2.10/Eigen/Sparse"
using namespace Eigen;

// redirect output to file with freopen()
#include <cstdio>

#include <omp.h>

#include <mpi.h>

// for getopt
#include <unistd.h>

// #include "read/varload.h"

// #include "particle/particle.h"
// #include "particle/particle2.h"
// #include "read/load_mesh.h"

// #include "read/load_csv.h"
#include "read/load_hdf5.h"
#include "read/read_config.h"

#include "particle/timeloop.h"

static inline void parse_vec3(const std::string &s, double &x, double &y,
                              double &z) {
  std::string t = s;
  std::replace(t.begin(), t.end(), ',', ' ');
  std::istringstream iss(t);
  iss >> x >> y >> z;
}

// #include "particle/nbdarr.h"

using namespace std;

int main(int argc, char *argv[]) {
  const unsigned dim = 3;
  string config_file = "config/main.conf";

  // Eigen::initParallel();

  // Redirect stdout to file
  // freopen( "output.log", "w", stdout );
  // freopen( "error.log", "w", stderr );

  // Allow nested parallel computation
  omp_set_nested(1);

  //// doesn't work here. Need to print this within #omp parallel
  // std::cout << "Available threads: " <<  omp_get_num_threads() << std::endl;

  // MPI initialization
  int numprocessors, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocessors);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  char name[50];
  int count;
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
  // put ':' in the starting of the string so that program can distinguish
  // between '?' and ':'
  while ((opt = getopt(argc, argv, "c:i:o:")) != -1) {
    switch (opt) {
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
  // auto PArr = load_particles<dim>();
  // Contact CN = load_contact();
  // RectWall<dim> Wall = load_wall<dim>();
  auto PArr = load_particles<dim>(CFGV);
  Contact CN = load_contact(CFGV);
  RectWall<dim> Wall = load_wall<dim>(CFGV);
  //-----------------------override values----------------------------
  if (CFGV.override_particle_acc || CFGV.override_particle_vel ||
      CFGV.override_particle_extforce) {

    double ax = 0, ay = 0, az = 0;
    double vx = 0, vy = 0, vz = 0;
    double extx = 0, exty = 0, extz = 0;

    if (CFGV.override_particle_acc)
      parse_vec3(CFGV.particle_acc_value, ax, ay, az);
    if (CFGV.override_particle_vel)
      parse_vec3(CFGV.particle_vel_value, vx, vy, vz);
    if (CFGV.override_particle_extforce)
      parse_vec3(CFGV.particle_extforce_value, extx, exty, extz);

    for (unsigned i = 0; i < PArr.size(); ++i) {

      if (CFGV.override_particle_id >= 0 && (int)i != CFGV.override_particle_id)
        continue;

      auto &P = PArr[i];

      // P.acc and P.vel are per-node arrays (loaded with
      // load_rowvecs<double,dim>)
      if (CFGV.override_particle_acc) {
        for (auto &a : P.acc) { // a is a dim-vector per node
          if (CFGV.add_particle_acc) {
            a[0] += ax;
            a[1] += ay;
            a[2] += az;
          } else {
            a[0] = ax;
            a[1] = ay;
            a[2] = az;
          }
        }
      }

      if (CFGV.override_particle_vel) {
        for (auto &v : P.vel) {
          v[0] = vx;
          v[1] = vy;
          v[2] = vz;
        }
      }
      if (CFGV.override_particle_extforce) {
        for (auto &ex : P.extforce) {
          ex[0] = extx;
          ex[1] = exty;
          ex[2] = extz;
        }
      }
    }
  }

  //-----------------------override values----------------------------
  // default value if not set
  Timeloop TL(100);

  // CFGV.print();
  // CFGV.apply<dim>(TL, CN, Wall);
  TL.apply_config(CFGV);
  CN.apply_config(CFGV);
  Wall.apply_config(CFGV);

  std::cout << "extforce_gradient" << TL.gradient_extforce << std::endl;

  // print info
  // CN.print();
  // Wall.print();
  // PArr[0].print();

  // Debug
  // std::cout << "Debug: making particles breakable so that the contact force
  // utilizes all the nodes, not just the boundary nodes" << std::endl;
  // std::cout << "Debug: Note that breaking bonds depends only on:
  // TL.enable_fracture." << std::endl;

  for (unsigned i = 0; i < PArr.size(); i++) {
    PArr[i].break_bonds = 1;
  }
  auto start = system_clock::now();
  // std::cout << "run_timeloop: My rank number is " << rank  << std::endl;
  run_timeloop<dim>(PArr, TL, CN, Wall, CFGV);

  // more compact code but much slower
  // run_timeloop_compact<dim> (PArr, TL, CN, Wall);
  auto stop = system_clock::now();

  // Close MPI
  MPI_Finalize();

  // are nanoseconds, microseconds, milliseconds, seconds, minutes, hours
  auto duration = duration_cast<seconds>(stop - start);
  // To get the value of duration use the count(), member function on the
  // duration object
  cout << "Runtime: " << duration.count() << "s" << endl;

  return 0;
}
