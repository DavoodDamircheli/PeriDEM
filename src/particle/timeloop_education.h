
#ifndef TIMELOOP_H
#define TIMELOOP_H
#include <iomanip>
#include <chrono>
using namespace std::chrono;

#include <chrono>
#include <iomanip>
#include <vector>
#include <mpi.h>
#include "read/read_config.h"
#include "compat/overloads.h"
#include "particle/particle2.h"
#include "particle/timeloop.h"
#include "particle/contact.h"
#include "read/rw_hdf5.h"

#include <omp.h>
#include <ctime>
#include <time.h>
#include <mpi.h>

// Convert NbdArr to connectivity:   -----------NbdArr2conn------------
/*

C++ function, NbdArr2conn
NbdArr = {
  {1, 2},     // 0 is connected to 1, 2
  {0, 2},     // 1 is connected to 0, 2
  {0, 1}      // 2 is connected to 0, 1
};

Conn = {
  {0, 1},     // Pair for 0 → 1
  {0, 2},     // Pair for 0 → 2
  {1, 2}      // Pair for 1 → 2
};
*/

vector<Matrix<unsigned, 1, 2>> NbdArr2conn(vector<vector<unsigned>> NbdArr) {
  vector<Matrix<unsigned, 1, 2>> Conn;

  for (unsigned i = 0; i < NbdArr.size(); i++) {
    for (unsigned j = 0; j < NbdArr[i].size(); j++) {
      auto q = NbdArr[i][j];

      // To avoid counting all the bonds twice, we only allow sub-diagonal
      // pairs, this essentially gets rid of the copy with reverse order
      if (i < q) {
        Matrix<unsigned, 1, 2> v = {i, q};
        //// append
        Conn.push_back(v);
      }
    }
  }
  return Conn;
};
/*
C++ function, ------format_ms---------------
Input

int ms = 3723000; // 3723000 milliseconds (1 hour, 2 minutes, and 3 seconds)
Output

"1:02:03"
*/
std::string format_ms(int ms) {
    int s = ms / 1000;
    //int rem_ms = ms % 1000;
    int min = s / 60;
    int rem_s = s % 60;
    int hr = min / 60;
    int rem_min = min % 60;
  std::stringstream ss;
  ss <<  hr  << ":"
      << std::setfill('0') << std::setw(2) << rem_min << ":"
      << std::setfill('0') << std::setw(2) << rem_s ;
  return ss.str();
}

/*
This C++ template function calculates the mean (average) of a collection of matrices (or vectors) stored in a std::vector

Input

vector<Matrix<double, 1, 3>> v = {
    Matrix<double, 1, 3>({1.0, 2.0, 3.0}),
    Matrix<double, 1, 3>({4.0, 5.0, 6.0}),
    Matrix<double, 1, 3>({7.0, 8.0, 9.0}),
};

Output

Matrix<double, 1, 3>({4.0, 5.0, 6.0});
*/

template <unsigned dim> auto mean(vector<Matrix<double, 1, dim>> v) {
  auto nnodes = v.size();
  Matrix<double, 1, dim> mean;
  mean.setZero();
  for (unsigned inx = 0; inx < nnodes; inx++) {
    mean += v[inx];
  }
  mean /= nnodes;
  return mean;
};

/*

This C++ code defines a Timeloop class, which manages time-stepping for a simulation. 
It includes variables to control the simulation state, 
methods to handle initializtion and resumption, 
and functionality for interacting with particles and walls.
*/

// Timeloop class to handle time-stepping
class Timeloop {
public:
  bool do_resume = 0;
  bool wall_resume = 0;
  bool save_file;

  unsigned resume_ind;
  double dt;
  unsigned timesteps, modulo;

  bool enable_fracture;
  bool override_fracture_toughness = 0;
  double new_snot;

  bool run_parallel;

  unsigned counter;
  unsigned first_counter, last_counter=0;

  bool gradient_extforce = 0;
  bool enable_torque = 0;
  bool enable_velocity_constraint = 0;
  unsigned extforce_maxstep = 0;

  int set_damping_timestep = -1;
  int set_zero_wall_speed_timestep = -1;

  int set_movable_index = -1;
  int set_movable_timestep = -1;
  int set_stoppable_index = -1;
  int set_stoppable_timestep = -1;

  bool reset_partzero_y = 0;
  unsigned reset_partzero_y_timestep;
  double wheel_rad;

  // forcefield variables
  int forcefield_type;
  double forcefield_scaling;

  // saving the runtime
  // double start_time;
  std::chrono::_V2::system_clock::time_point start_time;

  vector<double> run_time, t_ind;
/*

Single-Argument Constructor

*/
  Timeloop(unsigned ts) {
    timesteps = ts;

    do_resume = 0;
    resume_ind = 0;

    dt = 0.02 / 1e5;
    modulo = 100;

    // use_influence_function = 0;

    enable_fracture = 0;

    save_file = 1;

    run_parallel = 1;

    start_time = system_clock::now();
  };
/*

Two-Argument Constructor:


*/
  Timeloop(unsigned ts, unsigned m) : Timeloop(ts) { modulo = m; };


//------------------------------------------------------------------



//--------------------------------------------
//-----------------Methods -------------------
//--------------------------------------------

/*

Key Method: update_on_resume

This method updates the simulation state if resuming from a saved run.

Input Parameters

    vector<ParticleN<dim>> &PArr: A reference to a collection of particles.
    RectWall<dim> &Wall: A reference to a wall object.

Template Parameter:

    dim: Dimensionality of the simulation (e.g., 2D or 3D). This makes the function applicable to simulations of different dimensions.

Input Parameters:

    PArr: A reference to a vector of ParticleN<dim> objects, representing particles in the simulation.
    Wall: A reference to a RectWall<dim> object, representing the wall or boundary in the simulation.
Applications

This function is ideal for:

    Simulations involving particles and boundaries (e.g., fluid dynamics, granular flow).
    Large-scale computations where interruptions are possible and resumption is necessary.
    Scenarios where simulation states need to be saved and reloaded efficiently.
*/
  template <unsigned dim>
  void update_on_resume(vector<ParticleN<dim>> &PArr, RectWall<dim> &Wall) {
    if (do_resume) {
      counter = resume_ind + 1;
      first_counter = counter;
      last_counter = resume_ind + timesteps / modulo;

      string data_loc = "output/hdf5/";
      char buf[20];
      sprintf(buf, "tc_%05u.h5", resume_ind);
      string h5file = string(buf);
      string filename = data_loc + h5file;
      std::cout << "Loading from file: " << filename << std::endl;

      for (unsigned i = 0; i < PArr.size(); ++i) {
        char buf2[100];
        sprintf(buf2, "P_%05u", i);
        string particlenum = string(buf2);
        PArr[i].disp =
            load_rowvecs<double, dim>(filename, particlenum + "/CurrPos") -
            PArr[i].pos;
        PArr[i].vel = load_rowvecs<double, dim>(filename, particlenum + "/vel");
        PArr[i].acc = load_rowvecs<double, dim>(filename, particlenum + "/acc");
      }

      if (wall_resume) {
        // resume wall info
        char buf_w[20];
        sprintf(buf_w, "wall_%05u.h5", resume_ind);
        string wall_file = string(buf_w);
        string filename_wall = data_loc + wall_file;
        std::cout << "Loading wall from file: " << filename_wall << std::endl;
        auto vv = load_col<double>(filename_wall, "wall_info");
        Wall.set_lrtb(vv);

        // std::cout << "wall info now: " << Wall.lrtb() << std::endl;
      }
      std::cout << "Resuming from counter " << counter << std::endl;
    } else {
      counter = 1;
      first_counter = counter;
      last_counter = timesteps / modulo;
    }
  };

/*
ChatGPT

The function apply_config updates the current state of 
a Timeloop object by applying configuration values from a ConfigVal object (CFGV). 
It essentially sets various simulation parameters based on the provided configuration. 

Input:

    CFGV: An object of type ConfigVal, which holds configuration parameters for the simulation.

Output:

    None. The function modifies the current Timeloop object directly.

*/
  void apply_config(ConfigVal CFGV) {
    // update from config values
    timesteps = CFGV.timesteps;
    modulo = CFGV.modulo;
    dt = CFGV.dt;
    do_resume = CFGV.do_resume;
    wall_resume = CFGV.wall_resume;
    resume_ind = CFGV.resume_ind;
    save_file = CFGV.save_file;     //save_file: Determines whether the simulation state should be saved to files.
    enable_fracture = CFGV.enable_fracture;
    run_parallel = CFGV.is_parallel;

    if (CFGV.new_snot != (-1)) {
      override_fracture_toughness = 1;
      new_snot = CFGV.new_snot;
    }

    gradient_extforce = CFGV.gradient_extforce;
    extforce_maxstep = CFGV.extforce_maxstep;
    // self-contact
    enable_torque = CFGV.enable_torque;
    // enable_velocity_constraint = CFGV.enable_velocity_constraint;
    // set_torque_timestep = CFGV.set_torque_timestep;

    set_damping_timestep = CFGV.set_damping_timestep;
    set_zero_wall_speed_timestep = CFGV.set_zero_wall_speed_timestep;

    // turn on movable
    set_movable_index = CFGV.set_movable_index;
    set_movable_timestep = CFGV.set_movable_timestep;
    // turn on stoppable
    set_stoppable_index = CFGV.set_stoppable_index;
    set_stoppable_timestep = CFGV.set_stoppable_timestep;

    // move wheel to top of bulk
    reset_partzero_y = CFGV.reset_partzero_y;
    reset_partzero_y_timestep = CFGV.reset_partzero_y_timestep;
    wheel_rad = CFGV.wheel_rad;

    forcefield_type = CFGV.forcefield_type;
    forcefield_scaling = CFGV.forcefield_scaling;
  };

private:
  /* data */
};

// Runs the time loop simulation
template <unsigned dim>
void run_timeloop(vector<ParticleN<dim>> &PArr, Timeloop TL, Contact CN,
                  RectWall<dim> Wall, ConfigVal CFGV) {

  int numprocessors, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &numprocessors);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  unsigned total_particles_univ = PArr.size();
  unsigned counter;
  unsigned first_counter, last_counter;

  // generating all pairs a priori for contact detection
  unsigned total_particle_pair =
      total_particles_univ * (total_particles_univ - 1) / 2;
  vector<vector<unsigned>> pairs(total_particle_pair);
  unsigned p = 0;
  for (unsigned i = 0; i < total_particles_univ; i++) {
    for (unsigned j = i + 1; j < total_particles_univ; j++) {
      pairs[p].resize(2);
      pairs[p][0] = i;
      pairs[p][1] = j;
      p++;
    }
  }

  // string output_loc = "output/hdf5/";
  string output_loc = CFGV.output_dir;

    std::cout << "Output data dir: " << output_loc << std::endl;

  if (TL.do_resume) {
    counter = TL.resume_ind + 1;
    first_counter = counter;
    last_counter = TL.resume_ind + TL.timesteps / TL.modulo;

    // string data_loc = "output/hdf5/";
    string data_loc = CFGV.output_dir;
    char buf[20];
    sprintf(buf, "/tc_%05u.h5", TL.resume_ind);
    string h5file = string(buf);
    string filename = data_loc + h5file;
      std::cout << "Loading from file: " << filename << std::endl;

    for (unsigned i = 0; i < total_particles_univ; ++i) {
      char buf2[100];
      sprintf(buf2, "P_%05u", i);
      string particlenum = string(buf2);
      PArr[i].disp =
          load_rowvecs<double, dim>(filename, particlenum + "/CurrPos") -
          PArr[i].pos;
      PArr[i].vel = load_rowvecs<double, dim>(filename, particlenum + "/vel");
      PArr[i].acc = load_rowvecs<double, dim>(filename, particlenum + "/acc");

      if (TL.enable_fracture) {
        // load connectivity
        auto Conn =
            load_rowvecs<unsigned, 2>(filename, particlenum + "/Connectivity");
        PArr[i].NbdArr = conn2NArr(Conn, PArr[i].nnodes);
        PArr[i].gen_xi();
      }
    }

    if (TL.wall_resume) {
      // resume wall info
      char buf_w[20];
      sprintf(buf_w, "/wall_%05u.h5", TL.resume_ind);
      string wall_file = string(buf_w);
      string filename_wall = data_loc + wall_file;
        std::cout << "Loading wall from file: " << filename_wall << std::endl;
      auto vv = load_col<double>(filename_wall, "wall_info");
      Wall.set_lrtb(vv);

      // std::cout << "wall info now: " << Wall.lrtb() << std::endl;
    }

      std::cout << "Resuming from counter " << counter << std::endl;
  } else {
    counter = 1;
    first_counter = counter;
    last_counter = TL.timesteps / TL.modulo;
  }

  if (TL.override_fracture_toughness) {
    for (unsigned i = 0; i < total_particles_univ; i++) {
      PArr[i].snot = TL.new_snot;
    }
  }

  // Only rank 0 saves the file plotinfo
  if (rank == 0) {

    // Save plotinfo
    // string plt_filename = "output/hdf5/plotinfo.h5";
    string plt_filename = CFGV.output_dir + "/plotinfo.h5";
    if (rank == 0) {
      std::cout << "Output plotinfo: " << plt_filename << std::endl;
    }
    H5::H5File pl_fp(plt_filename.c_str(), H5F_ACC_TRUNC);

    vector<int> al_wall = {Wall.allow_wall};
    vector<double> geom_wall_info = Wall.lrtb();
    vector<unsigned> f_l_counter = {first_counter, last_counter};
    // std::cout << "geom_wall_info" << geom_wall_info << std::endl;
    store_col<unsigned>(pl_fp, "f_l_counter", f_l_counter);

    vector<unsigned> vec_dim = {dim};
    store_col<unsigned>(pl_fp, "dimension", vec_dim);

    vector<double> vec_dt = {TL.dt};
    store_col<double>(pl_fp, "dt", vec_dt);

    vector<unsigned> vec_modulo = {TL.modulo};
    store_col<unsigned>(pl_fp, "modulo", vec_modulo);

    pl_fp.createGroup("/wall");
    store_col<int>(pl_fp, "/wall/allow_wall", al_wall);
    store_col<double>(pl_fp, "/wall/geom_wall_info", geom_wall_info);
    pl_fp.close();
  }

  time_t my_time = time(NULL);

  unsigned column_w = 12;

  char str_count[50];
    std::cout
        << "---------------------------------------------------------------"
           "-------"
        << std::endl;
    std::cout << "Starting time loop "
              << "(Parallel=" << TL.run_parallel << "): " << ctime(&my_time);
    sprintf(str_count, "ct[%d:%d]", first_counter, last_counter);
    std::cout << left << std::setw(column_w) << "t " << left
              << std::setw(column_w) << string(str_count) << left
              << std::setw(column_w+5) << "duration(s)" << left
              << std::setw(column_w) << "rem(h:m:s)" << left
              << std::setw(column_w) << "#contacts" << left
              << std::setw(column_w) << "time" << std::endl;

  auto start_time = system_clock::now();
  vector<double> run_time, t_ind;
  vector<unsigned> pw_comp_count_vec;

  unsigned pw_comp_count = 0;

  /****
   * Giant force vector that communicates between processes *
   ****/
  unsigned total_univ_nodes = 0;
  // vector containing the starting indices of particles
  vector<unsigned> giant_index(total_particles_univ);
  for (unsigned i = 0; i < PArr.size(); i++) {
    giant_index[i] = total_univ_nodes;
    total_univ_nodes += PArr[i].nnodes;
  }
  vector<Matrix<double, 1, dim>> giant_f(total_univ_nodes);

  vector<vector<double>> lrtb(total_particles_univ);
  for (unsigned i = 0; i < PArr.size(); i++) {
    PArr[i].CurrPos = PArr[i].pos + PArr[i].disp;
    lrtb[i] = get_minmax<dim>(PArr[i].CurrPos);
  }

  for (unsigned t = 1; t <= TL.timesteps; ++t) {
    // set wall reaction to zero
    Wall.setzero_reaction();

    // initialize  the giant force vector by zero, will reduce it later via
    // addition over all processes
    for (unsigned i = 0; i < giant_f.size(); i++) {
      giant_f[i] = Matrix<double, 1, dim>::Zero();
    }

    // turns on previously disabled torque
    if (CFGV.set_torque_timestep != (-1)) {
      if (t == (unsigned)CFGV.set_torque_timestep) {
        TL.enable_torque = 1;
          std::cout << "Setting torque to " << TL.enable_torque
                    << " on timestep " << t << std::endl;
      }
    }

    if (CFGV.set_fix_velocity_timestep != (-1)) {
      if (t == (unsigned)CFGV.set_fix_velocity_timestep) {
        TL.enable_velocity_constraint = 1;
          std::cout << "Setting velocity constraint to "
                    << TL.enable_velocity_constraint << " on timestep " << t
                    << std::endl;
      }
    }

    if (CFGV.set_fracture_timestep != (-1)) {
      if (t == (unsigned)CFGV.set_fracture_timestep) {
        TL.enable_fracture = 1;
          std::cout << "Setting fracture to " << TL.enable_fracture
                    << " on timestep " << t << std::endl;
      }
    }

    if (CFGV.set_self_contact_timestep != (-1)) {
      if (t == (unsigned)CFGV.set_self_contact_timestep) {
        CN.self_contact = 1;
          std::cout << "Setting self_contact to " << CN.self_contact
                    << " on timestep " << t << std::endl;
      }
    }

    if (TL.set_damping_timestep != (-1)) {
      if (t == (unsigned)TL.set_damping_timestep) {
        CN.allow_damping = 1;
          std::cout << "Setting damping to " << CN.allow_damping
                    << " on timestep " << t << std::endl;
      }
    }

    if (TL.set_zero_wall_speed_timestep != (-1)) {
      if (t == (unsigned)TL.set_zero_wall_speed_timestep) {
	if (dim == 2) {
	    Wall.speed_left = 0;
	   Wall.speed_right = 0;
	     Wall.speed_top = 0;
	  Wall.speed_bottom = 0;
	} else {
	  Wall.speed_x_min =0;
	  Wall.speed_y_min =0;
	  Wall.speed_z_min =0;
	  Wall.speed_x_max =0;
	  Wall.speed_y_max =0;
	  Wall.speed_z_max =0;
	}
          std::cout << "Setting all wall speed to zero on timestep " << t << std::endl;
      }
    }

    // brazil nut
    if (CFGV.brazil) {

	int total_freq = CFGV.brazil_reset_wall_bottom_freq + CFGV.brazil_nonzero_wall_bottom_speed_freq;
	int  this_t = ((int) t % total_freq);

      if (this_t <= CFGV.brazil_nonzero_wall_bottom_speed_freq) {
	  Wall.speed_bottom = CFGV.speed_wall_bottom;
      }
      else{
	  Wall.speed_bottom = 0;
	  Wall.bottom = CFGV.brazil_reset_wall_bottom_y;
      }
    }

    if (TL.set_movable_index != (-1)) {
      auto part_ind = (unsigned)TL.set_movable_index;
      auto part_ts = (unsigned)TL.set_movable_timestep;
      if (part_ts == t) {
          std::cout << "Setting particle " << TL.set_movable_index
                    << " to movable on timestep " << t << std::endl;
        PArr[part_ind].movable = 1;
      }
    }
    if (TL.set_stoppable_index != (-1)) {
      auto part_ind = (unsigned)TL.set_stoppable_index;
      auto part_ts = (unsigned)TL.set_stoppable_timestep;
      if (part_ts == t) {
          std::cout << "Setting particle " << TL.set_stoppable_index
                    << " to stoppable on timestep " << t << std::endl;
        PArr[part_ind].stoppable = 1;
      }
    }

    if (TL.reset_partzero_y) {
      auto part_ts = (unsigned)TL.reset_partzero_y_timestep;
      if (part_ts == t) {
        // find max_y of the bulk: i=1,...
        double max_bulk_y = PArr[1].pos[0](1) + PArr[1].disp[1](1);
        //double max_bulk_y;
	bool init = 1;	// whether initial assignment
        for (unsigned i = 1; i < total_particles_univ; ++i) {
          for (unsigned j = 0; j < PArr[i].nnodes; j++) {
	      double now_x = PArr[i].pos[j](0) + PArr[i].disp[j](0);
	      double now_y = PArr[i].pos[j](1) + PArr[i].disp[j](1);
	      // only local bulk height
	      if (abs(PArr[0].mean_CurrPos()(0)- now_x) <= TL.wheel_rad) {
		  if (init) {
		      // initial assignment to maximum as the first feasible element 
		      max_bulk_y = now_y;
		      init = !init;
		  }
		  else{
		      if (now_y > max_bulk_y) {
			  max_bulk_y = now_y;
		    }
		  }
	      }
          }
        }
        // move the mean by this amount
        double dest = max_bulk_y + TL.wheel_rad + CN.contact_rad;
        double to_move_by = dest - PArr[0].mean_CurrPos()(1);
          std::cout << "Setting particle zero mean y val to  " << dest
                    << std::endl;
        for (unsigned j = 0; j < PArr[0].nnodes; j++) {
          PArr[0].pos[j](1) += to_move_by;
        }
      }
    }

    for (unsigned i = 0; i < total_particles_univ; ++i) {
      PArr[i].disp += TL.dt * PArr[i].vel + (TL.dt * TL.dt * 0.5) * PArr[i].acc;
      PArr[i].CurrPos = PArr[i].pos + PArr[i].disp;
    }

    // for (unsigned i = 0; i < total_particles_univ; ++i) {
    //#pragma omp parallel for schedule(dynamic) if (TL.run_parallel)
    for (size_t i = rank; i < total_particles_univ; i += numprocessors) { // mpi loop
      if (PArr[i].movable) {
        auto temp_ft_i = PArr[i].get_peridynamic_force();
        for (unsigned node = 0; node < PArr[i].nnodes; node++) {
          giant_f[node + giant_index[i]] += temp_ft_i[node];
        }
        if (TL.enable_fracture) {
          if (PArr[i].breakable) {
            PArr[i].remove_bonds();
            // self-contact: update ft_i, 1==is_self_contact
            if (CN.self_contact) {
              update_contact_force_by_boundary<dim>(PArr[i], PArr[i], CN, TL.dt,
                                                    giant_f, giant_index[i],
                                                    giant_index[i], 1);
            }
          }
        }
        if (PArr[i].stoppable) {
          auto C_lrtb = lrtb[i];
          if (Wall.allow_wall) {
            if (!within_interior_fext(C_lrtb, Wall.lrtb(), CN.contact_rad)) {
              update_wall_contact_force_by_boundary<dim>(PArr[i], Wall, CN,
                                                         TL.dt, giant_f,
                                                         giant_index[i]);
            }
          }

          // if (CN.allow_forcefield) {
          // ParticleN<dim> P_C, double dt, vector<Matrix<double, 1, dim>>
          // &combined_contact_force, double forcefield_scaling, bool
          // normalized, bool by_boundary) {
          //apply_force_field<dim>(
              //PArr[i], TL.forcefield_type, TL.dt, t, TL.timesteps,
              //giant_f, giant_index[i], TL.forcefield_scaling, 1, 0);
          //}
        }

        // external forces
        if (TL.gradient_extforce) {
          double extf_sc;
          extf_sc = (double)t / (double)TL.extforce_maxstep;

          if (extf_sc > 1) {
            extf_sc = 1;
          }
          for (unsigned node = 0; node < PArr[i].nnodes; node++) {
            giant_f[node + giant_index[i]] += extf_sc * PArr[i].extforce[node];
          }

        } else {
          for (unsigned node = 0; node < PArr[i].nnodes; node++) {
            giant_f[node + giant_index[i]] += PArr[i].extforce[node];
          }
        }
      }

      // external torque about the centroid
      if (TL.enable_torque) {
        // centroid about which the torque is applied
        Matrix<double, 1, dim> c = PArr[i].mean_CurrPos();
        unsigned taxis = PArr[i].torque_axis;

        for (unsigned nn = 0; nn < PArr[i].CurrPos.size(); nn++) {
          Matrix<double, 1, dim> r_vec_proj = PArr[i].CurrPos[nn] - c;
          double r_norm = r_vec_proj.norm();
          double r_inv = (r_norm ? 1 / r_norm : 0);
          // project on the plane perpendicular to torque_axis
          // In 2d, always set torque_axis=2 (i.e, z-axis)
          if (dim == 3) {
            r_vec_proj(taxis) = 0;
          }
          // perpendicular to r vector -> r_perp
          Matrix<double, 1, dim> r_perp;
          if (dim == 3) {
            r_perp(taxis) = 0;
          }
          r_perp((taxis + 1) % 3) = -r_vec_proj((taxis + 2) % 3);
          r_perp((taxis + 2) % 3) = r_vec_proj((taxis + 1) % 3);
          // unit perpendicular direction to r vector -> r_perp
          auto perp_norm = r_perp.norm();
          if (perp_norm > 0) {
            r_perp /= perp_norm;
          } else {
            r_perp((taxis + 1) % 3) = 0;
            r_perp((taxis + 2) % 3) = 0;
          }
          // add force density due to torque to the total force density
          giant_f[nn + giant_index[i]] += (r_inv * PArr[i].torque_val * r_perp);
        }
      }
    }

    int pairwise_computations = 0;
    for (size_t p = rank; p < total_particle_pair;
         p += numprocessors) { // mpi loop
      unsigned i = pairs[p][0];
      unsigned j = pairs[p][1];
      if (PArr[i].movable || PArr[j].movable) {
        auto C_lrtb = lrtb[i];
        auto N_lrtb = lrtb[j];
        if (intersects_box_fext<dim>(C_lrtb, N_lrtb, CN.contact_rad)) {
          update_contact_force_by_boundary<dim>(PArr[i], PArr[j], CN, TL.dt,
                                                giant_f, giant_index[i],
                                                giant_index[j], 0);
          pairwise_computations++;
        }
      } // movable || movable
    }

    pw_comp_count += pairwise_computations;
    //pw_comp_count /= TL.modulo;

    // MPI_Barrier( MPI_COMM_WORLD);
    // Communicate
    MPI_Allreduce(MPI_IN_PLACE, giant_f[0].data(), dim * total_univ_nodes,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    //#pragma omp parallel for if (TL.run_parallel)
    for (unsigned i = 0; i < total_particles_univ; ++i) {
      // extract force from giant force vector after synchronized sum 
      for (unsigned node = 0; node < PArr[i].nnodes; node++) {
        PArr[i].force[node] = giant_f[giant_index[i] + node];
      }
      // update if movable
      if (PArr[i].movable) {
        // store the old acceleration before replacing by the new one
        auto temp_acc = PArr[i].acc;
        auto acc_old = PArr[i].acc;

        PArr[i].acc = (1 / PArr[i].rho) * (PArr[i].force);

        //	// velocity
        //	u0dot_univ{i} = uolddot_univ{i} + dt * 0.5 * uolddotdot_univ{i}
        //+ dt * 0.5 * u0dotdot_univ{i}; PArr[i].vel = PArr[i].vel  + (dt * 0.5)
        //*
        // PArr[i].acc + (dt * 0.5) * PArr[i].acc_old;
        temp_acc += PArr[i].acc;   // Now temp_acc = (acc + acc_old)
        temp_acc *= (0.5 * TL.dt); // now temp_acc = (dt*0.5) *(acc + acc_old)
        PArr[i].vel += temp_acc;

        // prescribed_velocity modification
        if ((i == (unsigned)CFGV.particle_w_fixed_velocity) &&
            (TL.enable_velocity_constraint)) {
          vector<Matrix<double, 1, dim>> prescribed_velocity(PArr[i].nnodes);

          //// Decomposing velocity into rotation and translation part (velocity=Ax+b)
          // translation part of the velocity
          Matrix<double, 1, dim> mean_vel;
          mean_vel.setZero();
          for (unsigned node = 0; node < PArr[i].nnodes; node++) {
            mean_vel += PArr[i].vel[node];
          }
          mean_vel /= PArr[i].nnodes;
          // rotation part of the velocity Ax+b
          vector<Matrix<double, 1, dim>> vel_rotation(PArr[i].nnodes);
          for (unsigned node = 0; node < PArr[i].nnodes; node++) {
            vel_rotation[node] = PArr[i].vel[node] - mean_vel;
          }

          // edit the translational part: mean_vel
          if (CFGV.fix_velocity_mean_x) {
            mean_vel(0) = CFGV.prescribed_velocity_mean_x;
          }

          //// edit the rotational part: vel_rotation
          if (CFGV.fix_velocity_angular) {
            // std::cout << "fixing angular vel " << std::endl;
            Matrix<double, 1, dim> i_mean_pos = PArr[i].mean_CurrPos();
            for (unsigned node = 0; node < PArr[i].nnodes; node++) {
              // velocity in the moving frame of reference
              // Matrix<double,1, dim> ref_vel = PArr[i].vel[node] - mean_vel;
              Matrix<double, 1, dim> ref_vel = vel_rotation[node];

              Matrix<double, 1, dim> r_vec = PArr[i].CurrPos[node] - i_mean_pos;
              double r = r_vec.norm();
              Matrix<double, 1, dim> r_unit;
              r_unit.setZero();
              if (r > 0) {
                r_unit = r_vec / r;
              }
              Matrix<double, 1, dim> v_r = (ref_vel.dot(r_unit)) * r_unit;
              Matrix<double, 1, dim> v_t = ref_vel - v_r;
              double v_t_norm = v_t.norm();
              Matrix<double, 1, dim> t_unit;
              t_unit.setZero();
              if (t > 0) {
                t_unit = v_t / v_t_norm;
              }

              // prescribed tangential velocity from prescribed angular velocity
              // Matrix<double,1, dim> prescribed_v_t = (r *
              // CFGV.prescribed_angular_vel) * t_unit;
              Matrix<double, 1, dim> prescribed_v_t;
              prescribed_v_t.setZero();
              prescribed_v_t(0) =
                  -(r * CFGV.prescribed_angular_vel) * r_unit(1);
              prescribed_v_t(1) = (r * CFGV.prescribed_angular_vel) * r_unit(0);

              // reconstruct full vel
              // prescribed_velocity[node] = (prescribed_v_t + v_r) +
              // prescribed_mean_vel;
              //// Turning off radial component
              // vel_rotation[node] = prescribed_v_t + v_r;
              vel_rotation[node] = prescribed_v_t;
            }
          }

          //// add the modified rotational and translational components back
          for (unsigned node = 0; node < PArr[i].nnodes; node++) {
            prescribed_velocity[node] = vel_rotation[node] + mean_vel;
          }
          //// assign the prescribed velocity
          PArr[i].vel = prescribed_velocity;

          // compute the applied force density to maintain the prescribed
          // velocity
          PArr[i].applied_force_density =
              PArr[i].rho * ((2 / TL.dt) * (prescribed_velocity - PArr[i].vel) -
                             acc_old) -
              PArr[i].force;
        }

        // clamped nodes are set to be zero
        if (PArr[i].clamped_nodes.size()) {
          for (unsigned cn = 0; cn < PArr[i].clamped_nodes.size(); cn++) {
            auto cid = PArr[i].clamped_nodes[cn];
            PArr[i].disp[cid] = Matrix<double, 1, dim>::Zero();
            PArr[i].vel[cid] = Matrix<double, 1, dim>::Zero();
            PArr[i].acc[cid] = Matrix<double, 1, dim>::Zero();
          }
        }

        // displacement update from velocity
        // PArr[i].disp += TL.dt * PArr[i].vel + (TL.dt * TL.dt * 0.5) *
        // PArr[i].acc;
        lrtb[i] = get_minmax<dim>(PArr[i].CurrPos);
      }
    }
    // std::cout << "Done updating states" << std::endl;

    // wall boundary update
    if (dim == 2) {
      Wall.left += Wall.speed_left * TL.dt;
      Wall.right += Wall.speed_right * TL.dt;
      Wall.top += Wall.speed_top * TL.dt;
      Wall.bottom += Wall.speed_bottom * TL.dt;
    } else {
      Wall.x_min += Wall.speed_x_min * TL.dt;
      Wall.y_min += Wall.speed_y_min * TL.dt;
      Wall.z_min += Wall.speed_z_min * TL.dt;
      Wall.x_max += Wall.speed_x_max * TL.dt;
      Wall.y_max += Wall.speed_y_max * TL.dt;
      Wall.z_max += Wall.speed_z_max * TL.dt;
    }

    // save
    // if (0) {
    if (TL.save_file) {
      // if (TL.save_file) {
      int writer_rank = 0;

      if ((t % TL.modulo) == 0) {

        if (rank != writer_rank) {
          // send connectivity data
          for (size_t i = rank; i < total_particles_univ;
               i += numprocessors) { // mpi loop
            // convert to connectivity to
            auto conn = NbdArr2conn(PArr[i].NbdArr);
            int size_conn = conn.size();
            // send to rank 0 with tag i
            MPI_Send(conn[0].data(), size_conn * 2, MPI_UNSIGNED, writer_rank,
                     i, MPI_COMM_WORLD);
            // std::cout << "Rank " << rank << " sent data of size " <<
            // size_conn << std::endl;
          }

        } else {
          char buf[100];
          sprintf(buf, "%05u", counter);
          string tcounter = string(buf);

          string filename = output_loc + "/tc_" + tcounter + ".h5";
          H5::H5File fp(filename.c_str(), H5F_ACC_TRUNC);

          //#pragma omp parallel for if (TL.run_parallel)
          // for (size_t i = rank; i < total_particles_univ; i+=numprocessors){
          // // mpi loop
          for (unsigned i = 0; i < total_particles_univ; i++) {
            char buf2[100];
            sprintf(buf2, "P_%05u", i);
            string particlenum = string(buf2);
            fp.createGroup(particlenum.c_str()); // Last Error
            store_rowvec<double, dim>(fp, particlenum + "/CurrPos",
                                      PArr[i].CurrPos);
            store_rowvec<double, dim>(fp, particlenum + "/vel", PArr[i].vel);
            store_rowvec<double, dim>(fp, particlenum + "/acc", PArr[i].acc);
            // This is actually the force density (Force/Volume), NOT the Force
            store_rowvec<double, dim>(fp, particlenum + "/force",
                                      PArr[i].force);
            if (i == (unsigned)CFGV.particle_w_fixed_velocity) {
              store_rowvec<double, dim>(fp,
                                        particlenum + "/applied_force_density",
                                        PArr[i].applied_force_density);
            }
            // We will output the Force (Newton), not the force density
            // (Newton/vol)
            // store_rowvec<double, dim>(fp, particlenum + "/force",
            // PArr[i].force*PArr[i].vol);

            // get connectivity data from other ranks
            int source = i % numprocessors;
            if (source != writer_rank) {

              int tag = i;
              // probe to get size
              MPI_Status status;
              MPI_Probe(source, tag, MPI_COMM_WORLD, &status);
              // When probe returns, the status object has the size and other
              // attributes of the incoming message. Get the message size
              int number_amount;
              MPI_Get_count(&status, MPI_UNSIGNED, &number_amount);

              //if (number_amount % 2) {
                //std::cout << "Error getting connectivity info" << std::endl;
              //}
              vector<Matrix<unsigned, 1, 2>> temp_conn(number_amount / 2);
              // receive from rank = ? tag i
              // MPI_Recv(temp_conn[0].data(), number_amount, MPI_UNSIGNED,
              // source, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              MPI_Recv(temp_conn[0].data(), number_amount, MPI_UNSIGNED, source,
                       tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              store_rowvec<unsigned, 2>(fp, particlenum + "/Connectivity",
                                        temp_conn);
            } else {
              // save connectivity, converting NbdArr
              store_rowvec<unsigned, 2>(fp, particlenum + "/Connectivity",
                                        NbdArr2conn(PArr[i].NbdArr));
            }

            // debug, works
            // store_col<double>(fp, particlenum+"/vol", PArr[i].vol);
          }
          fp.close();

          // save wall boundary info
          // if (Wall.move_bdry) {
          // if the wall moves, otherwise a waste of space
          // Save this file always
          char buf_wall[100];
          sprintf(buf_wall, "%05u", counter);
          string tcounter_wall = string(buf);

          string filename_wall = output_loc + "/wall_" + tcounter + ".h5";
          H5::H5File fp_w(filename_wall.c_str(), H5F_ACC_TRUNC);

          store_col<double>(fp_w, "wall_info", Wall.lrtb());

          if (CN.compute_wall_reaction) {
            store_rowvec<double, dim>(fp_w, "reaction", Wall.get_reaction());
          }

          fp.close();
          //}

          auto stop_time = system_clock::now();
          auto duration_raw = (stop_time - start_time);
          // are nanoseconds, microseconds, milliseconds, seconds, minutes, hours 
          auto duration = duration_cast<milliseconds>(duration_raw);
          // update for next loop
          start_time = stop_time;

          // prediction: from last n avg
          int avgsteps = 10;
          double avg_duration = 0;
	  int start_ind = (run_time.size() - avgsteps);
	  if (start_ind > 0) {
            for (int cc = start_ind; cc < (int) run_time.size(); cc++) {
              avg_duration += run_time[cc];
            }
          }
	  avg_duration /= avgsteps ;

          int rem_counts = last_counter - counter;
          int rem_duration_ms = (rem_counts * avg_duration);

	  // now time
          time_t my_time = time(NULL);
          char timestring[80];
          strftime(timestring, 80, "%F-%T", localtime(&my_time));

          std::cout
              << left << std::setw(column_w) << t 
	      << left << std::setw(column_w) << counter 
	      << left << std::setw(column_w+5) << (double)duration.count() / 1000
	      << left << std::setw(column_w) << format_ms(rem_duration_ms)
              << left << std::setw(column_w) << pw_comp_count
              << left << std::setw(column_w) << timestring << std::endl;

	  // store
          run_time.push_back(duration.count());
          t_ind.push_back(t);
	  pw_comp_count_vec.push_back(pw_comp_count);
	  pw_comp_count = 0;

          // update
          ++counter;
        }
      }
    }
  }

   //save runtime to file
   if (rank == 0) {
       string runtime_filename = "output/hdf5/run_time.h5";
       H5::H5File rt_fp(runtime_filename.c_str(), H5F_ACC_TRUNC);
       store_col<double>(rt_fp, "run_time", run_time);
       store_col<double>(rt_fp, "t_ind", t_ind);
       store_col<unsigned>(rt_fp, "pairwise_computations", pw_comp_count_vec);
       rt_fp.close();
   }
};

#endif /* ifndef TIMELOOP_H */
