#ifndef PARTICLE_2_H
#define PARTICLE_2_H

#include <Eigen/Dense>
#include <cmath>
#include <float.h>
#include <iostream>
#include <vector>

// #include "../../lib/eigen-3.2.10/Eigen/Dense"
using namespace Eigen;

#include "compat/overloads.h"

using namespace std;

#include <cstdlib>

// particle class
template <unsigned dim> class ParticleN {
public:
  unsigned nnodes;
  vector<Matrix<double, 1, dim>> pos, disp, CurrPos, vel, acc, force, extforce;
  vector<Matrix<double, 1, dim>> disp_old;

  // vector<Matrix<double, 1, dim>> pos,  CurrPos, vel, acc, force, extforce;
  vector<Matrix<double, 1, dim>> prescribed_velocity, applied_force_density;
  vector<double> vol;
  vector<unsigned> boundary_nodes;
  vector<unsigned> clamped_nodes;
  vector<vector<unsigned>> NbdArr;

  vector<unsigned> total_neighbors;

  // vector<vector<double>> xi_1;
  // vector<vector<double>> xi_2;

  vector<vector<Matrix<double, 1, dim>>> xi;
  vector<vector<double>> xi_norm;
  vector<vector<double>> stretch;

  double delta, rho, cnot, snot; // delta = peridynamic horizon
  double rLp, rLm, delta_y, s_tension_crit,
      s_compression_crit;     // rLp: tensile yield stretch, rLm: compressive
                              // yield stretch (>0), delta_y: smoothing
  bool use_yield_plateau_law; // switch between linear and yield-plateau law
  double rSp;                 // tension softening start  (rS^+)
  double rFp;                 // tension fracture stretch (rF^+)
  double rSm;                 // compression softening start magnitude (rS^-)
  double rFm;                 // compression fracture stretch magnitude (rF^-)

  int breakable, movable, stoppable;

  bool break_bonds;

  // torque
  unsigned torque_axis;
  double torque_val;

  ParticleN();

  ParticleN(unsigned N) {
    pos.resize(N);
    vol.resize(N);

    disp.resize(N);
    disp_old.resize(N);

    vel.resize(N);
    acc.resize(N);
    force.resize(N);
    extforce.resize(N);
    applied_force_density.resize(N);
    prescribed_velocity.resize(N);

    NbdArr.resize(N);
    stretch.resize(N);
    // boundary_nodes.resize(N);
    nnodes = N;

    xi.resize(N);
    xi_norm.resize(N);

    total_neighbors.resize(N);

    break_bonds = 0;

    breakable = 1;
    movable = 1;
    stoppable = 1;

    use_yield_plateau_law = false; // default: keep current linear model
    rLp = 0.0;
    rLm = 0.0;
    delta_y = 0.0;
    rSp = 0.0; // tension softening start  (rS^+)
    rFp = 0.0; // tension fracture stretch (rF^+)
    rSm = 0.0; // compression softening start magnitude (rS^-)
    rFm = 0.0; // compression fracture stretch magnitude (rF^-)
  };

  // return the mean current position
  Matrix<double, 1, dim> mean_CurrPos() {
    Matrix<double, 1, dim> mean;
    mean.setZero();
    for (unsigned i = 0; i < nnodes; i++) {
      mean += (pos[i] + disp[i]);
    }
    mean /= nnodes;
    return mean;
  };

  // computes peridynamic force
  // and updates public var: stretch
  vector<Matrix<double, 1, dim>> get_peridynamic_force() {
    vector<Matrix<double, 1, dim>> tot_peri_force;
    tot_peri_force.resize(nnodes);

    for (unsigned i = 0; i < nnodes; ++i) {

      Matrix<double, 1, dim> v = Matrix<double, 1, dim>::Zero();

      for (unsigned j = 0; j < total_neighbors[i]; ++j) {
        unsigned idx_nbr = NbdArr[i][j];

        auto eta_p_xi = (disp[idx_nbr] - disp[i]) + xi[i][j];
        // double eta_p_xi_norm = my_norm<dim>(eta_p_xi);
        double eta_p_xi_norm = eta_p_xi.norm();
        auto unit_dir = eta_p_xi / eta_p_xi_norm;

        double xi_norm_here = xi_norm[i][j];

        double diff_norm = (eta_p_xi_norm - xi_norm_here);
        double str = diff_norm / xi_norm_here;
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------

        double scalar_force = 0.0;

        if (!use_yield_plateau_law) {
          //  current behavior (linear)
          scalar_force = cnot * str;
        } else {
          // new yield-plateau behavior (different rLp/rLm, slope -> 0 after
          // yield)
          scalar_force = bond_response_yield_plateau(str);
        }
        auto v_more = (scalar_force * vol[idx_nbr]) * unit_dir;

        //--------------------------------------------------------------------
        //--------------------------------------------------------------------

        v += v_more;

        stretch[i][j] = str;
      }

      tot_peri_force[i] = v;
    }

    return tot_peri_force;
  };
  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  inline double smoothstep01(double t) const {
    // C1 smoothstep from 1 at t=0 to 0 at t=1 (zero slope at both ends)
    if (t <= 0.0)
      return 1.0;
    if (t >= 1.0)
      return 0.0;
    return 1.0 - 3.0 * t * t + 2.0 * t * t * t;
  }

  inline double bond_response_yield_plateau(double r) const {

    // -----------------------
    // Safety / hard break
    // -----------------------
    if (r >= rFp)
      return 0.0;
    if (r <= -rFm)
      return 0.0;

    // -----------------------
    // Elastic region
    // -----------------------
    if (r >= -rLm && r <= rLp) {
      return cnot * r;
    }

    // ============================================================
    // TENSION side: r > rLp
    // ============================================================
    if (r > rLp) {
      // yield->plateau (your original saturating exponential)
      const double x = r - rLp;
      const double F =
          cnot * rLp + cnot * delta_y * (1.0 - std::exp(-x / delta_y));

      // no softening yet
      if (r <= rSp)
        return F;

      // softening region: rSp < r < rFp
      const double xS = rSp - rLp;
      const double FSp =
          cnot * rLp + cnot * delta_y * (1.0 - std::exp(-xS / delta_y));

      const double t = (r - rSp) / (rFp - rSp); // in (0,1)
      return FSp * smoothstep01(t);
    }

    // ============================================================
    // COMPRESSION side: r < -rLm
    // ============================================================
    {
      // use magnitude a = -r
      const double a = -r;
      const double x = a - rLm;
      const double F =
          -cnot * rLm - cnot * delta_y * (1.0 - std::exp(-x / delta_y));

      // no softening yet (note: r >= -rSm means magnitude a <= rSm)
      if (a <= rSm)
        return F;

      // softening: rSm < a < rFm  <=>  -rFm < r < -rSm
      const double xS = rSm - rLm;
      const double FSm =
          -cnot * rLm - cnot * delta_y * (1.0 - std::exp(-xS / delta_y));

      const double t = (a - rSm) / (rFm - rSm);
      return FSm * smoothstep01(t);
    }
  }

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  //--------------------------------------------------------------------

  void remove_bonds() {
    if (!break_bonds)
      return;

    for (unsigned i = 0; i < nnodes; ++i) {

      auto j = stretch[i].begin();
      auto j_NbdArr = NbdArr[i].begin();
      auto j_xi = xi[i].begin();
      auto j_xi_norm = xi_norm[i].begin();

      while (j != stretch[i].end()) {

        const double s = *j;

        const bool break_tension = (s > s_tension_crit);
        const bool break_compression = (s < -s_compression_crit);

        if (break_tension || break_compression) {
          j = stretch[i].erase(j);
          j_NbdArr = NbdArr[i].erase(j_NbdArr);
          j_xi = xi[i].erase(j_xi);
          j_xi_norm = xi_norm[i].erase(j_xi_norm);

          --total_neighbors[i];
        } else {
          ++j;
          ++j_NbdArr;
          ++j_xi;
          ++j_xi_norm;
        }
      }
    }
  }

  //--------------------------------------------------------------------
  //--------------------------------------------------------------------
  //--------------------------------------------------------------------

  // Populate xi related information for the neighbors
  void gen_xi() {
    for (unsigned i = 0; i < nnodes; ++i) {
      unsigned ss = NbdArr[i].size();

      total_neighbors[i] = ss;

      xi_norm[i].resize(ss);
      xi[i].resize(ss);
      stretch[i].resize(ss);

      for (unsigned j = 0; j < ss; ++j) {
        unsigned idx_nbr = NbdArr[i][j];
        xi[i][j] = pos[idx_nbr] - pos[i];
        xi_norm[i][j] = (xi[i][j]).norm();
      }
    }
  };

  void print() {
    std::cout << "delta: " << delta << std::endl;
    std::cout << "rho: " << rho << std::endl;
    std::cout << "cnot: " << cnot << std::endl;
    std::cout << "snot: " << snot << std::endl;
    std::cout << "use_yield_plateau_law: " << use_yield_plateau_law
              << std::endl;
    std::cout << "rLp (tension yield): " << rLp << std::endl;
    std::cout << "rLm (compression yield): " << rLm << std::endl;
    std::cout << "delta_y (smoothing): " << delta_y << std::endl;
  };

private:
  /* data */
};

// print a particle
template <unsigned dim> ostream &operator<<(ostream &o, ParticleN<dim> P) {
  o << "[Particle: " << P.nnodes << " nodes]";
  return o;
};

#endif /* ifndef PARTICLE_2_H */
