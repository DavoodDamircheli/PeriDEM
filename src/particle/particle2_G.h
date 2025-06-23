#ifndef PARTICLE_2_H
#define PARTICLE_2_H

#include <iostream>
#include <vector>
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;

template <unsigned dim>
class ParticleN {
public:
    unsigned nnodes;
    vector<Matrix<double, 1, dim>> pos, disp, CurrPos, vel, acc, force, extforce;
    vector<Matrix<double, 1, dim>> prescribed_velocity, applied_force_density;
    vector<double> vol;
    vector<unsigned> boundary_nodes, clamped_nodes;
    vector<vector<unsigned>> NbdArr;  // Neighbor array
    vector<unsigned> total_neighbors;
    vector<vector<Matrix<double, 1, dim>>> xi;  // Bond vectors
    vector<vector<double>> xi_norm, stretch;    // Bond norms and stretch values

    double delta, rho, cnot, snot;
    bool break_bonds = false;
    int breakable = 1, movable = 1, stoppable = 1;

    ParticleN(unsigned N) : nnodes(N) {
        pos.resize(N); disp.resize(N); CurrPos.resize(N); vel.resize(N); acc.resize(N);
        force.resize(N); extforce.resize(N); vol.resize(N); 
        applied_force_density.resize(N); prescribed_velocity.resize(N);
        NbdArr.resize(N); xi.resize(N); xi_norm.resize(N); stretch.resize(N); 
        total_neighbors.resize(N);
    }

    // Compute the mean current position of all nodes
    Matrix<double, 1, dim> mean_CurrPos() const {
        Matrix<double, 1, dim> mean = Matrix<double, 1, dim>::Zero();
        for (unsigned i = 0; i < nnodes; i++) mean += (pos[i] + disp[i]);
        return mean / nnodes;
    }

    // Computes peridynamic force for each node
    vector<Matrix<double, 1, dim>> get_peridynamic_force() {
        vector<Matrix<double, 1, dim>> total_force(nnodes, Matrix<double, 1, dim>::Zero());
        for (unsigned i = 0; i < nnodes; ++i) {
            for (unsigned j = 0; j < total_neighbors[i]; ++j) {
                unsigned idx_nbr = NbdArr[i][j];
                auto eta_p_xi = (disp[idx_nbr] - disp[i]) + xi[i][j];
                double eta_p_xi_norm = eta_p_xi.norm();
                auto unit_dir = eta_p_xi / eta_p_xi_norm;

                double str = (eta_p_xi_norm - xi_norm[i][j]) / xi_norm[i][j];
                total_force[i] += (cnot * str * vol[idx_nbr]) * unit_dir;
                stretch[i][j] = str;
            }
        }
        return total_force;
    }

    // Remove bonds that exceed the stretch limit (bond breaking)
    void remove_bonds() {
        if (break_bonds) {
            for (unsigned i = 0; i < nnodes; ++i) {
                auto& stretch_i = stretch[i];
                auto& NbdArr_i = NbdArr[i];
                auto& xi_i = xi[i];
                auto& xi_norm_i = xi_norm[i];

                for (unsigned j = 0; j < stretch_i.size();) {
                    if (stretch_i[j] > snot) {
                        stretch_i.erase(stretch_i.begin() + j);
                        NbdArr_i.erase(NbdArr_i.begin() + j);
                        xi_i.erase(xi_i.begin() + j);
                        xi_norm_i.erase(xi_norm_i.begin() + j);
                        --total_neighbors[i];
                    } else {
                        ++j;
                    }
                }
            }
        }
    }

    // Initialize bond information for neighbors
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
                xi_norm[i][j] = xi[i][j].norm();
            }
        }
    }

    // Print particle information
    void print() const {
        cout << "delta: " << delta << " rho: " << rho << " cnot: " << cnot << " snot: " << snot << endl;
    }
};

// Overload << operator for particle output
template <unsigned dim>
ostream& operator<<(ostream& o, const ParticleN<dim>& P) {
    o << "[Particle: " << P.nnodes << " nodes]";
    return o;
}

#endif /* ifndef PARTICLE_2_H */

