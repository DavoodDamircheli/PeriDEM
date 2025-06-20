#ifndef CONTACT_H
#define CONTACT_H

#include "Eigen/Dense"
#include "read/read_config.h"
#include "particle/particle2.h"
#include <vector>
using namespace Eigen;
using namespace std;

// Segment mass calculation for 2D/3D based on radius and distance
template <unsigned dim>
vector<double> segment_mass(double R, double d) {
    if (abs(d) >= R) {
        cerr << "Error: d is greater than or equal to R" << endl;
        return {};
    }

    double area, I;
    if (dim == 2) {
        area = M_PI * pow(R, 2) / 2 - pow(R, 2) * asin(d / R) - sqrt(R * R - d * d) * d;
        I = 2 / 3 * pow((R * R - d * d), 1.5);
    } else {
        area = M_PI / 3 * pow((R - d), 2) * (2 * R - d);
        I = M_PI / 4 * pow(pow(R, 2) - pow(d, 2), 2);
    }

    return { I / area, area };  // Return centroid and area
}

// Vortex force field for a given position
template <unsigned dim>
Matrix<double, 1, dim> vortexfield(Matrix<double, 1, dim> pos, bool normalized, double scaling, double angle, unsigned time) {
    double freq = 6000;
    double slope_factor = (0.5 * sin(M_PI * float(time) / freq) >= 0) ? 1.0 : -1.0;

    Matrix<double, 1, dim> v;
    v(0) = cos(angle) * pos(0) - sin(angle) * pos(1);
    v(1) = sin(angle) * pos(0) + cos(angle) * pos(1);

    if (normalized) {
        v /= pos.norm();
    }
    return v * scaling * slope_factor;
}

// Shear force field
template <unsigned dim>
Matrix<double, 1, dim> shearfield(Matrix<double, 1, dim> pos, double scaling) {
    Matrix<double, 1, dim> v;
    v(0) = scaling * pos(1);
    v(1) = 0;
    return v;
}

// Contact class for handling contact properties
class Contact {
public:
    double contact_rad = 0;
    bool allow_contact = true;
    bool allow_friction = true;
    bool allow_damping = true;
    double normal_stiffness = 0, friction_coefficient = 0, damping_ratio = 0;
    bool self_contact = false;
    double self_contact_rad = 0;
    bool enable_cohesion = false;
    double cohesion_scaling = 1.0;

    Contact() = default;
    Contact(bool c, bool f, bool d) : allow_contact(c), allow_friction(f), allow_damping(d) {}

    void apply_config(const ConfigVal& CFGV) {
        allow_damping = CFGV.allow_damping;
        allow_friction = CFGV.allow_friction;
        normal_stiffness = (CFGV.normal_stiffness != -1) ? CFGV.normal_stiffness : normal_stiffness;
        friction_coefficient = (CFGV.friction_coefficient != -1) ? CFGV.friction_coefficient : friction_coefficient;
        damping_ratio = (CFGV.damping_ratio != -1) ? CFGV.damping_ratio : damping_ratio;
        self_contact = CFGV.self_contact;
        self_contact_rad = (CFGV.self_contact_rad != -1) ? CFGV.self_contact_rad : self_contact_rad;
        enable_cohesion = CFGV.enable_cohesion;
        cohesion_scaling = CFGV.cohesion_scaling;
    }
};

// Rectangular wall class template for handling boundaries
template <unsigned dim>
class RectWall {
public:
    bool allow_wall = false;
    double left = 0, right = 0, top = 0, bottom = 0;
    double x_min = 0, y_min = 0, z_min = 0, x_max = 0, y_max = 0, z_max = 0;
    Matrix<double, 1, dim> reaction_top, reaction_bottom, reaction_left, reaction_right;
    double speed_left = 0, speed_right = 0, speed_top = 0, speed_bottom = 0;

    RectWall(bool allow) : allow_wall(allow) {}

    vector<double> lrtb() const {
        return (dim == 2) ? vector<double>{left, right, top, bottom}
                          : vector<double>{x_min, y_min, z_min, x_max, y_max, z_max};
    }

    void set_lrtb(const vector<double>& v) {
        if (dim == 2) {
            left = v[0]; right = v[1]; top = v[2]; bottom = v[3];
        } else {
            x_min = v[0]; y_min = v[1]; z_min = v[2]; x_max = v[3]; y_max = v[4]; z_max = v[5];
        }
    }

    void update_boundary(double dt) {
        if (dim == 2) {
            left += speed_left * dt; right += speed_right * dt;
            top += speed_top * dt; bottom += speed_bottom * dt;
        } else {
            x_min += speed_left * dt; y_min += speed_right * dt;
            z_min += speed_top * dt; x_max += speed_bottom * dt;
        }
    }
};

// Update the force due to wall contact for a particle
template <unsigned dim>
void wall_side_force_update(double dist, const Contact& CN, unsigned dir, 
                            const Matrix<double, 1, dim>& vel, double vol, double rho, 
                            Matrix<double, 1, dim>& force_update, double dt, double wall_speed, 
                            Matrix<double, 1, dim>& wall_reaction) {
    if (abs(dist) <= CN.contact_rad) {
        auto ca = segment_mass<dim>(CN.contact_rad, abs(dist));
        auto contact_rad_contrib = (CN.contact_rad - ca[0]);
        double this_contact_force = CN.normal_stiffness * contact_rad_contrib * ca[1];
        double c_sign = (dist > 0) ? 1 : -1;

        force_update(dir) += c_sign * this_contact_force;
        wall_reaction(dir) -= vol * c_sign * this_contact_force;

        auto rel_vel = vel;
        rel_vel(dir) -= wall_speed;

        if (CN.allow_friction) {
            unsigned dir_tang = (dir + 1) % dim;
            if (abs(rel_vel(dir_tang)) > 0) {
                double friction_force = (-CN.friction_coefficient) * abs(this_contact_force) * (rel_vel(dir_tang) > 0 ? 1 : -1);
                force_update(dir_tang) += friction_force;
                wall_reaction(dir_tang) -= vol * friction_force;
            }
        }

        if (CN.allow_damping) {
            double damping_force = (-2 * CN.damping_ratio * sqrt(CN.normal_stiffness * rho) * sqrt(abs(ca[1]))) * rel_vel(dir);
            force_update(dir) += damping_force;
            wall_reaction(dir) -= vol * damping_force;
        }
    }
}

// Get the contact force between two particles
template <unsigned dim>
vector<Matrix<double, 1, dim>> get_contact_force(const ParticleN<dim>& P_C, const ParticleN<dim>& P_N, const Contact& CN) {
    vector<Matrix<double, 1, dim>> contact_forces(P_C.nnodes, Matrix<double, 1, dim>::Zero());
    for (unsigned i = 0; i < P_C.nnodes; ++i) {
        for (unsigned j = 0; j < P_N.nnodes; ++j) {
            auto dir = P_N.CurrPos[j] - P_C.CurrPos[i];
            if (dir.norm() <= CN.contact_rad) {
                double contact_force = (-CN.normal_stiffness * (CN.contact_rad - dir.norm()) * P_N.vol[j]) * (dir / dir.norm());
                contact_forces[i] += contact_force;
            }
        }
    }
    return contact_forces;
}

#endif /* CONTACT_H */

