#ifndef CONTACT_H
#define CONTACT_H

#include "Eigen/Dense"
//#include "../../lib/eigen-3.2.10/Eigen/Dense"
#include <vector>
using namespace Eigen;

#include "read/read_config.h"

#include "particle/particle2.h" //#include "particle/nbdarr.h" #include "compat/overloads.h"
using namespace std;

template <unsigned dim> vector<double> segment_mass(double R, double d) {

  if (abs(d) >= R) {
    std::cout << "Error: d is bigger or equal to R" << std::endl;
  }

  double Area, I;

  if (dim == 2) {
    Area = M_PI * pow(R, 2) / 2 - pow(R, 2) * asin(d / R) -
           sqrt(R * R - d * d) * d;
    I = 2 / 3 * pow((R * R - d * d), 1.5);
  } else {
    // 3D
    Area = M_PI / 3 * pow((R - d), 2) * (2 * R - d);
    I = M_PI / 4 * pow(pow(R, 2) - pow(d, 2), 2);
  }

  double centroid = I / Area;

  vector<double> ca;
  ca.push_back(centroid);
  ca.push_back(Area);

  return ca;
};

// Outputs a vortex forcefield for an input position
template <unsigned dim>
Matrix<double, 1, dim> vortexfield(Matrix<double, 1, dim> pos, bool normalized, double scaling, double angle, unsigned time)
{
    double pos_norm = pos.norm();
    // avoid divide by zero
    if (pos_norm==0) {
	       pos_norm = 1;
    }
    Matrix<double, 1, dim> v;

    //v(0) = -pos(1);
    //v(1) = pos(0);

    // rotation matrix
    // [cos, -sin; sin, cos]
    // [0, -1; 1, 0]
    //double angle = M_PI/2;
    //double angle = M_PI/2 + M_PI/4;

    double freq = 6000;

    double slope_factor; 
 
    if ( 0.5 * sin(M_PI*float(time)/freq) >= 0 )   //sinusoidal function to regulate current fluctuation using an arbitrary frequency
    {
      slope_factor = 1.0;  //Positive
    }
    else  
    {
      slope_factor = -1.0; //Negative
    }
    

    v(0) = cos(angle) * pos(0) - sin(angle) * pos(1);
    v(1) = sin(angle) * pos(0) + cos(angle) * pos(1);

    //v(0) = 0.001;
    //v(1) = 0.001;


    if (normalized) {
	       v /= pos_norm;
    }

    v *= scaling;

    v *= slope_factor; //Sinusoidal factor inclusion

    return v;
};

// Outputs a shear field
template <unsigned dim>
Matrix<double, 1, dim> shearfield(Matrix<double, 1, dim> pos, bool normalized, double scaling, double angle, unsigned time)
{
    Matrix<double, 1, dim> v;
    v(0) = scaling * pos(1);
    v(1) = 0;
    return v;
};


class Contact {
public:
  double contact_rad;
  bool allow_contact, allow_friction, allow_damping;
  double normal_stiffness, friction_coefficient, damping_ratio;
  // use only nonlocal boundary nodes for contact detection
  // Useful when there is no fracture
  bool nl_bdry_only = 0;

  // self-contact
  bool self_contact;
  double self_contact_rad;

  bool enable_cohesion=0;
  double cohesion_scaling=1;

  bool compute_wall_reaction = 0;


  Contact() {
    allow_contact = 1;
    allow_friction = 1;
    allow_damping = 1;
  };
  Contact(bool c, bool f, bool d) {
    allow_contact = c;
    allow_friction = f;
    allow_damping = d;
  };
  // virtual ~contact ();
  void print() {
    std::cout << "allow_contact: " << allow_contact
              << " allow_damping:" << allow_damping
              << " allow_friction: " << allow_friction << std::endl;
    std::cout << "contact radius: " << contact_rad << std::endl;
    std::cout << "normal_stiffness: " << normal_stiffness << std::endl;
    std::cout << "friction_coefficient: " << friction_coefficient << std::endl;
    std::cout << "damping_ratio: " << damping_ratio << std::endl;
    std::cout << "nl_bdry_only: " << nl_bdry_only << std::endl;

    std::cout << "\t\tContact\t\tFriction\tDamping" << std::endl;
    std::cout << "allow\t\t" << allow_contact << "\t\t" << allow_friction
              << "\t\t" << allow_damping << std::endl;
    std::cout << "const\t\t" << normal_stiffness << "\t" << friction_coefficient
              << "\t\t" << damping_ratio << std::endl;
  };

void apply_config(ConfigVal CFGV) {
    allow_damping  = CFGV.allow_damping;
    allow_friction = CFGV.allow_friction;
    nl_bdry_only   = CFGV.nl_bdry_only;

    // Override if specified in config file
    if (CFGV.damping_ratio != (-1)) {
	damping_ratio = CFGV.damping_ratio;
    }
    if (CFGV.friction_coefficient != (-1)) {
	friction_coefficient = CFGV.friction_coefficient;
    }
    if (CFGV.normal_stiffness != (-1)) {
	normal_stiffness = CFGV.normal_stiffness;
    }

    // self-contact
    self_contact = CFGV.self_contact;
    if (CFGV.self_contact_rad != (-1)) {
	self_contact_rad = CFGV.self_contact_rad;
    } else {
	self_contact_rad = CFGV.self_contact_rad;
    }

    enable_cohesion = CFGV.enable_cohesion;
    cohesion_scaling = CFGV.cohesion_scaling;

    compute_wall_reaction = CFGV.compute_wall_reaction;
};

private:
  /* data */
};

template <unsigned dim> class RectWall {
public:
  bool allow_wall;
  double left, right, top, bottom;

  // for 3d
  double x_min, y_min, z_min, x_max, y_max, z_max;

  // reaction forces
  Matrix<double, 1, dim> reaction_top   ; // = Matrix<double, 1, dim>::Zero();
  Matrix<double, 1, dim> reaction_bottom; // = Matrix<double, 1, dim>::Zero();
  Matrix<double, 1, dim> reaction_left  ; // = Matrix<double, 1, dim>::Zero();
  Matrix<double, 1, dim> reaction_right ; // = Matrix<double, 1, dim>::Zero();

  Matrix<double, 1, dim> reaction_x_min ; // = Matrix<double, 1, dim>::Zero();
  Matrix<double, 1, dim> reaction_y_min ; // = Matrix<double, 1, dim>::Zero();
  Matrix<double, 1, dim> reaction_z_min ; // = Matrix<double, 1, dim>::Zero();
  Matrix<double, 1, dim> reaction_x_max ; // = Matrix<double, 1, dim>::Zero();
  Matrix<double, 1, dim> reaction_y_max ; // = Matrix<double, 1, dim>::Zero();
  Matrix<double, 1, dim> reaction_z_max ; // = Matrix<double, 1, dim>::Zero();

  // wall boundary moving
  double speed_left = 0;
  double speed_right = 0;
  double speed_top = 0;
  double speed_bottom = 0;

  double speed_x_min = 0;
  double speed_y_min = 0;
  double speed_z_min = 0;
  double speed_x_max = 0;
  double speed_y_max = 0;
  double speed_z_max = 0;

  // cylindrical wall
    bool wall_cyl = 0;
    double wall_cyl_cent_x = 0;
    double wall_cyl_cent_y = 0;
    double wall_cyl_rad = 1.5e-3;

  Matrix<double, 1, dim> reaction_r ; // = Matrix<double, 1, dim>::Zero();

  RectWall(bool allow) : allow_wall(allow){};

  vector<double> lrtb() {
    vector<double> v;
    if (dim == 2) {
      v.push_back(left);
      v.push_back(right);
      v.push_back(top);
      v.push_back(bottom);
    } else if (dim == 3) {
      v.push_back(x_min);
      v.push_back(y_min);
      v.push_back(z_min);
      v.push_back(x_max);
      v.push_back(y_max);
      v.push_back(z_max);
    }

    return v;
  };

  void set_lrtb(vector<double> v) {
    if (dim == 2) {
      left   = v[0];
      right  = v[1];
      top    = v[2];
      bottom = v[3];
    } else if (dim == 3) {
      x_min = v[0];
      y_min = v[1];
      z_min = v[2];
      x_max = v[3];
      y_max = v[4];
      z_max = v[5];
    }
  };

  vector<Matrix<double, 1, dim>> get_reaction() {
    vector<Matrix<double, 1, dim>> v;
    if (dim == 2) {
      v.push_back(reaction_left);
      v.push_back(reaction_right);
      v.push_back(reaction_top);
      v.push_back(reaction_bottom);
    } else if (dim == 3) {
      v.push_back(reaction_x_min);
      v.push_back(reaction_y_min);
      v.push_back(reaction_z_min);
      v.push_back(reaction_x_max);
      v.push_back(reaction_y_max);
      v.push_back(reaction_z_max);
    }

    return v;
  };

  void	setzero_reaction(){
      reaction_top = Matrix<double, 1, dim>::Zero();
      reaction_bottom = Matrix<double, 1, dim>::Zero();
      reaction_left = Matrix<double, 1, dim>::Zero();
      reaction_right = Matrix<double, 1, dim>::Zero();

      reaction_x_min = Matrix<double, 1, dim>::Zero();
      reaction_y_min = Matrix<double, 1, dim>::Zero();
      reaction_z_min = Matrix<double, 1, dim>::Zero();
      reaction_x_max = Matrix<double, 1, dim>::Zero();
      reaction_y_max = Matrix<double, 1, dim>::Zero();
      reaction_z_max = Matrix<double, 1, dim>::Zero();

      reaction_r = Matrix<double, 1, dim>::Zero();
  };

  // virtual ~RectWall ();
  void print() {
    std::cout << "Wall size (dim=" << dim << ") " << std::endl;
    if (dim == 2) {
      std::cout << " left: " << left;
      std::cout << " right: " << right;
      std::cout << " bottom: " << bottom;
      std::cout << " top: " << top;
    } else if (dim == 3) {
      std::cout << " x_min : " << x_min;
      std::cout << " y_min : " << y_min;
      std::cout << " z_min : " << z_min;
      std::cout << " x_max : " << x_max;
      std::cout << " y_max : " << y_max;
      std::cout << " z_max : " << z_max;
    }
    std::cout << std::endl;
  };

void update_boundary(double dt){
    if (dim ==2) {
	left += speed_left * dt;
	right += speed_right * dt;
	top += speed_top * dt;
	bottom += speed_bottom * dt;
    }
    else{
	x_min += speed_x_min * dt;
	y_min += speed_y_min * dt;
	z_min += speed_z_min * dt;
	x_max += speed_x_max * dt;
	y_max += speed_y_max * dt;
	z_max += speed_z_max * dt;
    }
};

void apply_config(ConfigVal CFGV) {
    // wall
    if (CFGV.wall_top != (-999)) {
	if (dim ==2) {
	    top = CFGV.wall_top;
	}
	else{
	    z_max = CFGV.wall_top;
	}
    }
    if (CFGV.wall_right != (-999)) {
	if (dim ==2) {
	    right = CFGV.wall_right;
	}
	else{
	    y_max = CFGV.wall_right;
	}
    }

    // Load wall dimension (if defined) and speed 
    if (dim==2) {
	if (CFGV.wall_left != (-999)) { left = CFGV.wall_left; }
	if (CFGV.wall_right != (-999)) { right = CFGV.wall_right; }
	if (CFGV.wall_top != (-999)) { top = CFGV.wall_top; }
	if (CFGV.wall_bottom != (-999)) { bottom = CFGV.wall_bottom; }

	speed_left = CFGV.speed_wall_left;
	speed_right = CFGV.speed_wall_right;
	speed_top = CFGV.speed_wall_top;
	speed_bottom = CFGV.speed_wall_bottom;
    }
    else{
	if (CFGV.wall_x_min != (-999)) { x_min = CFGV.wall_x_min; }
	if (CFGV.wall_y_min != (-999)) { y_min = CFGV.wall_y_min; }
	if (CFGV.wall_z_min != (-999)) { z_min = CFGV.wall_z_min; }
	if (CFGV.wall_x_max != (-999)) { x_max = CFGV.wall_x_max; }
	if (CFGV.wall_y_max != (-999)) { y_max = CFGV.wall_y_max; }
	if (CFGV.wall_z_max != (-999)) { z_max = CFGV.wall_z_max; }

	speed_x_min = CFGV.speed_wall_x_min;
	speed_y_min = CFGV.speed_wall_y_min;
	speed_z_min = CFGV.speed_wall_z_min;
	speed_x_max = CFGV.speed_wall_x_max;
	speed_y_max = CFGV.speed_wall_y_max;
	speed_z_max = CFGV.speed_wall_z_max;

	if (CFGV.wall_cyl) {
	wall_cyl = CFGV.wall_cyl;
	wall_cyl_cent_x = CFGV.wall_cyl_cent_x;
	wall_cyl_cent_y = CFGV.wall_cyl_cent_y;
	wall_cyl_rad = CFGV.wall_cyl_rad;
	    
	}
    }

};


private:
};

// updates the total force by an wall edge and the signed distance from wall
// edge dir: 0 for x, 1 for y, 2 for z (direction in which wall contact force
// will act)
template <unsigned dim>
// void wall_side_force_update(double dist, Contact CN, unsigned dir,
// Matrix<double, 1, dim> vel_i, double rho, Matrix<double, 1, dim>
// &f_toupdate_i)
// wall_cyl_side_force_update<dim>(dist_from_cyl_wall, CN, cyl_inward_normal_x, cyl_inward_normal_y, P_C.vel[i], P_C.vol[i], P_C.rho, giant_f[i+i_starting_index], dt, Wall.speed_r, Wall.reaction_r);
void wall_cyl_side_force_update(double dist, Contact CN, double cyl_inward_normal_x, double cyl_inward_normal_y,
                            Matrix<double, 1, dim> vel_i, double vol_i, double rho,
                            Matrix<double, 1, dim> &f_toupdate_i, double dt, double wall_side_speed, Matrix<double, 1, dim> &wall_reaction_force) {

  if (abs(dist) <= CN.contact_rad) {
    // auto ca = segment_mass(CN.contact_rad, abs(dist));
    auto ca = segment_mass<dim>(CN.contact_rad, abs(dist));
    // centroid = ca[0];
    // Area = ca[1];

    // std::cout << "here" << std::endl;
    auto contact_rad_contrib = (CN.contact_rad - ca[0]);
    auto positive_contact_rad_contrib =
        (contact_rad_contrib > 0 ? contact_rad_contrib : 0);

    auto this_contact_force =
        (CN.normal_stiffness * positive_contact_rad_contrib * ca[1]);

    // y-value of contact force, for wall_bottom
    // combined_contact_force[0](dir) += this_contact_force;
    // double c_sign = (dist > 0 ? 1 : -1);
    // double c_sign = (dist > 0 ? 1 : -1);
    double c_sign = 1;

    f_toupdate_i(0) += (c_sign * this_contact_force * cyl_inward_normal_x);
    f_toupdate_i(1) += (c_sign * this_contact_force * cyl_inward_normal_y);
    // get the Force (not density) on the wall point w:
    // F(w) = \sum_{x} K_n(R_c - |d(x)|) V(w)V(x), where $x$ is particle node
    // Here, this_contact_force = K_n(R_c - |d(x)|) V(w). So, we just multiply V(x)
    wall_reaction_force(0) -= (vol_i * c_sign * this_contact_force * cyl_inward_normal_x);
    wall_reaction_force(1) -= (vol_i * c_sign * this_contact_force * cyl_inward_normal_y);

    // compute the relative velocity from the speed of wall boundary
    auto rel_vel_i = vel_i;
    // rel_vel_i[dir] -= Wall.wall_side_speed;
    
    double rel_vel_normal_proj = rel_vel_i[0] * cyl_inward_normal_x + rel_vel_i[1] * cyl_inward_normal_y;
    // rel_vel_normal_proj -= wall_side_speed;

    if (CN.allow_friction) {
      // Friction force
      //
      if (dim == 2) {
	  // not implemented
        // // tangential axis relative to dir, i.e. 0 if 1, 1 if 0
        // auto dir_tang = (dir + 1) % 2;
        //
        // // What if the tangential velocity component is zero?
        // if (abs(rel_vel_i(dir_tang)) > 0) {
        //   double velocity_tangential_sign = (rel_vel_i(dir_tang) > 0 ? 1 : -1);
        //   // Debug
        //   double friction_force = (-CN.friction_coefficient) *
        //                           abs(this_contact_force) *
        //                           velocity_tangential_sign;
	//   // modification of friction for small velocity
        //   if (abs(rel_vel_i(dir_tang)) <
        //       CN.friction_coefficient * abs(this_contact_force) * dt / rho) {
        //     // std::cout << "below threshold" << std::endl;
        //     // Correct version
        //     friction_force = -rel_vel_i(dir_tang) * rho / dt;
        //
        //   }
        //   // friction_force = (- CN.friction_coefficient) *
        //   // abs(this_contact_force) * velocity_tangential_sign;
        //
        //   // x-value of the contact force
        //   // combined_contact_force[i](0) += friction_force;
        //   f_toupdate_i((dir + 1) % 2) += friction_force;
	//   // get the force density
        //   wall_reaction_force((dir + 1) % 2) -= (vol_i * friction_force);
        // }
      } else {
	  // Not implemented
        // // dim 3
        // // two components of perpendicular to wall edge direction
        // unsigned dir_tang_1 = (dir + 1) % 3;
        // unsigned dir_tang_2 = (dir + 2) % 3;
        //
	// // projection of velocity to the tangential direction (i.e. normal to wall)
        // auto vel_proj_1 = rel_vel_i(dir_tang_1);
        // auto vel_proj_2 = rel_vel_i(dir_tang_2);
        // auto vel_proj_norm = sqrt(pow(vel_proj_1, 2) + pow(vel_proj_2, 2));
        //
        // // What if the norm is zero?
        // if (vel_proj_norm > 0) {
        //   auto unit_comp_1 = vel_proj_1 / vel_proj_norm;
        //   auto unit_comp_2 = vel_proj_2 / vel_proj_norm;
        //
        //   double friction_force_1 = (-CN.friction_coefficient) *
        //                             abs(this_contact_force) * unit_comp_1;
        //   double friction_force_2 = (-CN.friction_coefficient) *
        //                             abs(this_contact_force) * unit_comp_2;
        //
        //     //f_friction = -CN.friction_coefficient * contact_force_norm *
        //                  //rel_v_tangential_unit;
	//  //// Correction for small velocity
	//   if (vel_proj_norm < CN.friction_coefficient * abs(this_contact_force) * dt /rho) {
	//       friction_force_1 = -vel_proj_norm/rho/dt * unit_comp_1;
	//       friction_force_2 = -vel_proj_norm/rho/dt * unit_comp_2;
	//   }
        //
        //   // x-value of the contact force
        //   // combined_contact_force[i](0) += friction_force;
        //   f_toupdate_i(dir_tang_1) += friction_force_1;
        //   f_toupdate_i(dir_tang_2) += friction_force_2;
        //
	//   // multiply by volume to get force from force density
        //   wall_reaction_force(dir_tang_1) -= (vol_i * friction_force_1);
        //   wall_reaction_force(dir_tang_2) -= (vol_i * friction_force_2);
        // }
      }
    }

    if (CN.allow_damping) {
      // Damping force
      auto damping_coefficient =
	  2 * CN.damping_ratio * sqrt(CN.normal_stiffness * rho) *
	  sqrt(abs(ca[1])); // area is occasionally -ve for very small values
			    //
      // double damping_force = (-damping_coefficient) * rel_vel_i(dir);
      double damping_force = (-damping_coefficient) * rel_vel_normal_proj;
      // y-value of the contact force, for wall_bottom
      // combined_contact_force[i](1) += total_damping_force;
      //
      // f_toupdate_i(dir) += damping_force;
      f_toupdate_i(0) += damping_force * cyl_inward_normal_x;
      f_toupdate_i(1) += damping_force * cyl_inward_normal_y;

    // multiply by volume to get force from force density
      // wall_reaction_force(dir) -= (vol_i * damping_force);
      wall_reaction_force(0) -= (vol_i * damping_force * cyl_inward_normal_x) ;
      wall_reaction_force(1) -= (vol_i * damping_force * cyl_inward_normal_y) ;
    }
  }
};

// updates the total force by an wall edge and the signed distance from wall
// edge dir: 0 for x, 1 for y, 2 for z (direction in which wall contact force
// will act)
template <unsigned dim>
// void wall_side_force_update(double dist, Contact CN, unsigned dir,
// Matrix<double, 1, dim> vel_i, double rho, Matrix<double, 1, dim>
// &f_toupdate_i)
void wall_side_force_update(double dist, Contact CN, unsigned dir,
                            Matrix<double, 1, dim> vel_i, double vol_i, double rho,
                            Matrix<double, 1, dim> &f_toupdate_i, double dt, double wall_side_speed, Matrix<double, 1, dim> &wall_reaction_force) {

  if (abs(dist) <= CN.contact_rad) {
    // auto ca = segment_mass(CN.contact_rad, abs(dist));
    auto ca = segment_mass<dim>(CN.contact_rad, abs(dist));
    // centroid = ca[0];
    // Area = ca[1];

    // std::cout << "here" << std::endl;
    auto contact_rad_contrib = (CN.contact_rad - ca[0]);
    auto positive_contact_rad_contrib =
        (contact_rad_contrib > 0 ? contact_rad_contrib : 0);

    auto this_contact_force =
        (CN.normal_stiffness * positive_contact_rad_contrib * ca[1]);

    // y-value of contact force, for wall_bottom
    // combined_contact_force[0](dir) += this_contact_force;
    double c_sign = (dist > 0 ? 1 : -1);
    f_toupdate_i(dir) += (c_sign * this_contact_force);
    // get the Force (not density) on the wall point w:
    // F(w) = \sum_{x} K_n(R_c - |d(x)|) V(w)V(x), where $x$ is particle node
    // Here, this_contact_force = K_n(R_c - |d(x)|) V(w). So, we just multiply V(x)
    wall_reaction_force(dir) -= (vol_i * c_sign * this_contact_force);

    // compute the relative velocity from the speed of wall boundary
    auto rel_vel_i = vel_i;
    rel_vel_i[dir] -= wall_side_speed;

    if (CN.allow_friction) {
      // Friction force
      //
      if (dim == 2) {
        // tangential axis relative to dir, i.e. 0 if 1, 1 if 0
        auto dir_tang = (dir + 1) % 2;

        // What if the tangential velocity component is zero?
        if (abs(rel_vel_i(dir_tang)) > 0) {
          double velocity_tangential_sign = (rel_vel_i(dir_tang) > 0 ? 1 : -1);
          // Debug
          double friction_force = (-CN.friction_coefficient) *
                                  abs(this_contact_force) *
                                  velocity_tangential_sign;
	  // modification of friction for small velocity
          if (abs(rel_vel_i(dir_tang)) <
              CN.friction_coefficient * abs(this_contact_force) * dt / rho) {
            // std::cout << "below threshold" << std::endl;
            // Correct version
            friction_force = -rel_vel_i(dir_tang) * rho / dt;

          }
          // friction_force = (- CN.friction_coefficient) *
          // abs(this_contact_force) * velocity_tangential_sign;

          // x-value of the contact force
          // combined_contact_force[i](0) += friction_force;
          f_toupdate_i((dir + 1) % 2) += friction_force;
	  // get the force density
          wall_reaction_force((dir + 1) % 2) -= (vol_i * friction_force);
        }
      } else {
        // dim 3
        // two components of perpendicular to wall edge direction
        unsigned dir_tang_1 = (dir + 1) % 3;
        unsigned dir_tang_2 = (dir + 2) % 3;

	// projection of velocity to the tangential direction (i.e. normal to wall)
        auto vel_proj_1 = rel_vel_i(dir_tang_1);
        auto vel_proj_2 = rel_vel_i(dir_tang_2);
        auto vel_proj_norm = sqrt(pow(vel_proj_1, 2) + pow(vel_proj_2, 2));

        // What if the norm is zero?
        if (vel_proj_norm > 0) {
          auto unit_comp_1 = vel_proj_1 / vel_proj_norm;
          auto unit_comp_2 = vel_proj_2 / vel_proj_norm;

          double friction_force_1 = (-CN.friction_coefficient) *
                                    abs(this_contact_force) * unit_comp_1;
          double friction_force_2 = (-CN.friction_coefficient) *
                                    abs(this_contact_force) * unit_comp_2;

            //f_friction = -CN.friction_coefficient * contact_force_norm *
                         //rel_v_tangential_unit;
	 //// Correction for small velocity
	  if (vel_proj_norm < CN.friction_coefficient * abs(this_contact_force) * dt /rho) {
	      friction_force_1 = -vel_proj_norm/rho/dt * unit_comp_1;
	      friction_force_2 = -vel_proj_norm/rho/dt * unit_comp_2;
	  }

          // x-value of the contact force
          // combined_contact_force[i](0) += friction_force;
          f_toupdate_i(dir_tang_1) += friction_force_1;
          f_toupdate_i(dir_tang_2) += friction_force_2;

	  // multiply by volume to get force from force density
          wall_reaction_force(dir_tang_1) -= (vol_i * friction_force_1);
          wall_reaction_force(dir_tang_2) -= (vol_i * friction_force_2);
        }
      }
    }

    if (CN.allow_damping) {
      // Damping force
      auto damping_coefficient =
	  2 * CN.damping_ratio * sqrt(CN.normal_stiffness * rho) *
	  sqrt(abs(ca[1])); // area is occasionally -ve for very small values
      double damping_force = (-damping_coefficient) * rel_vel_i(dir);
      // y-value of the contact force, for wall_bottom
      // combined_contact_force[i](1) += total_damping_force;
      f_toupdate_i(dir) += damping_force;

    // multiply by volume to get force from force density
      wall_reaction_force(dir) -= (vol_i * damping_force);
    }
  }
};

// force on the central particle by the neighboring particle
template <unsigned dim>
vector<Matrix<double, 1, dim>>
get_contact_force(ParticleN<dim> P_C, ParticleN<dim> P_N, Contact CN) {
  vector<Matrix<double, 1, dim>> combined_contact_force;
  combined_contact_force.resize(P_C.nnodes);

  for (unsigned i = 0; i < P_C.nnodes; ++i) {

    // full force on the i-th node
    Matrix<double, 1, dim> ff_all_i = Matrix<double, 1, dim>::Zero();
    for (unsigned j = 0; j < P_N.nnodes; ++j) {
      auto dir = P_N.CurrPos[j] - P_C.CurrPos[i];

      if ((dir).norm() <= CN.contact_rad) {
        // c_NbdArr[i].push_back(j);	// saving contact neighbors
        double dir_norm = dir.norm();
        if (dir_norm == 0) {
          std::cout << "Error: nodes too close" << std::endl;
        }
        auto unit_dir = dir / dir_norm;
        double contact_rad_contrib = (CN.contact_rad - dir_norm);

        // volume of the neighboring node
        auto N_vol = P_N.vol[j];

        // normal repulsive contact force from the neighbor
        //
        auto f_contact =
            (-CN.normal_stiffness * contact_rad_contrib * N_vol) * unit_dir;
        ff_all_i += f_contact;

        if (CN.allow_damping) {
          auto rel_v = P_C.vel[i] - P_N.vel[j]; // caution: -ve
          double rel_v_proj = rel_v.dot(unit_dir);
          auto rel_v_normal_comp = rel_v_proj * unit_dir;

          double damping_coefficient = 2 * CN.damping_ratio *
                                       sqrt(CN.normal_stiffness * P_C.rho) *
                                       sqrt(N_vol);
          auto f_damping = (-damping_coefficient) * rel_v_normal_comp;

          ff_all_i += f_damping;
        }

        if (CN.allow_friction) {
          auto contact_force_norm = f_contact.norm();

          auto rel_v = P_C.vel[i] - P_N.vel[j]; // caution: -ve

          // v - (v.e)e, relative velocity of center in the tangential direction
          auto rel_v_tangential_comp = rel_v - rel_v.dot(unit_dir) * unit_dir;
          auto rel_v_tangential_unit =
              rel_v_tangential_comp / rel_v_tangential_comp.norm(); // in place

          auto f_friction = -CN.friction_coefficient * contact_force_norm *
                            rel_v_tangential_unit;

          ff_all_i += f_friction;
        }
      }
    }

    combined_contact_force[i] = ff_all_i;
  }

  return combined_contact_force;
};

// force on the central particle by the neighboring particle
template <unsigned dim>
vector<Matrix<double, 1, dim>>
get_wall_contact_force(ParticleN<dim> P_C, RectWall<dim> Wall, Contact CN) {
  vector<Matrix<double, 1, dim>> combined_contact_force;
  combined_contact_force.resize(P_C.nnodes);
  // set zero to initialize
  for (unsigned i = 0; i < P_C.nnodes; i++) {
    // combined_contact_force[i] = Matrix<double, 1, dim>::Zero();
    combined_contact_force[i].setZero();
  }

  for (unsigned i = 0; i < P_C.nnodes; i++) {
    double y_dist_from_bottom = P_C.CurrPos[i](1) - Wall.bottom;
    double x_dist_from_left = P_C.CurrPos[i](0) - Wall.left;
    // double y_dist_from_top = Wall.top - P_C.CurrPos[i](1);
    // double x_dist_from_right = Wall.right - P_C.CurrPos[i](0);
    //// Using signed distance, to determine c_sign in wall_side_force_update()
    double y_dist_from_top = P_C.CurrPos[i](1) - Wall.top;
    double x_dist_from_right = P_C.CurrPos[i](0) - Wall.right;

    // single side interaction
    wall_side_force_update<dim>(y_dist_from_bottom, CN, 1, P_C.vel[i], P_C.rho,
                                combined_contact_force[i]);
    wall_side_force_update<dim>(x_dist_from_left, CN, 0, P_C.vel[i], P_C.rho,
                                combined_contact_force[i]);
    wall_side_force_update<dim>(x_dist_from_right, CN, 0, P_C.vel[i], P_C.rho,
                                combined_contact_force[i]);
    wall_side_force_update<dim>(y_dist_from_top, CN, 1, P_C.vel[i], P_C.rho,
                                combined_contact_force[i]);

    // double wall side interaction
  }

  return combined_contact_force;
};

// force on the central particle by the neighboring particle
// Use only nodes on the boundary. This applies when there is no bond-breaking
template <unsigned dim>
vector<Matrix<double, 1, dim>> get_contact_force_by_boundary(ParticleN<dim> P_C,
                                                             ParticleN<dim> P_N,
                                                             Contact CN) {
  // debug, instead of defining here, use existing force from outside the
  // function
  vector<Matrix<double, 1, dim>> combined_contact_force;
  // initiate with zero
  combined_contact_force.resize(P_C.nnodes);
  for (unsigned i = 0; i < P_C.nnodes; i++) {
    combined_contact_force[i].setZero();
  }


  // if bonds can break, work with all the nodes, otherwise work with outer
  // nodes
  unsigned tn_i = (P_C.break_bonds ? P_C.nnodes : P_C.boundary_nodes.size());
  unsigned tn_j = (P_N.break_bonds ? P_N.nnodes : P_N.boundary_nodes.size());

  // std::cout << tn_i << std::endl;
  for (unsigned p = 0; p < tn_i; ++p) {
    // if bonds can break, return the absolute node, otherwise the pth boundary
    // node
    auto i = (P_C.break_bonds ? p : P_C.boundary_nodes[p]);

    // full force on the i-th node
    Matrix<double, 1, dim> ff_all_i = Matrix<double, 1, dim>::Zero();
    for (unsigned q = 0; q < tn_j; ++q) {
      // if bonds can break, return the absolute node, otherwise the pth
      // boundary node
      auto j = (P_N.break_bonds ? q : P_N.boundary_nodes[q]);

      auto dir = P_N.CurrPos[j] - P_C.CurrPos[i];

      if ((dir).norm() <= CN.contact_rad) {
        // c_NbdArr[i].push_back(j);	// saving contact neighbors
        double dir_norm = dir.norm();
        if (dir_norm == 0) {
          std::cout << "Error: nodes too close" << std::endl;
        }
        auto unit_dir = dir / dir_norm;

	//double contact_rad_contrib;

	double contact_rad_contrib = (CN.contact_rad - dir_norm);
        // volume of the neighboring node
        auto N_vol = P_N.vol[j];

        // normal repulsive contact force from the neighbor
        //
        auto f_contact =
            (-CN.normal_stiffness * contact_rad_contrib * N_vol) * unit_dir;
        ff_all_i += f_contact;
        // combined_contact_force[i] += f_contact;
	

        if (CN.allow_damping) {
          auto rel_v = P_C.vel[i] - P_N.vel[j]; // caution: -ve
          double rel_v_proj = rel_v.dot(unit_dir);
          auto rel_v_normal_comp = rel_v_proj * unit_dir;

          double damping_coefficient = 2 * CN.damping_ratio *
                                       sqrt(CN.normal_stiffness * P_C.rho) *
                                       sqrt(N_vol);
          auto f_damping = (-damping_coefficient) * rel_v_normal_comp;

          ff_all_i += f_damping;
          // combined_contact_force[i] += f_damping;
        }

        if (CN.allow_friction) {
          auto contact_force_norm = f_contact.norm();

          auto rel_v = P_C.vel[i] - P_N.vel[j]; // caution: -ve

          // v - (v.e)e, relative velocity of center in the tangential direction
          auto rel_v_tangential_comp = rel_v - rel_v.dot(unit_dir) * unit_dir;
          auto rel_v_tangential_unit =
              rel_v_tangential_comp / rel_v_tangential_comp.norm(); // in place

          auto f_friction = -CN.friction_coefficient * contact_force_norm *
                            rel_v_tangential_unit;

          ff_all_i += f_friction;
          // combined_contact_force[i] += f_friction;
        }
      }
    }

    combined_contact_force[i] = ff_all_i;
  //std::cout << "p= " << p << " I am here ......."  << std::endl;
  }
  //std::cout << " I am out ......."  << std::endl;

  return combined_contact_force;
};

// simple contact force update, for understanding
template <unsigned dim>
void simple_update_contact_force(
    ParticleN<dim> P_C, ParticleN<dim> P_N, Contact CN,
    vector<Matrix<double, 1, dim>> &combined_contact_force) {
  for (unsigned i = 0; i < P_C.nnodes; ++i) {
    for (unsigned j = 0; j < P_N.nnodes; ++j) {

      auto dir = P_N.CurrPos[j] - P_C.CurrPos[i];
      double dir_norm = dir.norm();

      if (dir_norm == 0) {
        std::cout << "Error: nodes too close" << std::endl;
      }

      if (dir_norm <= CN.contact_rad) {
        auto unit_dir = dir / dir_norm;
        double contact_rad_contrib = (CN.contact_rad - dir_norm);

        // volume of the neighboring node
        auto N_vol = P_N.vol[j];

        // normal repulsive contact force from the node in neighbor
        auto f_contact =
            (-CN.normal_stiffness * contact_rad_contrib * N_vol) * unit_dir;
        // std::cout << "i=" << i << " j="  << j << " " <<  f_contact <<
        // std::endl; std::cout << "i" << " j " <<  " force norm: " <<
        // f_contact.norm() << std::endl;
        combined_contact_force[i] += f_contact;
      }
    }
  }
};


// force on the central particle by the neighboring particle
template <unsigned dim>
void update_contact_force_by_boundary(
    ParticleN<dim> P_C, ParticleN<dim> P_N, Contact CN, double dt,
    vector<Matrix<double, 1, dim>> &giant_f, unsigned i_starting_ind, unsigned j_starting_ind, bool is_self_contact) {
  int contacts=0;
  // if specified, work with all the nodes, otherwise work with outer nodes
  unsigned tn_i = (CN.nl_bdry_only ? P_C.boundary_nodes.size() : P_C.nnodes);
  unsigned tn_j = (CN.nl_bdry_only ? P_N.boundary_nodes.size() : P_N.nnodes);

  for (unsigned p = 0; p < tn_i; ++p) {
    // if specified, return the absolute node, otherwise the pth boundary node
    auto i = (CN.nl_bdry_only ? P_C.boundary_nodes[p] : p);

    for (unsigned q = 0; q < tn_j; ++q) {
      // if specified, return the absolute node, otherwise the pth boundary node
      auto j = (CN.nl_bdry_only ? P_N.boundary_nodes[q] : q);

      // auto dir = P_N.CurrPos[j] - P_C.CurrPos[i];
      // double dir_norm = dir.norm();
    // Debug: to avoid roundoff error (similar to computing peridynamic force)
    auto xi_ij = P_N.pos[j] - P_C.pos[i];
    auto dir = (P_N.disp[j] - P_C.disp[i]) + xi_ij;
    double dir_norm = dir.norm();
      
      if ( dir_norm <= CN.contact_rad ) {
	  // If is_self_contact=0, this is always 1
	  bool do_compute=1;

	  auto xi_ij_norm = xi_ij.norm();
	  double len_ref = CN.contact_rad;
	  // peridynamic spring constant (to be multiplied with the stretch value)
	  // assuming that normal_stiffness = material.cnot(delta=contact_rad) / contact_rad
	  double spring_const = CN.normal_stiffness * CN.contact_rad;

	  if (is_self_contact) {
	      if (i==j) {
		  do_compute = 0;
	      }
	      else if ( std::find(P_C.NbdArr[i].begin(), P_C.NbdArr[i].end(), j) != P_C.NbdArr[i].end() ) {
		  // if a bond exists between them, don't compute self-contact force
		  do_compute = 0;
	      }
	      else if ( xi_ij_norm < CN.contact_rad  ){
		  // This case works best when R_c < delta
		  // Otherwise (R_c >= delta), long-range repulsion acts between far away nodes  (delta<|xi|<R_c) whereas no repulsion between short-range connected bonds (|xi|<delta), which is inconsistent and will lead to instability 
		  len_ref = xi_ij_norm;
		  spring_const = P_N.cnot;
	      }
	      else { // xi_ij_norm >= CN.contact_rad
		  len_ref = CN.contact_rad;
	      }
	  }

      if (do_compute && (dir_norm < len_ref)) { // compression only: dir_norm < len_ref
	          
	contacts++;
        // c_NbdArr[i].push_back(j);	// saving contact neighbors
        if (dir_norm == 0) {
          std::cout << "Error: nodes too close" << std::endl;
        }
        // volume of the neighboring node
        auto N_vol = P_N.vol[j];

        auto unit_dir = dir / dir_norm;

        // normal repulsive contact force from the neighbor
        // auto f_contact =
	//     (-K_n * contact_rad_contrib * N_vol) * unit_dir;
	    
	// using stretch formulation
	double str = (dir_norm - len_ref) / len_ref;
	// moved this logic to if; repulsive-only: only when s is negative, repulsive force acts on node i
	// if (str > 0) {
	//     str = 0;
	// }
	auto f_contact = spring_const * str * N_vol * unit_dir;

	auto contact_force_norm = f_contact.norm();

        // ff_all_i += f_contact;
        //combined_contact_force[i] += f_contact;
	if (P_C.stoppable) {
	    giant_f[i + i_starting_ind] += f_contact;
	}


	if (!is_self_contact) { // for self-contact, equal and opposite force is redundant

	    if (P_N.stoppable) {
		// giant_f[j + j_starting_ind] -= f_contact;
		giant_f[j + j_starting_ind] -= f_contact / N_vol * P_C.vol[i];
	    }
	}

	//// if (CN.enable_cohesion) {
	//if (0) {
	//    //// Cohesion
	//    
	//    //double cohesion_dist_contrib = (CN.contact_rad - 2 * dir_norm * CN.cohesion_scaling);
	//    //
	//    // reflection of repulsion curve
	//    //double cohesion_dist_contrib =  -dir_norm * CN.cohesion_scaling;
	//    // constant 
	//    double cohesion_dist_contrib =  -CN.cohesion_scaling * spring_const;
	//    // auto f_cohesion = (-K_n * cohesion_dist_contrib * N_vol) * unit_dir;
	//    auto f_cohesion = (-spring_const * cohesion_dist_contrib * N_vol) * unit_dir;


	//    //combined_contact_force[i] += f_cohesion;
	//    if (P_C.stoppable) {
	//	giant_f[i + i_starting_ind] += f_cohesion;
	//    }
	//    if (P_N.stoppable) {
	//	giant_f[j + j_starting_ind] -= f_cohesion;
	//    }
	//}
	    

        if (CN.allow_damping) {
          auto rel_v = P_C.vel[i] - P_N.vel[j]; // caution: -ve

          double rel_v_proj = rel_v.dot(unit_dir);
          auto rel_v_normal_comp = rel_v_proj * unit_dir;

          double damping_coefficient = 2 * CN.damping_ratio *
                                       sqrt(spring_const * P_C.rho) *
                                       sqrt(N_vol);

	  Matrix<double, 1, dim> f_damping;
          f_damping = (-damping_coefficient) * rel_v_normal_comp;

	  // Edit for small velocity
	  // Did not work, probably because the idea is incorrect (that the velocity should be zero at the next timestep if the sign changes from this to the next timestep)
	  //if ( (P_C.rho/dt * rel_v_normal_comp.norm()) < (contact_force_norm + damping_coefficient * rel_v_normal_comp.norm()) ) {
	      ////std::cout << "Edit here" << std::endl;
	      //double f_damping_abs = P_C.rho / dt * rel_v_normal_comp.norm() - contact_force_norm;

	      //f_damping = -f_damping_abs / rel_v_normal_comp.norm() * rel_v_normal_comp;
	  //}

          //combined_contact_force[i] += f_damping;
	  if (P_C.stoppable) {
	      giant_f[i + i_starting_ind] += f_damping;
	  }

	if (!is_self_contact) { // for self-contact, equal and opposite force is redundant
	  if (P_N.stoppable) {
	      // giant_f[j + j_starting_ind] -= f_damping;
	      giant_f[j + j_starting_ind] -= f_damping / sqrt(N_vol) * sqrt(P_C.vol[i]);
	  }
	}
        }

        if (CN.allow_friction) {

          auto rel_v = P_C.vel[i] - P_N.vel[j]; // caution: -ve

	  // What if this is zero?

          // v - (v.e)e, relative velocity of center in the tangential direction
          auto rel_v_tangential_comp = rel_v - rel_v.dot(unit_dir) * unit_dir;

          auto rel_v_tangential_norm = rel_v_tangential_comp.norm();


          if (rel_v_tangential_norm > 0) {
            auto rel_v_tangential_unit =
                rel_v_tangential_comp /
                rel_v_tangential_comp.norm(); // in place

            Matrix<double, 1, dim> f_friction;

            f_friction = -CN.friction_coefficient * contact_force_norm *
                         rel_v_tangential_unit;
            // Correction for small velocity
	    if (rel_v_tangential_norm < CN.friction_coefficient * contact_force_norm * dt / P_C.rho) {
		//std::cout << "small velocity detected" << std::endl;
	      f_friction = -rel_v_tangential_norm/P_C.rho/dt * rel_v_tangential_unit;
	    }
            //combined_contact_force[i] += f_friction;
	    if (P_C.stoppable) {
		giant_f[i + i_starting_ind] += f_friction;
	    }
	if (!is_self_contact) { // for self-contact, equal and opposite force is redundant
	    if (P_N.stoppable) {
		giant_f[j + j_starting_ind] -= f_friction;
	    }
	}
          }
        }
	  } // do compute
      }	// < R_c
    }
  }
    if (!is_self_contact) {
    //std::cout << " computations " << contacts  << std::endl;
    }
};

// force on the central particle by the neighboring particle
template <unsigned dim>
void update_wall_contact_force(
    ParticleN<dim> P_C, RectWall<dim> Wall, Contact CN,
    vector<Matrix<double, 1, dim>> &combined_contact_force) {

  // combined_contact_force.resize(P_C.nnodes);
  //// set zero to initialize
  // for (unsigned i = 0; i < P_C.nnodes; i++) {
  ////combined_contact_force[i] = Matrix<double, 1, dim>::Zero();
  // combined_contact_force[i].setZero();
  //}

  for (unsigned i = 0; i < P_C.nnodes; i++) {
    double y_dist_from_bottom = P_C.CurrPos[i](1) - Wall.bottom;
    double x_dist_from_left = P_C.CurrPos[i](0) - Wall.left;
    // double y_dist_from_top = Wall.top - P_C.CurrPos[i](1);
    // double x_dist_from_right = Wall.right - P_C.CurrPos[i](0);
    //// Using signed distance, to determine c_sign in wall_side_force_update()
    double y_dist_from_top = P_C.CurrPos[i](1) - Wall.top;
    double x_dist_from_right = P_C.CurrPos[i](0) - Wall.right;

    // single side interaction
    wall_side_force_update(y_dist_from_bottom, CN, 1, P_C.vel[i], P_C.rho,
                           combined_contact_force[i]);
    wall_side_force_update(x_dist_from_left, CN, 0, P_C.vel[i], P_C.rho,
                           combined_contact_force[i]);
    wall_side_force_update(x_dist_from_right, CN, 0, P_C.vel[i], P_C.rho,
                           combined_contact_force[i]);
    wall_side_force_update(y_dist_from_top, CN, 1, P_C.vel[i], P_C.rho,
                           combined_contact_force[i]);

    // double wall side interaction
  }
  // return combined_contact_force;
};

// force on the central particle by the neighboring particle
// Using boundary nodes
template <unsigned dim>
vector<Matrix<double, 1, dim>>
get_wall_contact_force_by_boundary(ParticleN<dim> P_C, RectWall<dim> Wall,
                                   Contact CN) {

  vector<Matrix<double, 1, dim>> combined_contact_force;
  combined_contact_force.resize(P_C.nnodes);
  // set zero to initialize
  for (unsigned i = 0; i < P_C.nnodes; i++) {
    // combined_contact_force[i] = Matrix<double, 1, dim>::Zero();
    combined_contact_force[i].setZero();
  }

  // if bonds can break, work with all the nodes, otherwise work with outer
  // nodes
  unsigned tn_i = (P_C.break_bonds ? P_C.nnodes : P_C.boundary_nodes.size());
  for (unsigned p = 0; p < tn_i; p++) {
    // if bonds can break, return the absolute node, otherwise the pth boundary
    // node
    auto i = (P_C.break_bonds ? p : P_C.boundary_nodes[p]);

    double y_dist_from_bottom = P_C.CurrPos[i](1) - Wall.bottom;
    double x_dist_from_left = P_C.CurrPos[i](0) - Wall.left;
    // double y_dist_from_top = Wall.top - P_C.CurrPos[i](1);
    // double x_dist_from_right = Wall.right - P_C.CurrPos[i](0);
    //// Using signed distance, to determine c_sign in wall_side_force_update()
    double y_dist_from_top = P_C.CurrPos[i](1) - Wall.top;
    double x_dist_from_right = P_C.CurrPos[i](0) - Wall.right;

    // single side interaction
    wall_side_force_update(y_dist_from_bottom, CN, 1, P_C.vel[i], P_C.rho,
                           combined_contact_force[i]);
    wall_side_force_update(x_dist_from_left, CN, 0, P_C.vel[i], P_C.rho,
                           combined_contact_force[i]);
    wall_side_force_update(x_dist_from_right, CN, 0, P_C.vel[i], P_C.rho,
                           combined_contact_force[i]);
    wall_side_force_update(y_dist_from_top, CN, 1, P_C.vel[i], P_C.rho,
                           combined_contact_force[i]);

    // double wall side interaction
  }
  return combined_contact_force;
};



// apply ambient force field to all nodes
template <unsigned dim>
void apply_force_field(
    ParticleN<dim> P_C, int forcefield_type, double dt, unsigned t, unsigned timesteps, vector<Matrix<double, 1, dim>> &giant_f, unsigned i_starting_ind, double forcefield_scaling, bool normalized, bool by_boundary) {
  // if specified, work with all the nodes, otherwise work with outer nodes
  unsigned tn_i = (by_boundary ? P_C.boundary_nodes.size() : P_C.nnodes);
  for (unsigned p = 0; p < tn_i; p++) {
    auto i = (by_boundary ? P_C.boundary_nodes[p] : p);

    //double angle = M_PI/2;
    double angle = M_PI/4;
    //double angle = (M_PI - M_PI/2)/(timesteps - 1)* t + M_PI/2;
    //double angle = (M_PI*3/2 - M_PI/2)/(timesteps - 1)* t + M_PI/2;


    if (forcefield_type == 1) {
	// constant force towards the center of the domain
	//combined_contact_force[i] += vortexfield<dim>(P_C.CurrPos[i], normalized, forcefield_scaling, -M_PI, t); 
	giant_f[i + i_starting_ind] += vortexfield<dim>(P_C.CurrPos[i], normalized, forcefield_scaling, -M_PI, t); 
    }
    else if (forcefield_type == 2) {
	// shear field
	//combined_contact_force[i] += shearfield<dim>(P_C.CurrPos[i], normalized, forcefield_scaling, angle, t); 
	giant_f[i + i_starting_ind] += shearfield<dim>(P_C.CurrPos[i], normalized, forcefield_scaling, angle, t); 
    }
  }
};

// force on the central particle by the neighboring particle
// Using boundary nodes
template <unsigned dim>
void update_wall_contact_force_by_boundary(
    ParticleN<dim> P_C, RectWall<dim> &Wall, Contact CN, double dt,
    vector<Matrix<double, 1, dim>> &giant_f, unsigned i_starting_index) {

  // if specified, work with all the nodes, otherwise work with outer nodes
  unsigned tn_i = (CN.nl_bdry_only ? P_C.boundary_nodes.size() : P_C.nnodes);

  for (unsigned p = 0; p < tn_i; p++) {
    // if specified, work with all the nodes, otherwise work with outer nodes
    auto i = (CN.nl_bdry_only ? P_C.boundary_nodes[p] : p);

    if (dim == 2) {
      double y_dist_from_bottom = P_C.CurrPos[i](1) - Wall.bottom;
      double x_dist_from_left = P_C.CurrPos[i](0) - Wall.left;
      // double y_dist_from_top = Wall.top - P_C.CurrPos[i](1);
      // double x_dist_from_right = Wall.right - P_C.CurrPos[i](0);
      //// Using signed distance, to determine c_sign in
      ///wall_side_force_update()
      double y_dist_from_top = P_C.CurrPos[i](1) - Wall.top;
      double x_dist_from_right = P_C.CurrPos[i](0) - Wall.right;

      // single side interaction
      // Debug: added TL
      wall_side_force_update<dim>(y_dist_from_bottom, CN, 1, P_C.vel[i], P_C.vol[i],
                                  P_C.rho, giant_f[i+i_starting_index], dt, Wall.speed_bottom, Wall.reaction_bottom);
      wall_side_force_update<dim>(x_dist_from_left, CN, 0, P_C.vel[i], P_C.vol[i], P_C.rho,
                                  giant_f[i+i_starting_index], dt, Wall.speed_left, Wall.reaction_left);
      wall_side_force_update<dim>(x_dist_from_right, CN, 0, P_C.vel[i], P_C.vol[i], P_C.rho,
                                  giant_f[i+i_starting_index], dt, Wall.speed_right, Wall.reaction_right);
      wall_side_force_update<dim>(y_dist_from_top, CN, 1, P_C.vel[i], P_C.vol[i], P_C.rho,
                                  giant_f[i+i_starting_index], dt, Wall.speed_top, Wall.reaction_top);

      //std::cout << Wall.get_reaction() << std::endl;
      // double wall side interaction
    } else {
	  // these are signed distances
	  double dist_from_x_min = P_C.CurrPos[i](0) - Wall.x_min;
	  double dist_from_y_min = P_C.CurrPos[i](1) - Wall.y_min;
	  double dist_from_z_min = P_C.CurrPos[i](2) - Wall.z_min;
	  double dist_from_x_max = P_C.CurrPos[i](0) - Wall.x_max;
	  double dist_from_y_max = P_C.CurrPos[i](1) - Wall.y_max;
	  double dist_from_z_max = P_C.CurrPos[i](2) - Wall.z_max;


	  wall_side_force_update<dim>(dist_from_z_min, CN, 2, P_C.vel[i], P_C.vol[i], P_C.rho, giant_f[i+i_starting_index], dt, Wall.speed_z_min, Wall.reaction_z_min);
	  wall_side_force_update<dim>(dist_from_z_max, CN, 2, P_C.vel[i], P_C.vol[i], P_C.rho, giant_f[i+i_starting_index], dt, Wall.speed_z_max, Wall.reaction_z_max);

	if (Wall.wall_cyl) {
	  // cylindrical wall
	  double dist_from_cyl_cent_x = Wall.wall_cyl_cent_x - P_C.CurrPos[i](0);
	  double dist_from_cyl_cent_y = Wall.wall_cyl_cent_y - P_C.CurrPos[i](1) ;

	  double dist_from_cyl_cent = pow(pow(dist_from_cyl_cent_x, 2) + pow(dist_from_cyl_cent_y, 2), 0.5);
	  double dist_from_cyl_wall = dist_from_cyl_cent - Wall.wall_cyl_rad;

	  double cyl_inward_normal_x = (dist_from_cyl_cent > 0 ? dist_from_cyl_cent_x/dist_from_cyl_cent : 0);
	  double cyl_inward_normal_y = (dist_from_cyl_cent > 0 ? dist_from_cyl_cent_y/dist_from_cyl_cent : 0);

// void wall_side_force_update(double dist, Contact CN, unsigned dir,
//                             Matrix<double, 1, dim> vel_i, double vol_i, double rho,
//                             Matrix<double, 1, dim> &f_toupdate_i, double dt, double wall_side_speed, Matrix<double, 1, dim> &wall_reaction_force) {
	  wall_cyl_side_force_update<dim>(dist_from_cyl_wall, CN, cyl_inward_normal_x, cyl_inward_normal_y, P_C.vel[i], P_C.vol[i], P_C.rho, giant_f[i+i_starting_index], dt, 0.0, Wall.reaction_r);

	} else {
	  // single side interaction
	  wall_side_force_update<dim>(dist_from_x_min, CN, 0, P_C.vel[i], P_C.vol[i], P_C.rho, giant_f[i+i_starting_index], dt, Wall.speed_x_min, Wall.reaction_x_min);
	  wall_side_force_update<dim>(dist_from_x_max, CN, 0, P_C.vel[i], P_C.vol[i], P_C.rho, giant_f[i+i_starting_index], dt, Wall.speed_x_max, Wall.reaction_x_max);

	  wall_side_force_update<dim>(dist_from_y_min, CN, 1, P_C.vel[i], P_C.vol[i], P_C.rho, giant_f[i+i_starting_index], dt, Wall.speed_y_min, Wall.reaction_y_min);
	  wall_side_force_update<dim>(dist_from_y_max, CN, 1, P_C.vel[i], P_C.vol[i], P_C.rho, giant_f[i+i_starting_index], dt, Wall.speed_y_max, Wall.reaction_y_max);


	}
    }
  }
  // return combined_contact_force;
};

// see if two boxes intersect
bool intersects_box(vector<double> A_lrbt, vector<double> B_lrbt) {
  double A_Left = A_lrbt[0];
  double A_Right = A_lrbt[1];
  double A_Top = A_lrbt[2];
  double A_Bottom = A_lrbt[3];

  double B_Left = B_lrbt[0];
  double B_Right = B_lrbt[1];
  double B_Top = B_lrbt[2];
  double B_Bottom = B_lrbt[3];

  return ((A_Left <= B_Right) && (A_Right >= B_Left) && (A_Top >= B_Bottom) &&
          (A_Bottom <= B_Top));
};

// see if two boxes intersect, when the first box is extended in all directions
template <unsigned dim>
bool intersects_box_fext(vector<double> A_lrbt, vector<double> B_lrbt,
                         double c) {
    bool out;
  if (dim == 2) {
    double A_Left = A_lrbt[0] - c;
    double A_Right = A_lrbt[1] + c;
    double A_Top = A_lrbt[2] + c;
    double A_Bottom = A_lrbt[3] - c;

    double B_Left = B_lrbt[0];
    double B_Right = B_lrbt[1];
    double B_Top = B_lrbt[2];
    double B_Bottom = B_lrbt[3];

    out = ((A_Left <= B_Right) && (A_Right >= B_Left) && (A_Top >= B_Bottom) &&
            (A_Bottom <= B_Top));

  } else if (dim == 3) {

    double A_min_x = A_lrbt[0] - c;
    double A_min_y = A_lrbt[1] - c;
    double A_min_z = A_lrbt[2] - c;
    double A_max_x = A_lrbt[3] + c;
    double A_max_y = A_lrbt[4] + c;
    double A_max_z = A_lrbt[5] + c;

    double B_min_x = B_lrbt[0];
    double B_min_y = B_lrbt[1];
    double B_min_z = B_lrbt[2];
    double B_max_x = B_lrbt[3];
    double B_max_y = B_lrbt[4];
    double B_max_z = B_lrbt[5];

    out = ((A_min_x <= B_max_x) && (A_max_x >= B_min_x) &&
            (A_max_y >= B_min_y) && (A_min_y <= B_max_y) &&
            (A_min_z <= B_max_z) && (A_max_z >= B_min_z));
  }

  return out;
};

// get the min and max
template <unsigned dim>
vector<double> get_minmax(vector<Matrix<double, 1, dim>> P) {
  vector<double> out;

  if (dim == 2) {
    // fill it up with first values, left, right, top, bottom
    // fill it up with first values, x_min, x_max, y_max, y_min
    out.push_back(P[0](0));
    out.push_back(P[0](0));
    out.push_back(P[0](1));
    out.push_back(P[0](1));

    for (unsigned i = 0; i < P.size(); i++) {
      out[0] = min(out[0], P[i](0));
      out[1] = max(out[1], P[i](0));
      out[2] = max(out[2], P[i](1));
      out[3] = min(out[3], P[i](1));
    }
  } else if (dim == 3) {
    // fill it up with first values, x_min, y_min, z_min, x_max, y_max, z_max

    out.push_back(P[0](0));
    out.push_back(P[0](1));
    out.push_back(P[0](2));
    out.push_back(P[0](0));
    out.push_back(P[0](1));
    out.push_back(P[0](2));

    for (unsigned i = 0; i < P.size(); i++) {
      out[0] = min(out[0], P[i](0));
      out[1] = min(out[1], P[i](1));
      out[2] = min(out[2], P[i](2));
      out[3] = max(out[3], P[i](0));
      out[4] = max(out[4], P[i](1));
      out[5] = max(out[5], P[i](2));
    }
  }

  return out;
};

// Whether box A is within wall_lrtb
// box A is extended in all directions by c
bool within_interior_fext(vector<double> A_lrbt, vector<double> B_lrbt,
                          double c) {
  double A_Left = A_lrbt[0] - c;
  double A_Right = A_lrbt[1] + c;
  double A_Top = A_lrbt[2] + c;
  double A_Bottom = A_lrbt[3] - c;

  double B_Left = B_lrbt[0];
  double B_Right = B_lrbt[1];
  double B_Top = B_lrbt[2];
  double B_Bottom = B_lrbt[3];

  return ((B_Left < A_Left) && (A_Right < B_Right) && (B_Top > A_Top) &&
          (B_Bottom < A_Bottom));
};

#endif /* ifndef CONTACT_H */
