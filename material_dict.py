import numpy as np

# class Material(object):
#     """docstring for Material"""
#     # def __init__(self, delta, rho, snot, cnot, bulk_modulus, E = None, nu = None, Gnot = None, shear_modulus = None, name=None):
    #     super(Material, self).__init__()
    #     self.delta  = delta
    #     self.rho    = rho     
    #     self.snot   = snot
    #     self.cnot   = cnot
    #     self.bulk_modulus = bulk_modulus
    #
    #     self.E = E
    #     self.nu = nu
    #     self.Gnot = Gnot
    #     self.shear_modulus = shear_modulus
    #
    #     self.name = name
    #
    # def generate(self, delta, rho_scale=1, K_scale=1, G_scale=1, Gnot_scale=1):

        # if self.name == 'peridem':
            # self.bulk_modulus = 2.0e9 * K_scale
            # self.shear_modulus = 1.33e+09 * G_scale
            # self.rho=1200.0	* rho_scale
            # self.Gnot = 135.0 * Gnot_scale
            # self.nu = 1/3

            # self.E = 9 * self.bulk_modulus * self.shear_modulus / ( 9 * self.bulk_modulus + self.shear_modulus);

            # self.cnot = 24 * self.E /( (1 - self.nu) * np.pi * (self.delta**3) );
            # self.snot = np.sqrt(4 * np.pi * self.Gnot /(9*self.E*self.delta));
        
        # if self.name ==  'sodalime_similar_to_peridem'
            # E = 1e9 * E_scale
            # rho=1200 * rho_scale
            # # nu = 1/4
            # nu = 1/3
            # Gnot = 135 * Gnot_scale

            # cnot = 6*E/( np.pi * (delta**3) * (1 - nu))

            # snot = np.sqrt(4 * np.pi * Gnot /(9*E*delta))

            # bulk_modulus = E/ (3 * ( 1 - 2 *nu)) 
            # # extra
            # shear_modulus = E/ 2/ (1 + nu)
# class Material(object):
#     def __init__(self, delta, rho, snot, s_tension_crit,s_compression_crit,cnot, bulk_modulus,
#                  E=None, nu=None, Gnot=None, shear_modulus=None, name=None,
#                  # --- NEW (optional) yield/plateau params ---
#                  rLp=None, rLm=None, delta_y=None, use_yield_plateau_law=False):
#         super(Material, self).__init__()
#         self.delta  = delta
#         self.rho    = rho
#         self.snot   = snot
#         self.cnot   = cnot
#
#         self.bulk_modulus = bulk_modulus
#
#         self.s_tension_crit=s_tension_crit
#         self.s_compression_crit=s_compression_crit
#
#         self.E = E
#         self.nu = nu
#         self.Gnot = Gnot
#         self.shear_modulus = shear_modulus
#         self.name = name
#
#         # --- NEW ---
#         self.rLp = rLp                  # tension yield stretch
#         self.rLm = rLm                  # compression yield stretch (positive)
#         self.delta_y = delta_y          # smoothing for yield->plateau
#         self.use_yield_plateau_law = use_yield_plateau_law
#

# class Material(object):
#     def __init__(
#         self,
#         delta, rho, snot,
#         s_tension_crit, s_compression_crit,
#         cnot, bulk_modulus,
#         E=None, nu=None, Gnot=None, shear_modulus=None, name=None,
#
#         # --- yield/plateau params ---
#         rLp=None, rLm=None, delta_y=None, use_yield_plateau_law=False,
#
#         # --- NEW: softening + break params ---
#         rSp=None, rFp=None,   # tension: softening start / fracture
#         rSm=None, rFm=None,   # compression magnitudes: softening start / fracture
#
#         # --- optional: default ratios for auto-fill ---
#         soft_start_mult=3.0,   # rS = soft_start_mult * rL
#         fracture_mult=6.0,     # rF = fracture_mult   * rL
#         min_gap=0.0            # if you want, set e.g. 5e-6 to enforce separation
#     ):
#         super(Material, self).__init__()
#
#         # base
#         self.delta  = float(delta)
#         self.rho    = float(rho)
#         self.snot   = float(snot)
#         self.cnot   = float(cnot)
#         self.bulk_modulus = float(bulk_modulus)
#
#         self.s_tension_crit = float(s_tension_crit)
#         self.s_compression_crit = float(s_compression_crit)
#
#         self.E = E
#         self.nu = nu
#         self.Gnot = Gnot
#         self.shear_modulus = shear_modulus
#         self.name = name
#
#         # yield/plateau
#         self.rLp = rLp
#         self.rLm = rLm
#         self.delta_y = delta_y
#         self.use_yield_plateau_law = bool(use_yield_plateau_law)
#
#         # softening/break (may be None for now; we may auto-fill below)
#         self.rSp = rSp
#         self.rFp = rFp
#         self.rSm = rSm
#         self.rFm = rFm
#
#         # defaults for auto-fill
#         self.soft_start_mult = float(soft_start_mult)
#         self.fracture_mult = float(fracture_mult)
#         self.min_gap = float(min_gap)
#


import os
import numpy as np
import matplotlib.pyplot as plt
from tenstion_compresson_ottwa_consti_v3 import f_redesigned_vec

# import the functions from your attached script
# from constitutive_laws import f_redesigned_vec, plot_constitutive_law
# (or paste them directly above the class)

class Material:
    def __init__(self, *, delta, rho, snot, cnot, bulk_modulus,
                 E=None, nu=None, Gnot=None, shear_modulus=None, name=None,
                 # law params (optional)
                 rLp=None, rLm=None, delta_y=None,
                 rSp=None, rFp=None, rSm=None, rFm=None,
                 use_yield_plateau_law=False,
                 s_tension_crit=None, s_compression_crit=None):
        self.delta = delta
        self.rho = rho
        self.snot = snot
        self.cnot = cnot
        self.bulk_modulus = bulk_modulus
        self.E = E
        self.nu = nu
        self.Gnot = Gnot
        self.shear_modulus = shear_modulus
        self.name = name

        # store law params
        self.use_yield_plateau_law = use_yield_plateau_law
        self.rLp = rLp
        self.rLm = rLm
        self.delta_y = delta_y
        self.rSp = rSp
        self.rFp = rFp
        self.rSm = rSm
        self.rFm = rFm

        self.s_tension_crit = s_tension_crit
        self.s_compression_crit = s_compression_crit

    def constitutive_params(self):
        """Return dict in the format your plotting code expects."""
        if not self.use_yield_plateau_law:
            raise ValueError("use_yield_plateau_law is False; no redesigned law to plot.")

        # IMPORTANT: in your plotting file, compression parameters are given as positive magnitudes
        # and the plot uses -rL_neg, -rS_neg, -rF_neg for markers.
        return dict(
            mu=float(self.cnot),
            delta=float(self.delta_y),

            rL_pos=float(self.rLp), rS_pos=float(self.rSp), rF_pos=float(self.rFp),
            rL_neg=float(self.rLm), rS_neg=float(self.rSm), rF_neg=float(self.rFm),
        )
    


    def plot_constitutive_law(self, *, rmin=None, rmax=None, npts=1200,
                             show=True, savepath=None, dpi=200):
        """
        Plot (and optionally save) the constitutive law using your existing script logic.
        """
        p = self.constitutive_params()

        # auto domain like your script
        if rmin is None:
            rmin = -1.15 * p["rF_neg"]
        if rmax is None:
            rmax =  1.15 * p["rF_pos"]

        r = np.linspace(rmin, rmax, npts)
        f = f_redesigned_vec(r, **p)  # from your attached script

        plt.figure(figsize=(9, 5))
        plt.plot(r, f, "k", lw=2.2, label=r"$f(r)$")

        # markers (same as your script)
        plt.axvline(p["rL_pos"], linestyle="--", lw=1.6, label=r"$r_L^+$ (yield)")
        plt.axvline(p["rS_pos"], linestyle=":",  lw=1.6, label=r"$r_S^+$ (softening)")
        plt.axvline(p["rF_pos"], linestyle="-.", lw=1.6, label=r"$r_F^+$ (fracture)")

        plt.axvline(-p["rL_neg"], linestyle="--", lw=1.6, label=r"$-r_L^-$ (yield)")
        plt.axvline(-p["rS_neg"], linestyle=":",  lw=1.6, label=r"$-r_S^-$ (softening)")
        plt.axvline(-p["rF_neg"], linestyle="-.", lw=1.6, label=r"$-r_F^-$ (fracture)")

        plt.axhline(0, color="gray", lw=0.8)
        plt.axvline(0, color="gray", lw=0.8)
        plt.grid(True, alpha=0.3)

        title = self.name or "Material"
        plt.title(f"Constitutive law: {title}")
        plt.xlabel("stretch $r$")
        plt.ylabel("force $f(r)$")

        # unique legend entries
        handles, labels = plt.gca().get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        plt.legend(uniq.values(), uniq.keys(), fontsize=9, loc="upper left")

        plt.tight_layout()

        if savepath is not None:
            os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
            plt.savefig(savepath, dpi=dpi)

        if show:
            plt.show()
        else:
            plt.close()


    def print(self):
        """Print material properties"""
        print("=== Material properties ===")
        print(f"delta: {self.delta}")
        print(f"rho:   {self.rho}")

        if hasattr(self, "cnot"):
            print(f"cnot:  {self.cnot}")

        if hasattr(self, "snot"):
            print(f"snot:  {self.snot}")

        if hasattr(self, "E") and self.E is not None:
            print(f"E:     {self.E}")
        if hasattr(self, "K") and self.E is not None:
            print(f"K:     {self.K}")
        

        if hasattr(self, "nu") and self.E is not None:
            print(f"nu:     {self.nu}")


        # ----------------------------------
        # Tension / compression criteria
        # ----------------------------------
        if hasattr(self, "s_tension_crit"):
            print(f"s_tension_crit:     {self.s_tension_crit}")

        if hasattr(self, "s_compression_crit"):
            print(f"s_compression_crit: {self.s_compression_crit}")

        # ----------------------------------
        # Ratio-based parameters (if exist)
        # ----------------------------------
        for name in ["rLp", "rLm", "rSp", "rFp", "rSm", "rFm"]:
            if hasattr(self, name):
                print(f"{name}: {getattr(self, name)}")

        # ----------------------------------
        # Softening / fracture tuning params
        # ----------------------------------
        for name in ["min_gap", "aS", "aF"]:
            if hasattr(self, name):
                print(f"{name}: {getattr(self, name)}")

    # def print(self):
    #     """print info
    #     """
    #     print('delta: ', self.delta)
    #     print('rho: ', self.rho)
    #     print('cnot: ', self.cnot)
    #     print('snot: ', self.snot)
    #     print('E: ', self.E)
#              
#
def peridem(delta):
    """Generate material properties and peridynamic constants using delta
    """
    bulk_modulus = 2.0e9
    shear_modulus = 1.33e+09
    rho=1200.0	
    Gnot = 135.0

    #  nu = 0.2278
    nu = (3 * bulk_modulus - 2 * shear_modulus) / ( 2 * ( 3 * bulk_modulus + shear_modulus))
    #  E = 1.23e9
    E = 9 * bulk_modulus * shear_modulus / ( 9 * bulk_modulus + shear_modulus);


    cnot = 24 * E /( (1 - nu) * np.pi * (delta**3) );
    snot = np.sqrt(4 * np.pi * Gnot /(9*E*delta));

    # print('nu = ', nu)
    # print('E = ', E)

    # return Material(delta, rho, snot, cnot, bulk_modulus)
    return Material(delta, rho, snot, cnot, bulk_modulus, E = E, nu = nu, Gnot = Gnot, shear_modulus = shear_modulus )

def sodalime(delta):
    # print('There is some issue with these settings')
    E = 72e9
    rho=2440
    nu = 0.22 
    Gnot = 135

    cnot = 6*E/( np.pi * (delta**3) * (1 - nu))

    snot = np.sqrt(4 * np.pi * Gnot /(9*E*delta))

    bulk_modulus = E/ (3 * ( 1 - 2 *nu)) 

    # extra
    shear_modulus = E/ 2/ (1 + nu)

    return Material(delta, rho, snot, cnot, bulk_modulus, E = E, nu = nu, Gnot = Gnot, shear_modulus = shear_modulus )

def sodalime_similar_to_peridem(delta, E_scale=1, rho_scale=1, Gnot_scale=1):
    # print('There is some issue with these settings')
    E = 1e9 * E_scale
    rho=1200 * rho_scale
    # nu = 1/4
    nu = 1/3
    Gnot = 135 * Gnot_scale

    cnot = 6*E/( np.pi * (delta**3) * (1 - nu))

    snot = np.sqrt(4 * np.pi * Gnot /(9*E*delta))

    bulk_modulus = E/ (3 * ( 1 - 2 *nu)) 
    # extra
    shear_modulus = E/ 2/ (1 + nu)

    return Material(delta, rho, snot, cnot, bulk_modulus, E = E, nu = nu, Gnot = Gnot, shear_modulus = shear_modulus )

def peridem_1d_deformable(delta, rho_scale=1, K_scale=1, G_scale=1, Gnot_scale=1):
    """ Smaller fracture toughness
    """
    bulk_modulus = 2.0e9 * K_scale
    shear_modulus = 1.33e+09 * G_scale
    rho=1200.0	* rho_scale
    Gnot = 135.0 * Gnot_scale

    nu = 1/3
    E = 9 * bulk_modulus * shear_modulus / ( 9 * bulk_modulus + shear_modulus);

    # cnot = 24 * E /( (1 - nu) * np.pi * (delta**3) );
    cnot = 5 * E/ (delta**5)

    # incorrect snot
    snot = np.sqrt(4 * np.pi * Gnot /(9*E*delta));

    # print('nu = ', nu)
    # print('E = ', E)

    # return Material(delta, rho, snot, cnot, bulk_modulus)
    return Material(delta, rho, snot, cnot, bulk_modulus, E = E, nu = nu, Gnot = Gnot, shear_modulus = shear_modulus )


def kalthoff(delta, rho_scale=1, K_scale=1, G_scale=1, Gnot_scale=1):

    """ 
    Silling (See also Guo Gao):
    For the EMU model of the Kalthoff-Winkler experiment, the material parameters corresponded to a
    Young’s modulus of 191 GPa, a Poisson ratio of 0.3, a mass density of 8000 kg/m
    , a fracture toughness of 90 MPa-m 1/2
    , and a tensile strength of 2000 MPa.

    Thickness 9 or 19 mm (Guo, Gao)
    The projectile length l = 100 mm and diameter d = 50 mm which equals to the distance between notches.

    From https://www.sciencedirect.com/topics/materials-science/maraging-steel:
    The material parameters of Maraging steel 18Ni(300), which are taken from Park et al. (2012), are as follows: Young's modulus E0 = 1.9 × 105 MPa, Poisson's ratio ν0 = 0.3, the mass density ρ = 8000 kg/m3, the failure strength ft = 2812.25 MPa, and the fracture energy Gc = 22.2 N/mm, resulting in Griffith's internal length lch = 0.53 mm. The Rayleigh wave speed is cR = 2745 m/s.
    """
    rho = 8000.0	* rho_scale
    ## what should be Gnot???  Keeping intact
    #Gnot = 22200
    # Gnot = 90
    # Gnot = 42.4 𝑀𝑃𝑎. 𝑚𝑚 = 42400 J/m^2
    Gnot = 42400 * Gnot_scale
    E = 191e9
    # nu = 0.3
    nu = 0.3
    # E = 9 * bulk_modulus * shear_modulus / ( 9 * bulk_modulus + shear_modulus);
    bulk_modulus = E/(3*(1- 2*nu))
    shear_modulus = E/(2*(1+nu))

    # cnot = 24 * E /( (1 - nu) * np.pi * (delta**3) );
    # snot = np.sqrt(4 * np.pi * Gnot /(9*E*delta));

    cnot = 72 * bulk_modulus / (5 * np.pi * delta**3)
    snot = np.sqrt( Gnot / (6 * shear_modulus/np.pi + 16/9/(np.pi**2) * ( bulk_modulus - 2 * shear_modulus))/delta)

    # print('nu = ', nu)
    # print('E = ', E)

    # return Material(delta, rho, snot, cnot, bulk_modulus)
    return Material(delta, rho, snot, cnot, bulk_modulus, E = E, nu = nu, Gnot = Gnot, shear_modulus = shear_modulus )

def kalthoff3d(delta, rho_scale=1, K_scale=1, G_scale=1, Gnot_scale=1):

    """ 
    Silling (See also Guo Gao):
    For the EMU model of the Kalthoff-Winkler experiment, the material parameters corresponded to a
    Young’s modulus of 191 GPa, a Poisson ratio of 0.3, a mass density of 8000 kg/m
    , a fracture toughness of 90 MPa-m 1/2
    , and a tensile strength of 2000 MPa.

    Thickness 9 or 19 mm (Guo, Gao)
    The projectile length l = 100 mm and diameter d = 50 mm which equals to the distance between notches.

    From https://www.sciencedirect.com/topics/materials-science/maraging-steel:
    The material parameters of Maraging steel 18Ni(300), which are taken from Park et al. (2012), are as follows: Young's modulus E0 = 1.9 × 105 MPa, Poisson's ratio ν0 = 0.3, the mass density ρ = 8000 kg/m3, the failure strength ft = 2812.25 MPa, and the fracture energy Gc = 22.2 N/mm, resulting in Griffith's internal length lch = 0.53 mm. The Rayleigh wave speed is cR = 2745 m/s.
    """
    rho = 8000.0	* rho_scale
    ## what should be Gnot???  Keeping intact
    #Gnot = 22200
    # Gnot = 90
    # Gnot = 42.4 𝑀𝑃𝑎. 𝑚𝑚 = 42400 J/m^2
    Gnot = 42400 * Gnot_scale
    E = 190e9
    # nu = 0.3
    nu = 0.3
    # E = 9 * bulk_modulus * shear_modulus / ( 9 * bulk_modulus + shear_modulus);
    bulk_modulus = E/(3*(1- 2*nu))
    shear_modulus = E/(2*(1+nu))

    # cnot = 24 * E /( (1 - nu) * np.pi * (delta**3) );
    # snot = np.sqrt(4 * np.pi * Gnot /(9*E*delta));

    cnot = 18 * bulk_modulus / (np.pi * delta**4)
    snot = np.sqrt( Gnot / (3 * nu + (0.75**2) * ( bulk_modulus - 5 * shear_modulus/3))/delta)

    # print('nu = ', nu)
    # print('E = ', E)

    # return Material(delta, rho, snot, cnot, bulk_modulus)
    return Material(delta, rho, snot, cnot, bulk_modulus, E = E, nu = nu, Gnot = Gnot, shear_modulus = shear_modulus )
def kalthoff3d_beta(delta, rho_scale=1, K_scale=1, G_scale=1, Gnot_scale=1):

    """ 
    Silling (See also Guo Gao):
    For the EMU model of the Kalthoff-Winkler experiment, the material parameters corresponded to a
    Young’s modulus of 191 GPa, a Poisson ratio of 0.3, a mass density of 8000 kg/m
    , a fracture toughness of 90 MPa-m 1/2
    , and a tensile strength of 2000 MPa.

    Thickness 9 or 19 mm (Guo, Gao)
    The projectile length l = 100 mm and diameter d = 50 mm which equals to the distance between notches.

    From https://www.sciencedirect.com/topics/materials-science/maraging-steel:
    The material parameters of Maraging steel 18Ni(300), which are taken from Park et al. (2012), are as follows: Young's modulus E0 = 1.9 × 105 MPa, Poisson's ratio ν0 = 0.3, the mass density ρ = 8000 kg/m3, the failure strength ft = 2812.25 MPa, and the fracture energy Gc = 22.2 N/mm, resulting in Griffith's internal length lch = 0.53 mm. The Rayleigh wave speed is cR = 2745 m/s.
    """
    rho = 8000.0	* rho_scale
    ## what should be Gnot???  Keeping intact
    #Gnot = 22200
    # Gnot = 90
    # Gnot = 42.4 𝑀𝑃𝑎. 𝑚𝑚 = 42400 J/m^2
    Gnot = 90000 * Gnot_scale
    E = 190e9
    # nu = 0.3
    nu = 0.30
    # E = 9 * bulk_modulus * shear_modulus / ( 9 * bulk_modulus + shear_modulus);
    bulk_modulus = E/(3*(1- 2*nu))
    shear_modulus = E/(2*(1+nu))

    # cnot = 24 * E /( (1 - nu) * np.pi * (delta**3) );
    # snot = np.sqrt(4 * np.pi * Gnot /(9*E*delta));

    cnot = 18 * bulk_modulus / (np.pi * delta**4)
    snot = np.sqrt( Gnot / (3 * nu + (0.75**2) * ( bulk_modulus - 5 * shear_modulus/3))/delta)

    # print('nu = ', nu)
    # print('E = ', E)

    # return Material(delta, rho, snot, cnot, bulk_modulus)
    return Material(delta, rho, snot, cnot, bulk_modulus, E = E, nu = nu, Gnot = Gnot, shear_modulus = shear_modulus )


def peridem_deformable(delta, rho_scale=1, K_scale=1, G_scale=1, Gnot_scale=1):
    """ Smaller fracture toughness
    """
    bulk_modulus = 2.0e9 * K_scale
    shear_modulus = 1.33e+09 * G_scale
    rho=1200.0	* rho_scale
    Gnot = 135.0 * Gnot_scale

    nu = 1/3
    E = 9 * bulk_modulus * shear_modulus / ( 9 * bulk_modulus + shear_modulus);

    cnot = 24 * E /( (1 - nu) * np.pi * (delta**3) );
    snot = np.sqrt(4 * np.pi * Gnot /(9*E*delta));

    # print('nu = ', nu)
    # print('E = ', E)

    # return Material(delta, rho, snot, cnot, bulk_modulus)
    return Material(delta, rho, snot, cnot, bulk_modulus, E = E, nu = nu, Gnot = Gnot, shear_modulus = shear_modulus )

def peridem_softer_1(delta):
    """ Smaller fracture toughness
    """
    bulk_modulus = 2.0e9
    shear_modulus = 1.33e+09
    rho=1200.0	
    Gnot = 135.0/20

    #  nu = 0.2278
    nu = (3 * bulk_modulus - 2 * shear_modulus) / ( 2 * ( 3 * bulk_modulus + shear_modulus))
    #  E = 1.23e9
    E = 9 * bulk_modulus * shear_modulus / ( 9 * bulk_modulus + shear_modulus);

    cnot = 24 * E /( (1 - nu) * np.pi * (delta**3) );
    snot = np.sqrt(4 * np.pi * Gnot /(9*E*delta));

    # print('nu = ', nu)
    # print('E = ', E)

    # return Material(delta, rho, snot, cnot, bulk_modulus)
    return Material(delta, rho, snot, cnot, bulk_modulus, E = E, nu = nu, Gnot = Gnot, shear_modulus = shear_modulus )

def peridem_3d(delta):
  bulk_modulus = 2.e+09
  shear_modulus = 1.33e+09
  rho=1200
  nu = (3 * bulk_modulus - 2 * shear_modulus) / ( 2 * ( 3 * bulk_modulus + shear_modulus))
  E = 9 * bulk_modulus * shear_modulus / ( 9 * bulk_modulus + shear_modulus)

  ## %% 3d: From silling-askari
  cnot = 18 * bulk_modulus / (np.pi * delta**4)
  Gnot = 135
  snot = np.sqrt(5 * Gnot / (9 * bulk_modulus * delta))
  return Material(delta, rho, snot, cnot, bulk_modulus, E = E, nu = nu, Gnot = Gnot, shear_modulus = shear_modulus )

def impactor_2d(delta, rho_scale=1, K_scale=1, G_scale=1, Gnot_scale=1):

    """ 
    , a fracture toughness of 90 MPa-m 1/2
    , and a tensile strength of 2000 MPa.
    # Steel (Carbon Steel, SI Units)
    rho_steel = 7850.0      # kg/m³
    E_steel = 200e9         # Pa (200 GPa)
    nu_steel = 0.30         # Unitless
    Gc_steel = 50e3         # J/m² (50 kJ/m²)
        """
    rho = 8000.0	* rho_scale
    Gnot = 50000 * Gnot_scale   #--
    E = 210e9  #200Gpa
    nu = 0.34
    bulk_modulus = E/(3*(1- 2*nu))
    shear_modulus = E/(2*(1+nu))
    
    cnot = 18 * bulk_modulus / (np.pi * delta**4)
    snot = np.sqrt( Gnot / (3 * nu + (0.75**2) * ( bulk_modulus - 5 * shear_modulus/3))/delta)


    return Material(delta, rho, snot, cnot, bulk_modulus, E = E, nu = nu, Gnot = Gnot, shear_modulus = shear_modulus )



def plate_2d(delta, rho_scale=1, K_scale=1, G_scale=1, Gnot_scale=1):

    """ 
    , a fracture toughness of 90 MPa-m 1/2
    , and a tensile strength of 2000 MPa.
    material properties of PMMA (Polymethyl Methacrylate) plates9
    # PMMA (Acrylic Glass, SI Units)
    rho_pmma = 1180.0       # kg/m³ (density)
    E_pmma = 3.0e9          # Pa (3 GPa, elastic modulus)
    nu_pmma = 0.34          # Unitless (Poisson's ratio)
    Gc_pmma = 400.0         # J/m² (0.4 kJ/m², critical energy release rate) 

    """
    rho = 1180.0	* rho_scale
    Gnot = 400 * Gnot_scale   #--Critical Energy Release Rate (GcGc​)	0.3–0.5 kJ/m²
    E = 3e9  #2Gpa
    nu = 0.34
    bulk_modulus = E/(3*(1- 2*nu))
    shear_modulus = E/(2*(1+nu))


    cnot = 9 * bulk_modulus / (np.pi * delta**3)
    #snot = np.sqrt( Gnot / (3 * nu + (0.75**2) * ( bulk_modulus - 5 * shear_modulus/3))/delta)
    snot = np.sqrt((4*np.pi*Gnot)/(9*E*delta))

    return Material(delta, rho, snot, cnot, bulk_modulus, E = E, nu = nu, Gnot = Gnot, shear_modulus = shear_modulus )



def ottawa_sand_old2(delta):
    """
    Return Ottawa sand material properties with peridynamic parameters
    for a given horizon delta (in meters).

    Parameters
    ----------
    delta : float
        Peridynamic horizon [m]

    """
    # Grain-based elastic constants (quartz)
    rho = 2650.0              # bulk density [kg/m^3]
    nu = 0.25 #or 0.08                # Poisson's ratio
    
    #nu = 0.25
    E = 95.0e9                  # Young's modulus [Pa]
    K = 37e9                  # Bulk modulus [Pa]
    G = 44e9                  # Shear modulus [Pa]
    Gnot = 1.5              # Fracture energy [N/m]
    K = E / (3*(1 - 2*nu))
    # In Bond-Based (3D), nu is fixed at 0.25
    # Therefore, K = E / (3 * (1 - 2*0.25)) = E / 1.5
    #K = E / 1.5
    # Peridynamic parameters
    #cnot = 120 * E / ((1-nu)*np.pi * delta**4)
    # snot =100* np.sqrt(5 *np.pi* Gnot / (12 * E * delta))
    #
        #-------------paper------c0-----2
    cnot = 12.0 * E / (np.pi * delta**4 * (1.0 - 2.0*nu))
    

    #-------------paper------c0-----1
    #cnot = 12.0 * E / (np.pi * delta**4 * (1.0 - nu))
    print("k is ",K)
    # Critical bond strain (baseline)
    #snot = np.sqrt(5.0 * np.pi * Gnot / (9.0 * K * delta))
 
    s_tension_crit     = (1e0)*np.sqrt(5*np.pi*Gnot / (9*K*delta))
    #s_tension_crit     =1 
    s_compression_crit = -(1e0)* np.sqrt(5*np.pi*Gnot / (9*K*delta))
    #s_compression_crit = 100
   
    #cnot               = (2)*18*K / (np.pi * delta**4)    
    #snot               = np.sqrt(5*np.pi*Gnot / (9*K*delta))
    snot               = s_tension_crit 
    
    print(" we are using this")
    #return Material(delta, rho, snot, s_tension_crit, s_compression_crit, cnot, K, E = E, nu = nu, Gnot = Gnot, shear_modulus = G )
    return Material(
    delta=delta,
    rho=rho,
    snot=snot,
    s_tension_crit=s_tension_crit,
    s_compression_crit=s_compression_crit,
    cnot=cnot,
    bulk_modulus=K,
    E=E,
    nu=nu,
    Gnot=Gnot,
    shear_modulus=G
)

def ottawa_sand_old(delta):
    """
    Return Ottawa sand material properties with peridynamic parameters
    for a given horizon delta (in meters).

    Parameters
    ----------
    delta : float
        Peridynamic horizon [m]

    """
    # Grain-based elastic constants (quartz)
    rho = 2650.0              # bulk density [kg/m^3]
    #nu = 0.17 #or 0.08                # Poisson's ratio
    
    nu = 0.25
    E = 72e9                  # Young's modulus [Pa]
    #K = 37e9                  # Bulk modulus [Pa]
    G = 31e9                  # Shear modulus [Pa]
    Gnot = 135.0              # Fracture energy [N/m]
    #K = E / (3*(1 - 2*nu))
    # In Bond-Based (3D), nu is fixed at 0.25
    # Therefore, K = E / (3 * (1 - 2*0.25)) = E / 1.5
    K = E / 1.5
    # Peridynamic parameters
    # cnot = 12 * E / ((1-nu)*np.pi * delta**4)
    # snot =100* np.sqrt(5 *np.pi* Gnot / (12 * E * delta))
    #
    cnot = 18*K / (np.pi * delta**4)    
    #cnot = 9*E
    snot = np.sqrt(5*np.pi*Gnot / (9*K*delta))
    # --- NEW yield/plateau parameters ---
    use_yield_plateau_law =True 

    sigmaLp = 1.0e2   # [Pa] example: tension yield "stress-like" parameter
    sigmaLm = 1.0e6   # [Pa] example: compression yield parameter (often larger)

    rLp = sigmaLp / cnot
    rLm = sigmaLm / cnot
    rSp = 0.0   # tension softening start  (rS^+)
    rFp = 0.0   # tension fracture stretch (rF^+)
    rSm = 0.0   # compression softening start magnitude (rS^-)
    rFm = 0.0   # compression fracture stretch magnitude (rF^-)
    #------------------------------------------
    min_gap=5e-6 
    aS = 3.0
    aF = 6.0
    rSp = max(aS * rLp, rLp + min_gap)
    rFp = max(aF * rLp, rSp + min_gap)

    rSm = max(aS * rLm, rLm + min_gap)
    rFm = max(aF * rLm, rSm + min_gap)
    #----------------------------------------
    delta_y = 0.05 * rLp   # smoothness (5% of tensile yield stretch)

    return Material(delta, rho, snot, cnot, K,
                    E=E, nu=nu, Gnot=Gnot, shear_modulus=G,
                    rLp=rLp, rLm=rLm, delta_y=delta_y,
                    use_yield_plateau_law=use_yield_plateau_law)







def ottawa_sand(delta,plot=False, plot_path=None):
    """
    Return Ottawa sand material properties with peridynamic parameters
    for a given horizon delta (in meters).

    Parameters
    ----------
    delta : float
        Peridynamic horizon [m]
    """
    # Grain-based elastic constants (quartz)
    rho = 2650.0              # [kg/m^3]
    nu  = 0.17                # Bond-Based (3D) fixed
    E   = 72e9                # [Pa]
    G   = 31e9                # [Pa]
    Gnot = 135.0              # [N/m]

    # In Bond-Based (3D), nu=0.25 => K = E / (3*(1-2nu)) = E/1.5
    #K = E / 1.5               # [Pa]

    K = E / (3*(1 - 2*nu))
    # Peridynamic parameters (bond-based 3D)
    cnot = 18.0 * K / (np.pi * delta**4)
    snot = np.sqrt(5.0 * np.pi * Gnot / (9.0 * K * delta))

    # --- Yield/plateau + softening/break law ---
    use_yield_plateau_law = True 

    # "stress-like" calibration parameters (user-chosen)
    sigmaLp = 1.0e2   # [Pa] tension
    sigmaLm = 1.0e4   # [Pa] compression

    # Yield stretches
    # rLp = sigmaLp /E 
    # rLm = sigmaLm /E 
    rLp = 10*sigmaLp/cnot  
    rLm = sigmaLm /cnot 


    # Softening/break suggested from multiples of yield
    min_gap = 2*rLp 
    aS      = 3.0
    aF      = 6.0

    rSp = max(aS * rLp, rLp + min_gap)   # tension softening start
    rFp = max(aF * rLp, rSp + min_gap)   # tension fracture

    rSm = max(aS * rLm, rLm + min_gap)   # compression softening start (magnitude)
    rFm = max(aF * rLm, rSm + min_gap)   # compression fracture (magnitude)

    # Smoothness for yield->plateau transition
    # (guard against rLp being extremely small)
    delta_y = max(0.05 * rLp, 1e-12)
    
    # NOTE: you MUST pass s_tension_crit and s_compression_crit too
    # If you are using rFp/rFm as the actual break criteria in C++,
    # you can set these equal to rFp/rFm, or set them large and rely on rF.
    s_tension_crit = rFp
    s_compression_crit = rFm
    print('cnot',cnot)
    print('snot',snot)
    print('rLp',rLp)
    print('rLm',rLm)
    print(' rSp',rSp)
    print(' rSm',rSm)
    print(' rFp',rFp)
    print('rFp',rFp)
    mat =  Material(
        delta=delta,
        rho=rho,
        snot=snot,
        s_tension_crit=s_tension_crit,
        s_compression_crit=s_compression_crit,
        cnot=cnot,
        bulk_modulus=K,
        E=E, nu=nu, Gnot=Gnot, shear_modulus=G,
        rLp=rLp, rLm=rLm, delta_y=delta_y,
        rSp=rSp, rFp=rFp, rSm=rSm, rFm=rFm,
        use_yield_plateau_law=use_yield_plateau_law,
        name="OttawaSand"
    )
    plot = 1 
    if plot and mat.use_yield_plateau_law:
        mat.plot_constitutive_law(savepath=plot_path, show=(plot_path is None))

    return mat


#-----------------------------------------------------------------------

def ottawa_sand_1(delta):
    """
    Return material properties for peridynamic compression
    of a sphere by rigid plates, based on the reference example.

    Parameters
    ----------
    delta : float
        Peridynamic horizon [m]
    """
    """
    Minimum Index Density: ~1,480 kg/m³ (emax≈0.74)

    Maximum Index Density: ~1,760 kg/m³ (emin≈0.50)

    Typical Bulk Density: Often cited around 1,550 to 1,650 kg/m³ for medium-dense samples.
    Typical Values for Elastic Modulus (E)

    For standard geotechnical applications at moderate depths (confining pressure around 100 kPa), you can expect the following ranges:

    Loose State: 30 MPa to 50 MPa

    Medium Dense: 50 MPa to 80 MPa

    Dense State: 80 MPa to 120+ MPa
    """

    # -------------------------------
    # Material properties (elastic)
    # -------------------------------

    rho = 2650.0              # density [kg/m^3] (quartz-like)
    nu  = 0.25                # Poisson's ratio
    #E = 95.0e9                # bulk modulus [Pa]
    E = 75.0e9                # bulk modulus [Pa]
    #E=2*E 
    G = 44.0e9     # shear modulus [Pa]
    print("E is ", E)
    # -------------------------------
    # Fracture / damage parameters
    # -------------------------------
    #K = 37.0e9
    K = E / (3*(1 - 2*nu))
    Gnot = 1.0              # fracture energy [N/m] (kept consistent)
    # -------------------------------
    # Peridynamic parameters
    #-------------paper------c0-----2
    cnot = 12.0 * E / (np.pi * delta**4 * (1.0 - 2.0*nu))
    #-------------paper------c0-----1
    #cnot = 12.0 * E / (np.pi * delta**4 * (1.0 - nu))
    # Critical bond strain (baseline)
    snot =(1e-9)* np.sqrt(5.0 * np.pi * Gnot / (9.0 * K * delta))
    #snot = 0.007 
    # Asymmetric tension / compression limits
    s_tension_crit      =  snot
    s_compression_crit  =  snot


    return Material(
        delta=delta,
        rho=rho,
        snot=snot,
        s_tension_crit=s_tension_crit,
        s_compression_crit=s_compression_crit,
        cnot=cnot,
        bulk_modulus=K,
        E=E,
        nu=nu,
        Gnot=Gnot,
        shear_modulus=G
    )


#-----------------------------------------------------------------------





def sphere_contact_material(delta):
    """
    Return material properties for peridynamic compression
    of a sphere by rigid plates, based on the reference example.

    Parameters
    ----------
    delta : float
        Peridynamic horizon [m]
    """

    import numpy as np

    # -------------------------------
    # Material properties (elastic)
    # -------------------------------

    rho = 2650.0              # density [kg/m^3] (quartz-like)
    nu  = 0.25                # Poisson's ratio

    k = 12.5e9                # bulk modulus [Pa]
    E = 3.0 * (1.0 - 2.0*nu) * k   # Young's modulus [Pa]

    G = E / (2.0 * (1.0 + nu))     # shear modulus [Pa]
    print("E is ", E)
    # -------------------------------
    # Fracture / damage parameters
    # -------------------------------

    Gnot = 135.0              # fracture energy [N/m] (kept consistent)

    # -------------------------------
    # Peridynamic parameters
    # -------------------------------

    # Bulk modulus (explicit, for clarity)
    K = k
    #E = 2*E
    K = E / (3*(1 - 2*nu))
     
    # Bond stiffness
    #cnot = 18.0 * K / (np.pi * delta**4)
    
    
    #-------------paper------c0-----2
    cnot =3* 12.0 * E / (np.pi * delta**4 * (1.0 - 2.0*nu))
    

    #-------------paper------c0-----1
    #cnot = 12.0 * E / (np.pi * delta**4 * (1.0 - nu))
    print("k is ",K)
    # Critical bond strain (baseline)
    snot = np.sqrt(5.0 * np.pi * Gnot / (9.0 * K * delta))
    snot = 0.06
    # Asymmetric tension / compression limits
    s_tension_crit      =  snot
    s_compression_crit  =  snot

    print("Using sphere contact material (k = 12.5 GPa, nu = 0.35)")

    return Material(
        delta=delta,
        rho=rho,
        snot=snot,
        s_tension_crit=s_tension_crit,
        s_compression_crit=s_compression_crit,
        cnot=cnot,
        bulk_modulus=K,
        E=E,
        nu=nu,
        Gnot=Gnot,
        shear_modulus=G
    )

