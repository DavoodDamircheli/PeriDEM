import numpy as np

class Material(object):
    """docstring for Material"""
    def __init__(self, delta, rho, snot, cnot, bulk_modulus, E = None, nu = None, Gnot = None, shear_modulus = None, name=None):
        super(Material, self).__init__()
        self.delta  = delta
        self.rho    = rho     
        self.snot   = snot
        self.cnot   = cnot
        self.bulk_modulus = bulk_modulus

        self.E = E
        self.nu = nu
        self.Gnot = Gnot
        self.shear_modulus = shear_modulus

        self.name = name

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



    def print(self):
        """print info
        """
        print('delta: ', self.delta)
        print('rho: ', self.rho)
        print('cnot: ', self.cnot)
        print('snot: ', self.snot)
        print('E: ', self.E)
             

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



def ottawa_sand(delta):
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
    nu = 0.17 #or 0.08                # Poisson's ratio
    E = 72e9                  # Young's modulus [Pa]
    K = 37e9                  # Bulk modulus [Pa]
    G = 31e9                  # Shear modulus [Pa]
    Gnot = 135.0              # Fracture energy [N/m]

    # Peridynamic parameters
    cnot = 18 * K / (np.pi * delta**4)
    snot = np.sqrt(5 * Gnot / (9 * K * delta))

        

    return Material(delta, rho, snot, cnot, K, E = E, nu = nu, Gnot = Gnot, shear_modulus = G )


