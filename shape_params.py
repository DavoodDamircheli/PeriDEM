import numpy as np
from scipy import optimize

class Param(object):
    def __init__(self, theta):
        self.theta = theta
        self.phi = phi(theta)
        self.tau = tau(theta)
        self.c = csolve(theta)
        self.c_r = c_r(theta)
    def print_latex(self):
        print(
                "{:.1f}".format(self.theta), "&",
                "{:.4f}".format(phi(self.theta)), "&",
                "{:.4f}".format(tau(self.theta)), "&",
                "{:.4f}".format(csolve(self.theta)), "&",
                "{:.4f}".format(c_r(self.theta))
                )
        
# pertdisk
def phi(theta):
    return np.sqrt(8 * theta / np.pi) - 1
# ring
def tau(theta):
    return np.sqrt(1 - 2 * theta/np.pi)
# plus
def csolve(theta):
    def cfun(c):
        return c**2 * ( 1  + 4 * np.sqrt(2/(c**2) - 1) ) - 2*theta
    sol = optimize.root(cfun, [0.5], method='hybr')
    # print(sol.x)
    return sol.x[0]

# L_symm
def c_r(theta):
    return 1 - np.sqrt(1 - theta)

# theta_list = np.linspace(0.1, 1, num=10, endpoint=True)

# print(
        # "theta", "&",
        # "phi", "&",
        # "tau", "&",
        # "c", "&",
        # "c_r",
        # )
# for theta in theta_list:
    # print(
            # "{:.1f}".format(theta), "&",
            # "{:.4f}".format(phi(theta)), "&",
            # "{:.4f}".format(tau(theta)), "&",
            # "{:.4f}".format(csolve(theta)), "&",
            # "{:.4f}".format(c_r(theta))
            # )
