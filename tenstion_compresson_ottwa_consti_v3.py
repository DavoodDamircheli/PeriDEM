import numpy as np

def smoothstep_c1(t):
    """C1 smoothstep: 1 at t=0, 0 at t=1, zero slopes at both ends."""
    t = np.clip(t, 0.0, 1.0)
    return 1.0 - 3.0*t*t + 2.0*t*t*t

def branch_positive(r, mu, rL, rS, rF, delta):
    """
    r >= 0 branch:
      linear -> smooth yield-to-plateau -> smooth softening -> 0
    """
    # plateau level (matches your old law saturation)
    F = mu * (rL + delta)

    if r <= 0.0:
        return 0.0
    if r <= rL:
        return mu * r

    # yield-to-plateau (smooth, saturating)
    # f_y(r) = F - mu*delta*exp(-(r-rL)/delta)
    if r <= rS:
        return F - mu * delta * np.exp(-(r - rL) / delta)

    # softening to zero at rF, using value at rS as the "start level"
    if r < rF:
        fS = F - mu * delta * np.exp(-(rS - rL) / delta)  # value at start of softening
        t = (r - rS) / (rF - rS)
        return fS * smoothstep_c1(t)

    return 0.0

def f_redesigned(r, mu,
                 rL_pos, rS_pos, rF_pos,
                 rL_neg, rS_neg, rF_neg,
                 delta):
    """
    Full asymmetric law, no f0.
    Compression side uses magnitude a=-r and returns negative force.
    """
    if r >= 0.0:
        return branch_positive(r, mu, rL_pos, rS_pos, rF_pos, delta)
    else:
        a = -r
        return -branch_positive(a, mu, rL_neg, rS_neg, rF_neg, delta)

def f_redesigned_vec(r_arr, **p):
    r_arr = np.asarray(r_arr, dtype=float)
    return np.array([f_redesigned(x, **p) for x in r_arr])
import numpy as np
import matplotlib.pyplot as plt

# ---- assumes f_redesigned(...) and f_redesigned_vec(...) are already defined ----

def plot_constitutive_law(params, rmin=None, rmax=None, npts=1200):
    """
    Plot redesigned constitutive law f(r) with markers for
    rL, rS, rF (tension & compression).

    Parameters
    ----------
    params : dict
        Must contain:
          mu, delta
          rL_pos, rS_pos, rF_pos
          rL_neg, rS_neg, rF_neg
    rmin, rmax : float (optional)
        Plot range. If None, chosen automatically from rF.
    npts : int
        Number of sample points.
    """

    # auto domain
    if rmin is None:
        rmin = -1.15 * params["rF_neg"]
    if rmax is None:
        rmax =  1.15 * params["rF_pos"]

    r = np.linspace(rmin, rmax, npts)
    f = f_redesigned_vec(r, **params)

    plt.figure(figsize=(9, 5))
    plt.plot(r, f, "k", lw=2.2, label=r"$f(r)$")

    # ---------- vertical markers ----------
    def vline(x, label, color, style):
        plt.axvline(x, color=color, linestyle=style, lw=1.6, label=label)

    # tension
    vline(params["rL_pos"], r"$r_L^+$ (yield)", "tab:blue", "--")
    vline(params["rS_pos"], r"$r_S^+$ (softening)", "tab:green", ":")
    vline(params["rF_pos"], r"$r_F^+$ (fracture)", "tab:red", "-.")

    # compression
    vline(-params["rL_neg"], r"$-r_L^-$ (yield)", "tab:blue", "--")
    vline(-params["rS_neg"], r"$-r_S^-$ (softening)", "tab:green", ":")
    vline(-params["rF_neg"], r"$-r_F^-$ (fracture)", "tab:red", "-.")

    # ---------- cosmetics ----------
    plt.axhline(0, color="gray", lw=0.8)
    plt.axvline(0, color="gray", lw=0.8)

    plt.xlabel("stretch $r$")
    plt.ylabel("force $f(r)$")
    plt.title("Redesigned Constitutive Law: Elastic → Yield → Softening → Break")
    plt.grid(True, alpha=0.3)

    # avoid duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    plt.legend(uniq.values(), uniq.keys(), fontsize=9, loc="upper left")

    plt.tight_layout()
    plt.show()
params = dict(
    mu=10.0,
    delta=0.005,

    rL_pos=0.02, rS_pos=0.04, rF_pos=0.08,
    rL_neg=0.035, rS_neg=0.06, rF_neg=0.09,
)

#plot_constitutive_law(params)

