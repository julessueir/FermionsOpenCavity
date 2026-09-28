"""
Free-fermion (U = 0) mean-field theory for a 1D tight-binding chain coupled
to a single driven-dissipative cavity mode.

Companion to the full Keldysh Dyson equation code in ../LowU: same physical setup
(2-site unit cell, hopping t, cavity detuning delta and linewidth Gamma,
light-matter coupling g, order parameter phi playing the role of the
Hartree field g*phi*tau_z), but here restricted to the interaction-free
limit, where the mean-field equations for (mu, phi) can be solved directly
without a self-consistent Green's-function loop. Used to produce the
figures in plots.ipynb.
"""

import numpy as np
from scipy.optimize import root
from numba import jit, njit

#===========================================================#
##### Bare photon quantities ( symetrized propagators ) #####
#===========================================================#

def bareBosonDistribution(omega, delta , Gamma): # Symetrized boson distribution function \tilde{B}(\omega) defined in Eq. (12). 
    return (omega**2 + delta**2+0.25*(Gamma**2))/(2*delta*omega)

def retarded_D(omega, delta, Gamma): # Retarded cavity propagator \tilde{D}^R(\omega) defined in Eq. (10).
    return delta/( (omega +0.5j * Gamma)**2 - delta**2 )

def imagRetardedBoson(omega, delta , Gamma):
    return np.imag(retarded_D(omega,delta, Gamma))

##### Fermionic quantities ######

@njit()
def energies_meanField(k_par, t, g, mu, phi): # k_par : np.array of shape (N_k)
    # Two mean-field quasiparticle bands eps_{k,+/-} define in Eq. (25), where g*phi is the Hartree term.
    k_par = np.atleast_1d(k_par)
    barEps = - mu
    deltaEps = np.sqrt( 2*(t**2)*(1 + np.cos(k_par)) + (g*phi)**2  )
    out = np.zeros((k_par.shape[0], 2), dtype=float)
    out[:, 0] = barEps + deltaEps  # eps_{k,+}
    out[:, 1] = barEps - deltaEps  # eps_{k,-}
    return out

def u_kz(k_par, t, g, phi):
    # h_z / |h_k| = g*phi / deltaEps: z-projection of h_k defined in Eq. (24). 
    return  g*phi /(np.sqrt( 2*(t**2)*(1 + np.cos(k_par)) + (g*phi)**2  ))

def bareFermionOccupation(epsilon, delta, Gamma):
    # Ftilde(epsilon): the analytic solution of the kinetic equation defined in Eq. (34).  
    B = bareBosonDistribution(2*epsilon, delta , Gamma)
    return B - np.sign(epsilon)*np.sqrt(B**2 - 1)


#==============================#
##### Mean-field equations #####
#==============================#


def responseAtZeroFreq(t, delta, Gamma, distrib = 'Boson', N_int = 100):
    # Static (zero-frequency) linear response integral
    # (1/N) * sum_k (F_{k,+} - F_{k,-}) / (eps_{k,+} - eps_{k,-}) at phi -> 0,
    # i.e. the coefficient that controls the linear stability of phi = 0 in
    # meanFieldEquations below. Feeds into criticalCouplingLM.
    N = 200
    k_par = np.linspace(-np.pi, np.pi, num = N_int+1, endpoint = False)[1:] # (N_int,2)
    epsilons = energies_meanField(k_par, t, 0, 0, 0) # (N_int,2)
    if distrib == 'Boson':
        Fs = bareFermionOccupation(epsilons, delta, Gamma) # (N_int,2) Bosonic
    elif distrib == 'zero temp':
        Fs = np.sign(epsilons) # (N_int,2)    T = 0
    elif distrib == 'LFET':
        T_LF = 0.25*(delta**2 + 0.25*Gamma**2) /delta
        Fs = np.tanh(epsilons/(2*T_LF))
    integrandOne = (Fs[:, 0] - Fs[:, 1])/(epsilons[:,0] - epsilons[:,1] + 1e-10)
    integralOne = (1/N_int) * np.sum(integrandOne)
    return  integralOne

def criticalCouplingLM(t, delta, Gamma, distrib = 'Boson'):
    # Critical coupling g_c at which phi = 0 becomes linearly unstable,
    # obtained by linearizing eq_one in meanFieldEquations around phi = 0.
    integral = responseAtZeroFreq(t, delta, Gamma, distrib = distrib)
    g_c = np.sqrt( ((delta**2) + (Gamma**2)/4 )  / ( (2*delta) *  integral )  )
    return g_c


@njit()
def integralsInMeanFieldEquations( epsilons, Fs ):
    """
    Fs : np.array of shape (N, 2) such that F[i, 0] = F(eps_{k[i], +}) and  F[i, 1] = F(eps_{k[i], -}) ,
    epsilons : np.array of shape (N, 2) such that epsilons[i, 0] = eps_{k[i], +} and  epsilons[i, 1] = eps_{k[i], -}
    Returns the integral which appear in the mean-field equations
    """
    integrandOne = (Fs[:, 0] - Fs[:, 1])/(epsilons[:,0] - epsilons[:,1])
    integrandTwo = (Fs[:, 0] + Fs[:, 1])/2
    N_int  = epsilons.shape[0]
    integralOne, integralTwo = (1/N_int) * np.sum(integrandOne), (1/N_int) * np.sum(integrandTwo)
    return integralOne, integralTwo

def meanFieldEquations(t, g, delta, Gamma, bar_n, distrib = 'cavity', N_int = 100): # distrib = 'cavity', 'LFET', 'zero temp', float
    # Construct, at a fixed values of the parameters t, g and bar_n, the function which to phi and mu returns the two mean-field equations.
    def equations(vars): # vars = [mu, phi]
        mu, phi = vars
        k_par = np.linspace(-np.pi, np.pi, num = N_int+1, endpoint= False)[1:] # (N_int,2)
        epsilons = energies_meanField(k_par, t, g, mu, phi) # (N_int,2)
        if distrib == 'cavity':
            Fs = bareFermionOccupation(epsilons, delta, Gamma) # (N_int,2) Bosonic
        elif distrib == 'zero temp':
            Fs = np.sign(epsilons) # (N_int,2)    T = 0
        elif distrib == 'LFET':
            T_LF = 0.25*(delta**2 + 0.25*Gamma**2) /delta
            Fs = np.tanh(epsilons/(2*T_LF)) # (N_int,2)    T = LFET
        elif isinstance(distrib, (float, int)) :
            T = float(distrib)
            Fs = np.tanh(epsilons/(2*T)) # (N_int,2)    T = LFET
        integralOne, integralTwo = integralsInMeanFieldEquations( epsilons, Fs )
        eq_one = phi - ( (2*delta*(g**2)*phi)/((delta**2) + (Gamma**2)/4 ) ) * integralOne
        eq_two = 1 - bar_n - integralTwo
        # eq_two is computed but not used: at half filling (bar_n = 1), the
        # density equation is solved trivially by mu = 0 (particle-hole
        # symmetry), so mu**2 = 0 is used as the second equation instead.
        # Only valid at bar_n = 1.
        return [eq_one, mu**2]
    return equations # func

def solve_meanFieldEquations(t, g, delta, Gamma, bar_n, initial_mu = 0, initial_phi = 1e-5, distrib = 'cavity', verbose = True):
    equationsOfMeanField = meanFieldEquations(t, g, delta, Gamma, bar_n, distrib = distrib)
    initial_guess = [initial_mu , initial_phi]
    sol = root( equationsOfMeanField, initial_guess) # instance of the OptimizeResult class of scipy.
    if verbose :
        print(sol.message)
    return sol.x
