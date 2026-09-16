import numpy as np
from scipy.optimize import root
from numba import jit, njit


##### Bare photon quantities ( symetrized propagators ) #####

def bareBosonDistribution(omega, delta , Gamma):
    return (omega**2 + delta**2+0.25*(Gamma**2))/(2*delta*omega)
 
def retarded_D(omega, delta, Gamma):
    return delta/( (omega +0.5j * Gamma)**2 - delta**2 )

def imagRetardedBoson(omega, delta , Gamma):
    return np.imag(retarded_D(omega,delta, Gamma))

##### Fermionic quantities ######

@njit()
def energy_perp(k_perp, t):
    return 0 # d = 1
 
@njit()
def energies_meanField(k_par, t, g, mu, phi): # k_par : np.array of shape (N_k)
    k_par = np.atleast_1d(k_par)
    barEps = - mu
    deltaEps = np.sqrt( 2*(t**2)*(1 + np.cos(k_par)) + (g*phi)**2  )
    out = np.zeros((k_par.shape[0], 2), dtype=float)
    out[:, 0] = barEps + deltaEps
    out[:, 1] = barEps - deltaEps
    return out

def alpha(k_par, t, g, phi):
    return 2 * g*phi /(np.sqrt( 2*(t**2)*(1 + np.cos(k_par)) + (g*phi)**2  )) - 1

def u_kz(k_par, t, g, phi):
    return  g*phi /(np.sqrt( 2*(t**2)*(1 + np.cos(k_par)) + (g*phi)**2  )) 
    
def bareFermionOccupation(epsilon, delta, Gamma):
    B = bareBosonDistribution(2*epsilon, delta , Gamma)
    # B =( np.tanh(2* omegas/ (2*T_LF) ))**(-1) #### Thermal 
    return B - np.sign(epsilon)*np.sqrt(B**2 - 1)
    # return (bareBosonDistribution(epsilon, delta , Gamma))**(-1)

def matrixEigenvaluesDistributionFunction(epsilon_plus, epsilon_minus, alpha, omega, delta, Gamma):
    imD_plus, imD_minus  = imagRetardedBoson(omega- epsilon_plus, delta , Gamma), imagRetardedBoson(omega- epsilon_minus, delta , Gamma)
    B_omega_epsPlus , B_omega_epsMinus = bareBosonDistribution(omega- epsilon_plus, delta , Gamma), bareBosonDistribution(omega- epsilon_minus, delta , Gamma)
    B_epsPlus , B_epsMinus = bareBosonDistribution( epsilon_plus, delta , Gamma), bareBosonDistribution( epsilon_minus, delta , Gamma)

    a = imD_plus *( B_omega_epsPlus + (B_epsPlus**(-1)) ) + imD_minus *( B_omega_epsMinus + (B_epsMinus**(-1)) ) 
    b = alpha * (imD_plus *( B_omega_epsPlus + (B_epsPlus**(-1)) ) - imD_minus *( B_omega_epsMinus + (B_epsMinus**(-1)) ) )
    c = imD_plus *( B_omega_epsPlus * (B_epsPlus**(-1)) +1 ) + imD_minus *( B_omega_epsMinus * (B_epsMinus**(-1)) +1 ) 
    d = alpha * (imD_plus *( B_omega_epsPlus * (B_epsPlus**(-1))+1) - imD_minus *( B_omega_epsMinus * (B_epsMinus**(-1)) +1 ) )

    f,g = (a*c - b*d)/(a**2 - b**2), (a*d - b*c)/(a**2 - b**2) 
    return f,g


##### Mean-field equations #####


def responseAtZeroFreq(t, delta, Gamma, distrib = 'Boson', N_int = 100):
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
    # Construct, at a fixed values of the parameters t, g and bar_n, the function which to phi and mu returns the two mean-field equations Eq. 3.69 a-b  .
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
        return [eq_one, mu**2]
    return equations # func

def solve_meanFieldEquations(t, g, delta, Gamma, bar_n, initial_mu = 0, initial_phi = 1e-5, distrib = 'cavity', verbose = True):
    equationsOfMeanField = meanFieldEquations(t, g, delta, Gamma, bar_n, distrib = distrib)
    initial_guess = [initial_mu , initial_phi]
    sol = root( equationsOfMeanField, initial_guess) # instance of the OptimizeResult class of scipy.
    if verbose : 
        print(sol.message)
    return sol.x

def meanFieldEquations_FixedOccupations(k_grid, Fs, t, g, delta, Gamma, bar_n):
    def equations(vars): # vars = [mu, phi]
        mu, phi = vars
        mu = 0 
        epsilons = energies_meanField(k_grid, t, g, mu, phi) # (N_int,2)
        integralOne, integralTwo = integralsInMeanFieldEquations( epsilons, Fs )
        eq_one = phi - ( (2*delta*(g**2)*phi)/((delta**2) + (Gamma**2)/4 ) ) * integralOne
        eq_two = 1 - bar_n - integralTwo
        return [eq_one, eq_two]
    return equations # func

def solve_meanFieldEquations_FixedOccupations(k_grid, Fs, t, g, delta, Gamma, bar_n, initial_mu = 0, initial_phi = 1e-5, verbose = True):
    equationsOfMeanField = meanFieldEquations_FixedOccupations(k_grid, Fs, t, g, delta, Gamma, bar_n)
    initial_guess = [initial_mu , initial_phi]
    sol = root( equationsOfMeanField, initial_guess) # instance of the OptimizeResult class of scipy.
    if verbose : 
        print(sol.message)
    return sol.x

##### RPA ###### Programmed in a brute force way

# @njit()
def responseRetarded_fromF(omega, F, k_grid, g, phi, t, eta = 1e-4):
    """
    F : np.array of shape (N, 2) containing the F_k,± so that F[i, 0] = F_{k_grid[i], +} and F[i, 1] = F_{k_grid[i], -}
    k_grid : np.array of shape (N,)
    omega : np.array of arbitrary shape

    returns : PiR the retarded greens function for every omega.
    """
    epsilons = energies_meanField(k_grid, t, g, 0, phi)
    ukz =2*g*phi / ( epsilons[:, 0] - epsilons[:, 1] )
    numerator = - 0.5*( F[:, 0] - F[:, 1] )*( epsilons[:, 0] - epsilons[:, 1] )*( 1 -  ukz )
    denominator = (omega[..., np.newaxis] + 1j*eta)**2 - ( epsilons[:, 0] - epsilons[:, 1] )**2  # (omega.shape , N)
    return np.sum( numerator/denominator, axis=-1  )

# @njit()
def responseKeldysh_fromF(omega, F, k_grid, g, phi, t, eta = 1e-4):
    """
    F : np.array of shape (N, 2) containing the F_k,± so that F[i, 0] = F_{k_grid[i], +} and F[i, 1] = F_{k_grid[i], -}
    k_grid : np.array of shape (N,)
    omega : np.array of arbitrary shape

    returns : PiK the retarded greens function for every omega.
    """
    epsilons = energies_meanField(k_grid, t, g, 0, phi)
    ukz =2*g*phi / ( epsilons[:, 0] - epsilons[:, 1] )
    numerator = 0.5*( F[:, 0] * F[:, 1] - 1 )*( epsilons[:, 0] - epsilons[:, 1] )*( 1 -  ukz )
    denominator = (omega[..., np.newaxis] + 1j*eta)**2 - ( epsilons[:, 0] - epsilons[:, 1] )**2  # (omega.shape , N)
    return 0.5j * np.imag( np.sum( numerator/denominator, axis=-1  )) 

# @njit()
def distributionResponseFunction(omega, F, k_grid, g, phi, t):
    return np.array( responseKeldysh_fromF(omega, F, k_grid, g, phi, t, eta = 1e-4)/ (2j*np.imag(responseRetarded_fromF(omega, F, k_grid, g, phi, t, eta = 1e-4))), dtype = float) # real

# @njit()
def computeRPA_boson(omega, F, k_grid, g, phi,t, delta , Gamma, eta = 1e-1 , thermal = False):
    N = k_grid.shape[0]
    if thermal == False : 
        Bzero = bareBosonDistribution(omega, delta, Gamma)
    else : 
        T_LF = (1/(4*delta))*(delta**2 + (Gamma**2 / 4))
        Bzero = (np.tanh(0.5 * omega / T_LF))**(-1)
    imPiR = (1/N)* np.imag(responseRetarded_fromF(omega, F, k_grid, g, phi, t, eta = eta))
    piK =  (1/N)*responseKeldysh_fromF(omega, F, k_grid, g, phi, t, eta = eta)
    B = Bzero - (Bzero  - ( piK / (2j*imPiR)) )* ( ((g**2)* imPiR)/((Gamma * omega/ delta) + (g**2)* imPiR ) )
    return B

# @njit()
def updateFermionsOccupations( F, k_grid, g, phi, mu, t, delta , Gamma ):
    epsilons = energies_meanField(k_grid, t, g, mu, phi)
    B = computeRPA_boson(epsilons, F, k_grid, g, phi,t, delta , Gamma )
    F_updated = (B)**(-1)
    return np.array(F_updated, dtype = float)



def loop_RPA_updates(k_grid, g, t, delta , Gamma, init_F, initial_phi, initial_mu , precision = 1e-3, maxLoop = 100): # brute force and praying for convergence
    N = k_grid.size

    phi_n, mu_n = solve_meanFieldEquations(t, g, delta, Gamma, 1, initial_mu = initial_mu, initial_phi = initial_phi, distrib = 'cavity', verbose = False)
    # epsilons = energies_meanField(k_grid, t, g, mu_n, phi_n)
    # F_n = bareFermionOccupation(epsilons, delta, Gamma)
    F_n = init_F

    error = 1
    error_mem = []


    while error > precision and len(error_mem) < maxLoop :
        print('Loop n°'+str(len(error_mem)))
        F_nPlusOne = updateFermionsOccupations(F_n, k_grid, g, phi_n, mu_n, t, delta , Gamma)
        init_phi = max(np.abs(phi_n), 1e0)
        mu_nPlusOne, phi_nPlusOne = solve_meanFieldEquations_FixedOccupations(k_grid, F_nPlusOne, t, g, delta, Gamma, 1, initial_mu = 0, initial_phi = init_phi, verbose = False)

        error = np.linalg.norm(F_nPlusOne - F_n) / np.linalg.norm(F_n)
        print(error)
        error_mem.append(error_mem)
        F_n, mu_n, phi_n = F_nPlusOne, mu_nPlusOne, phi_nPlusOne 
    
    if error > precision : 
        print('Convergence failed.')
    else : 
        print('Convergence succeeded.')

    return phi_n, mu_n, F_n , error_mem
