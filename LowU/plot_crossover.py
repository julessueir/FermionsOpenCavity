#!/usr/bin/env python3
"""
Plot crossover results from crossover_data.npz.
Run simulate_crossover.py first to generate the data.
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.rcParams.update({
    "text.usetex": True,            # Enable LaTeX rendering
    "font.family": "serif",         # Use a serif font
    "font.serif": ["Computer Modern"],  # Specify LaTeX font
    "text.latex.preamble": r"\usepackage{amsmath}" ,  # Optional: Load additional LaTeX packages
    'font.size': 18
})

# ================================================================
# Load data
# ================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
data = np.load(os.path.join(script_dir, "crossover_data.npz"))

omega = data["omega"]
U_values = data["U_values"]
GR = data["GR"]  # shape (n_U, n_omega, 2, 2)
GL = data["GL"]

g_eff_sq = float(data["param_g_eff_sq"])
t_hop = float(data["param_t_hop"])
delta_cav = float(data["param_delta_cav"])
Gamma_cav = float(data["param_Gamma_cav"])
g_cav = float(data["param_g_cav"])
beta = float(data["param_beta"])
T_LF = (1/(4*delta_cav))*(delta_cav**2 + (Gamma_cav**2 / 4))

def B(omega, delta, Gamma):
    return (omega**2 + delta**2 + 0.25* (Gamma**2))/(2*delta*omega)

def tildeF(omega, delta, Gamma):
    return np.sign(omega)* (B(2* np.abs(omega), delta, Gamma) - np.sqrt( (B(2* np.abs(omega), delta, Gamma))**2 - 1))

# ================================================================
# Derived quantities
# ================================================================


def distribution_function(GR_aa, GL_aa):
    spectral = -2.0 * GR_aa.imag
    mask = np.abs(spectral) > 1e-12
    F = np.full_like(spectral, np.nan)
    F[mask] = -GL_aa[mask].imag / spectral[mask]
    return F

def projector_largest_eigenvalue(H):
    """
    Returns the projector onto the eigenvector corresponding
    to the largest eigenvalue of a 2x2 Hermitian matrix H.
    """
    # Check Hermitian (optional but good practice)
    if not np.allclose(H, H.conj().T):
        raise ValueError("Matrix is not Hermitian")

    # Eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    # Index of largest eigenvalue
    idx = np.argmax(eigenvalues)

    # Corresponding eigenvector
    v = eigenvectors[:, idx]

    # Projector |v><v|
    P = np.outer(v, v.conj())

    return P

def distribution_functionMatrix(GR, GL):
    SpectralMatrix = - 2j * ( GR - GR.conj().T)
    P = projector_largest_eigenvalue(SpectralMatrix)
    GL_proj = np.linalg.trace(P@GL).imag
    Spectral_proj = np.linalg.trace(P@SpectralMatrix)
    return - GL_proj/Spectral_proj

# ================================================================
# Colormap: coolwarm, normalized by U^2 / g_eff_sq
# ================================================================

# ratio_values = U_values**2 / ( g_eff_sq )if g_eff_sq > 0 else U_values * 0 #  * t_hop / delta_cav 
ratio_values = U_values/ (np.sqrt( g_eff_sq) )if g_eff_sq > 0 else U_values * 0 #  * t_hop / delta_cav 
ratio_min, ratio_max = ratio_values.min(), ratio_values.max()

cmap = plt.cm.coolwarm
norm = mcolors.Normalize(vmin=ratio_min, vmax=ratio_max)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

title = (
    rf"$t={t_hop},\ \delta={delta_cav},"
    rf"\ \Gamma={Gamma_cav},\ g={g_cav},"
    rf"\ g^2/L^d={g_eff_sq},\ \beta={beta}$"
)

# ================================================================
# Figure 1: Distribution function F_00(w)
# ================================================================

omegaMax = 8 # Sets the window to plot
plot_components = False # whether to plot the (Pauli matrix) x,y,z components of the distribution function

fig1, ax1 = plt.subplots(figsize=(9, 5))
for i, U in enumerate(U_values):
    # F00 = 0.5 *(distribution_function(GR[i, :, 1, 0], GL[i, :, 1, 0]) - (distribution_function(GR[i, :, 0, 1], GL[i, :, 0, 1]) ))
    # ax1.plot(omega, 1+2* F00, color=cmap(norm(ratio_values[i])))
    F_0 = ( GL[i,:,1, 1].imag + GL[i,:,0, 0].imag ) / ( GR[i,:,0, 0].imag +  GR[i,:,1, 1].imag ) # Trace
    ax1.plot(omega, 1+ F_0, color=cmap(norm(ratio_values[i])))
    if plot_components :
        F_x = ( GL[i,:,1, 0].imag + GL[i,:,0, 1].imag ) / ( GR[i,:,0, 0].imag +  GR[i,:,1, 1].imag )
        F_y = ( GL[i,:,1, 0].imag - GL[i,:,0, 1].imag ) / ( GR[i,:,0, 0].imag +  GR[i,:,1, 1].imag )
        F_z = ( GL[i,:,1, 1].imag - GL[i,:,0, 0].imag ) / ( GR[i,:,0, 0].imag +  GR[i,:,1, 1].imag )
        ax1.plot(omega, F_x, ls = ':', label = 'x')
        ax1.plot(omega, F_y, ls = ':', label = 'y')
        ax1.plot(omega, F_z, ls = ':', label = 'z')
betaTest =  0.43 #1/(1.72657512659324* T_LF) #0.41 # T_Q = 1/(1.72657512659324* T_LF)
ax1.plot(omega,np.tanh(betaTest * omega * 0.5), ls =':', color= 'green', lw = 1.5)
ax1.plot(omega[omega != 0], tildeF(omega[omega != 0], delta_cav, Gamma_cav), linestyle = ':', color = 'black')
ax1.axvline(x = 2 * t_hop, ls = '--', color = 'gray', lw = 1)
ax1.axvline(x = -2 * t_hop, ls = '--', color = 'gray',  lw = 1)
ax1.annotate(r'$2J$', (2*t_hop+ .05,- 1.05), color = 'gray', rotation = -90, va = 'bottom')
ax1.annotate(r'$-2J$', (-(2*t_hop+.05),- 1.05), color = 'gray', rotation = -90, va = 'bottom')
# ax1.plot(omega, np.tanh(omega /(2*T_LF) ), linestyle = ':', color = 'black')
ax1.set_xlabel(r"$\omega/\delta$", fontsize=24)
ax1.set_yticks([-1, 0, 1])
ax1.set_yticklabels(['-1', '0', '1'])
ax1.set_xticks([-delta_cav, 0, delta_cav])
ax1.set_xticklabels(['-1', '0', '1'])
# ax1.set_ylabel(r"$F_{00}(\omega)$", fontsize=13)
ax1.set_ylabel(r"$F_{\rm loc}(\omega)$", fontsize=24)
# ax1.set_title("Keldysh distribution (sublattice 0)", fontsize=24)
if plot_components:
    ax1.legend()
ax1.set_xlim(-omegaMax,omegaMax)
ax1.set_ylim(-1.1, 1.1)
cbar1 = fig1.colorbar(sm, ax=ax1, shrink=0.9, pad=0.02)
# cbar1.set_label(r"$U^2 / (g^2/ L^d)$", fontsize=24)
cbar1.set_label(r"$U / (g/\sqrt{ L^d })$", fontsize=24)
# fig1.suptitle(title, fontsize=12)
fig1.tight_layout()
fig1.savefig(os.path.join(script_dir, "cavity_crossover_U_distribution.pdf"), transparent = True, dpi=400)
fig1.savefig(os.path.join(script_dir, "cavity_crossover_U_distribution.png"), transparent = True, dpi=400)

# ================================================================
# Figure 2: Spectral function A_00(w)
# ================================================================

def Gloc0_retared(omegas, J, n_k, Gamma):
    k = np.linspace(0, np.pi, num = n_k+1, endpoint = False)[1:]
    G_k_inv = np.zeros((omegas.shape[0],n_k, 2, 2), dtype = complex)
    G_k = np.zeros((omegas.shape[0],n_k, 2, 2), dtype = complex)
    G_k_inv[:,:, 0, 1]  = J*(1+ np.exp(-1j*k)[np.newaxis, :]) 
    G_k_inv[:,:, 1, 0]  = J*(1+ np.exp(1j*k)[np.newaxis, :]) 
    G_k_inv[:,:, 0, 0]  = omegas[:,np.newaxis] + np.zeros((n_k))[np.newaxis, :] + 1j* Gamma
    G_k_inv[:,:, 1, 1]  = omegas[:,np.newaxis] + np.zeros((n_k))[np.newaxis, :] +  1j* Gamma
    det = G_k_inv[:,:,0,0]* G_k_inv[:,:,1,1] - G_k_inv[:,:,1,0]*G_k_inv[:,:,0,1]
    G_k[:,:, 0, 0] = G_k_inv[:,:, 0, 0]/det
    G_k[:,:, 1, 1] = G_k_inv[:,:, 1, 1]/det
    G_k[:,:, 0, 1] = -G_k_inv[:,:, 1, 0]/det
    G_k[:,:, 1, 0] = -G_k_inv[:,:, 0, 1]/det
    Gloc = (1/n_k)* np.sum(G_k , axis = 1)
    return Gloc




fig2, ax2 = plt.subplots(figsize=(9, 5))
for i, U in enumerate(U_values):
    # A00 = -2.0 * GR[i, :, 0,1].imag
    # ax2.plot(omega, A00, color=cmap(norm(ratio_values[i])))
    A_0 = -1.0 * (GR[i, :, 1,1].imag + GR[i, :, 0,0].imag)
    ax2.plot(omega, A_0, label='0')
    if plot_components :
        A_z = -1.0 * (GR[i, :, 1,1].imag - GR[i, :, 0,0].imag)
        A_x = -1.0 * (GR[i, :, 0,1].imag + GR[i, :, 1,0].imag)
        A_y = -1.0 * (GR[i, :, 0,1].imag - GR[i, :, 1,0].imag)
        ax2.plot(omega, A_x, label='x')
        ax2.plot(omega, A_y, label='y')
        # ax2.plot(omega, A_z, label='z')

omegas = np.linspace(-omegaMax,omegaMax, num = 300)
n_k = 300
Gref = Gloc0_retared(omegas, t_hop, n_k, 0.1)
A_0 = -1.0 * (Gref[:, 1,1].imag + Gref[:, 0,0].imag)
ax2.plot(omegas, A_0, label='0', linestyle =':')
if plot_components :
    A_z = -1.0 * (Gref[:, 1,1].imag - Gref[:, 0,0].imag)
    A_x = -1.0 * (Gref[:, 0,1].imag + Gref[:, 1,0].imag)
    A_y = -1.0 * (Gref[:, 0,1].imag - Gref[:, 1,0].imag)
    ax2.plot(omegas, A_x, label='x', linestyle =':')
    ax2.plot(omegas, A_y, label='y', linestyle =':')
    # ax2.plot(omegas, A_z, label='z', linestyle =':')
ax2.set_xlabel(r"$\omega$", fontsize=13)
ax2.set_ylabel(r"$A_{00}(\omega)$", fontsize=13)
ax2.set_title("Spectral function", fontsize=13)
ax2.set_xlim(-omegaMax,omegaMax)
if plot_components :
    ax2.legend(loc = 'upper right')
# cbar2 = fig2.colorbar(sm, ax=ax2, shrink=0.9, pad=0.02)
# cbar2.set_label(r"$U^2 / (g^2/L^d)$", fontsize=13)
fig2.suptitle(title, fontsize=12)
fig2.tight_layout()
fig2.savefig(os.path.join(script_dir, "cavity_crossover_U_spectral.pdf"), transparent = True, dpi=400)
fig2.savefig(os.path.join(script_dir, "cavity_crossover_U_spectral.png"), transparent = True, dpi=400)

# plt.show()
