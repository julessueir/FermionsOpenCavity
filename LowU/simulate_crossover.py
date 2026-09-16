#!/usr/bin/env python3
"""
Run the cavity-Hubbard code for multiple values of U.
Saves results to crossover_data.npz for plotting separately.
"""

import subprocess
import os
import numpy as np

# ================================================================
# Fixed parameters
# ================================================================

base_params = {
    "numOfFreq":   50000,
    "freq_cutoff": 50.0,
    "nk":          500, # best around 400 to 500
    "t_hop":       2.5,  # In the normal phase the DoS goes from - 2t to 2t. 
    "mu":          0.0,
    "eps_perp":    0.0,
    "delta_cav":   5.0,
    "Gamma_cav":   5.0,
    "g_cav":       0.0, # controls the mean-field (Hartree) term
    "g_eff_sq":    0.025, # controls the fluctuations.
    "Jbath":       0.025,
    "beta":        3.0,
    "Niter":       300,
    "mixing":      0.5,
    "err":         1e-7,
}

# ================================================================
# Values of U to scan
# ================================================================

# U_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
U_values = list((np.linspace(0.0, 2.5, num = 6, endpoint = True)) )
U_values = [0.0, 0.25, 0.5, 1.0, 1.75 , 2.5]
# U_values = [0.]
# U_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0 ]
# U_values = [0.0, 0.25, 0.5 , 0.75, 1.0 ]
# U_values = [0.0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0]
# U_values = list(np.sqrt(np.linspace(0.0, 1.0, num = 6, endpoint = True)) )

# ================================================================
# Helpers
# ================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
param_file = os.path.join(script_dir, "param_cavity.in")
exe = os.path.join(script_dir, "quantum_boltz_cavity.ex")


def write_params(U):
    with open(param_file, "w") as f:
        for key, val in base_params.items():
            f.write(f"__{key}={val}\n")
        f.write(f"__U={U}\n")


def load_gf(filename):
    path = os.path.join(script_dir, filename)
    data = np.loadtxt(path)
    omega = data[:, 0]
    GR = np.zeros((len(omega), 2, 2), dtype=complex)
    GL = np.zeros((len(omega), 2, 2), dtype=complex)
    col = 1
    for i in range(2):
        for j in range(2):
            GR[:, i, j] = data[:, col] + 1j * data[:, col + 1]
            col += 2
    for i in range(2):
        for j in range(2):
            GL[:, i, j] = data[:, col] + 1j * data[:, col + 1]
            col += 2
    return omega, GR, GL


# ================================================================
# Compile & run
# ================================================================

print("Compiling...")
subprocess.run(["make", "quantum_boltz_cavity.ex"], cwd=script_dir, check=True)

all_omega = None
all_GR = {}
all_GL = {}

for U in U_values:
    print(f"\n{'='*50}")
    print(f"  Running U = {U}")
    print(f"{'='*50}")
    write_params(U)
    result = subprocess.run(
        [exe, param_file], cwd=script_dir, capture_output=True, text=True
    )
    lines = result.stdout.strip().split("\n")
    keywords = ("Converged", "sum rule", "WARNING")
    for line in lines:
        if any(kw in line for kw in keywords):
            print(line)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        raise RuntimeError(f"C++ code failed for U={U}")

    omega, GR, GL = load_gf("Gloc_converged.out")
    idx = np.argsort(omega)
    omega = omega[idx]
    GR = GR[idx]
    GL = GL[idx]

    if all_omega is None:
        all_omega = omega
    all_GR[U] = GR
    all_GL[U] = GL

# ================================================================
# Save
# ================================================================

outfile = os.path.join(script_dir, "crossover_data.npz")

# Pack GR and GL into arrays of shape (n_U, n_omega, 2, 2)
U_arr = np.array(U_values)
GR_arr = np.stack([all_GR[U] for U in U_values], axis=0)
GL_arr = np.stack([all_GL[U] for U in U_values], axis=0)

np.savez(outfile,
         omega=all_omega,
         U_values=U_arr,
         GR=GR_arr,
         GL=GL_arr,
         **{f"param_{k}": v for k, v in base_params.items()})

print(f"\nData saved to {outfile}")
print(f"  omega: {all_omega.shape}")
print(f"  U_values: {U_arr.shape}")
print(f"  GR: {GR_arr.shape}  (n_U, n_omega, 2, 2)")
print(f"  GL: {GL_arr.shape}")

subprocess.run(["python", "plot_crossover.py"],cwd=script_dir, check=True)
