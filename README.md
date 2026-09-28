# FermionsOpenCavity

Code accompanying:

> J. Sueiro and M. Schirò, *Non-Thermal Effects in Fermionic Atoms Coupled to Open Cavities*,
> [arXiv:2609.20434](https://arxiv.org/abs/2609.20434)

It computes the steady-state properties of fermions on a 1D lattice coupled to a driven-dissipative
cavity mode, in two complementary regimes treated in the paper:

- **`Free_fermions/`** — the non-interacting limit (U = 0, Sect. 3): closed-form mean-field theory,
  solved directly without a self-consistency loop.
- **`LowU/`** — the low-U expansion (Sect. 4): a steady-state Keldysh Dyson equation with a
  2nd-order (IPT) Hubbard self-energy, solved self-consistently on the real-frequency axis.

## Repository structure

```
Free_fermions/
    freeFermions.py    # mean-field equations, distribution functions, cavity propagator
    plots.ipynb         # reproduces Figs. 2, 4 and 5 of the paper

LowU/
    quantum_boltz_cavity.cpp   # Keldysh Dyson equation self-consistency loop
    ness_*.hpp / ness.cpp      # vendored NESSi library (see Acknowledgments)
    find_param.h                # parameter-file parser
    Makefile
    simulate_crossover.py       # builds & runs quantum_boltz_cavity.ex over a scan of U, saves crossover_data.npz
    plot_crossover.py           # replots crossover_data.npz without resimulating
    thermal_test.py             # thermal-FDT regression test (see below)
```

## Requirements

### Python (both parts)
- numpy, scipy, numba, matplotlib
- jupyter / ipykernel (to run `Free_fermions/plots.ipynb`)

### C++ (`LowU/` only)
- A C++17 compiler with OpenMP support (the Makefile defaults to `g++-14`)
- FFTW3, Eigen3, Boost (on macOS: `brew install fftw eigen boost`)
- GNU Make

## Usage

### Free fermions (U = 0, Sect. 3)

Open `Free_fermions/plots.ipynb` and run all cells; it imports `freeFermions.py` and generates
every panel of Figs. 2, 4 and 5.

### Low-U expansion (Sect. 4)

- `simulate_crossover.py` builds `quantum_boltz_cavity.ex` (via `make`) and runs it over a scan of
  U, saving the results to `crossover_data.npz` and producing `cavity_crossover_U_*.pdf/png`.
- `plot_crossover.py` regenerates the plots from the saved `crossover_data.npz` without rerunning
  the simulation.
- `thermal_test.py` is a regression test: it forces the cavity photon propagator to be exactly
  thermal at a temperature T and checks that the fermions converge to the same temperature T, as
  required by the fluctuation-dissipation theorem. Run with `python3 thermal_test.py`.
- Simulation parameters (couplings, grid sizes, convergence criteria, ...) are set in
  `param_cavity.in`-style files; see `simulate_crossover.py` for the parameter dictionary and
  `quantum_boltz_cavity.cpp` for their meaning.

## Acknowledgments

The `ness_*.hpp` / `ness.cpp` files in `LowU/` are a vendored, standalone subset of the NESSi
library, adapted from the code used in:

> A. Picano, J. Li and M. Eckstein, *A quantum Boltzmann equation for strongly correlated
> electrons*, Phys. Rev. B **104**, 085108 (2021), doi:10.1103/PhysRevB.104.085108

which itself builds on:

> *NESSi: The Non-Equilibrium Systems Simulation package*, Computer Physics Communications
> **257**, 107484 (2020), doi:10.1016/j.cpc.2020.107484
