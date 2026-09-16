#!/usr/bin/env python3
"""
Thermal-FDT regression test driver.

Builds quantum_boltz_cavity.ex, runs it in --thermal-test mode (photon
forced exactly thermal), and plots F(eps) from the simulation against the
exact tanh(beta_test*eps/2) it should reproduce.

Parameters are read from a persistent, editable file -- param_thermal_test.in
by default, same __key=value format as the main code's param_cavity.in. It is
created automatically (filled with defaults) the first time you run this
script. Just edit that file and rerun to change parameters.

By default the test temperature is T_LF (the low-frequency effective
temperature derived from delta_cav/Gamma_cav, matching the code's own
beta_LF). Set __beta_test in the param file (or pass --T/--beta below) to
test at a different, explicit temperature instead.

Command-line flags are optional one-off overrides layered on top of the
param file for this run only -- they do NOT modify the param file itself.

Examples:
    python3 thermal_test.py                           # uses param_thermal_test.in as-is
    python3 thermal_test.py --g_eff_sq 0.2 --nk 400    # one-off override, file untouched
    python3 thermal_test.py --T 0.5
    python3 thermal_test.py --param-file my_test.in    # use a different param file
    python3 thermal_test.py --no-run                   # just replot existing data
    python3 thermal_test.py --no-build --no-show       # rerun without rebuilding/popping a window
"""
import argparse
import builtins
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,            # Enable LaTeX rendering
    "font.family": "serif",         # Use a serif font
    "font.serif": ["Computer Modern"],  # Specify LaTeX font
    "text.latex.preamble": r"\usepackage{amsmath}" ,  # Optional: Load additional LaTeX packages
    'font.size': 18
})


def print(*args, **kwargs):  # always flush, so our messages interleave correctly
    kwargs.setdefault("flush", True)  # with the C++ subprocess's inherited stdout
    builtins.print(*args, **kwargs)

SCRIPT_DIR = Path(__file__).resolve().parent
EXE = SCRIPT_DIR / "quantum_boltz_cavity.ex"
DEFAULT_PARAM_FILE = SCRIPT_DIR / "param_thermal_test.in"
WORKING_PARAM_FILE = SCRIPT_DIR / "thermal_test_params.in"  # resolved file actually passed to the executable
DATA_FILE = SCRIPT_DIR / "thermal_fdt_test.dat"
PNG_FILE = SCRIPT_DIR / "thermal_fdt_test.png"

# Maps CLI dest -> __key in the param file, with the defaults used to create
# a fresh param file (same values run_thermal_fdt_test uses in the C++ code).
PARAM_KEYS = {
    "nk": "__nk",
    "numOfFreq": "__numOfFreq",
    "freq_cutoff": "__freq_cutoff",
    "t_hop": "__t_hop",
    "delta_cav": "__delta_cav",
    "Gamma_cav": "__Gamma_cav",
    "g_eff_sq": "__g_eff_sq",
    "Niter": "__Niter",
    "mixing": "__mixing",
    "err": "__err",
    "Jbath": "__Jbath",
}
DEFAULTS = {
    "__nk": 200,
    "__numOfFreq": 4000,
    "__freq_cutoff": 20.0,
    "__t_hop": 1.0,
    "__delta_cav": 2.0,
    "__Gamma_cav": 2.0,
    "__g_eff_sq": 0.1,
    "__Niter": 80,
    "__mixing": 0.5,
    "__err": 1e-11,
    "__Jbath": 0.05,
    "__beta_test": -1,  # < 0 means "use T_LF"; set explicitly to fix the test temperature
}
PARAM_ORDER = list(DEFAULTS.keys())  # for a stable, readable file layout


def parse_args():
    p = argparse.ArgumentParser(
        description="Run and plot the thermal-FDT regression test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--param-file", type=Path, default=DEFAULT_PARAM_FILE,
                   help="persistent, editable parameter file (__key=value, created with defaults if missing)")
    p.add_argument("--nk", type=int, help="number of k points")
    p.add_argument("--numOfFreq", type=int, help="number of frequency grid points")
    p.add_argument("--freq_cutoff", type=float, help="frequency grid half-width")
    p.add_argument("--t_hop", type=float, help="hopping amplitude")
    p.add_argument("--delta_cav", type=float, help="cavity detuning")
    p.add_argument("--Gamma_cav", type=float, help="cavity linewidth")
    p.add_argument("--g_eff_sq", type=float, help="cavity coupling squared")
    p.add_argument("--Niter", type=int, help="max self-consistency iterations")
    p.add_argument("--mixing", type=float, help="mixing parameter (weight kept on old Gloc)")
    p.add_argument("--err", type=float, help="convergence tolerance")
    p.add_argument("--Jbath", type=float, help="seed-only bath broadening")

    temp_group = p.add_mutually_exclusive_group()
    temp_group.add_argument("--T", type=float, help="force the test temperature to this value (default: T_LF, or whatever __beta_test says in the param file)")
    temp_group.add_argument("--beta", type=float, help="force the test inverse temperature to this value")

    p.add_argument("--no-build", action="store_true", help="skip 'make quantum_boltz_cavity.ex' before running")
    p.add_argument("--no-run", action="store_true", help="skip running the simulation, just replot the existing data file")
    p.add_argument("--no-show", action="store_true", help="don't open an interactive plot window, only save the PNG")
    return p.parse_args()


def ensure_param_file(path):
    if not path.exists():
        write_param_dict(path, DEFAULTS)
        print(f"Created {path} with default parameters -- edit it and rerun to change them.")


def load_param_file(path):
    params = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key in DEFAULTS:
            params[key] = float(value)
    return params


def write_param_dict(path, params):
    lines = [f"{key}={params[key]}" for key in PARAM_ORDER if key in params]
    path.write_text("\n".join(lines) + "\n")


def resolve_params(args):
    ensure_param_file(args.param_file)

    params = dict(DEFAULTS)
    params.update(load_param_file(args.param_file))

    overrides = {}
    for dest, key in PARAM_KEYS.items():
        value = getattr(args, dest)
        if value is not None:
            overrides[key] = value
    if args.T is not None:
        overrides["__beta_test"] = 1.0 / args.T
    elif args.beta is not None:
        overrides["__beta_test"] = args.beta
    params.update(overrides)

    write_param_dict(WORKING_PARAM_FILE, params)
    print(f"Using parameters from {args.param_file}" +
          (f", with one-off overrides: {overrides}" if overrides else ""))
    return params


def build():
    print("Building quantum_boltz_cavity.ex ...")
    subprocess.run(["make", "quantum_boltz_cavity.ex"], cwd=SCRIPT_DIR, check=True)


def run_test():
    subprocess.run([str(EXE), "--thermal-test", str(WORKING_PARAM_FILE)], cwd=SCRIPT_DIR, check=True)


def plot():
    eps, F_numeric, F_exact = np.loadtxt(DATA_FILE, unpack=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(eps, F_exact, "-", color="k", lw=1.5, label= r'${\rm tanh} \beta_{\rm ph} \varepsilon / 2$' )
    ax.plot(eps, F_numeric, "o", ms=3, color="C3", alpha=0.7, label= r'$F_{\rm simu}$')
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel(r"$\varepsilon_{k,\nu}$")
    ax.set_ylabel(r"$F(\varepsilon)$")
    ax.set_title("Thermal-FDT test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PNG_FILE, dpi=150)
    print(f"Wrote {PNG_FILE}")
    return fig


def main():
    args = parse_args()

    if not args.no_run:
        resolve_params(args)
        if not args.no_build:
            build()
        run_test()
    elif not DATA_FILE.exists():
        sys.exit(f"--no-run given but {DATA_FILE} does not exist yet; run without --no-run first.")

    fig = plot()
    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
