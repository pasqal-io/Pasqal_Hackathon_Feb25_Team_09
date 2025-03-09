# Quantum Simulation of Piezoelectric AlN Alloys Using Pasqal's Pulser

## Project Overview
This project explores quantum simulation techniques for optimizing Aluminum Nitride (AlN)-based piezoelectric materials using Pasqal's Pulser platform. The objective is to investigate alloying strategies, such as CrN doping, to enhance key piezoelectric coefficients e33, d33, and elastic constants C33 for applications in energy harvesting.

## Project Context
- **AlN Wurtzite Structure**: Naturally exhibits piezoelectricity with d33 ≈ 5.5 pC/N.
- **Alloying Potential**: CrN incorporation has been shown to increase d33 by up to four times.
- **Application Areas**: Enhancing energy efficiency in sensors and energy harvesting devices.

---

## Phase 1: Compute Hamiltonian and Piezoelectric Coefficients
### Objective
Develop and simulate a Hamiltonian for piezoelectric AlN using quantum algorithms to compute a realistic ground state energy and extract key piezoelectric coefficients.

### Approach
- Utilize Variational Quantum Eigensolver (VQE) with a Stillinger-Weber (SW) two-body potential to model AlN’s lattice dynamics (phonons).
- Encode system dynamics via qubit displacements (±0.01 Å) targeting a ground state energy range of -11 eV to -13 eV for a 16-atom cluster.
- Validate results against Density Functional Theory (DFT) benchmarks.

### Expected Outcome
- Baseline piezoelectric properties:
  - e33 ≈ 1.5 C/m²
  - C33 ≈ 395 GPa

---

## Phase 2: Alloying AlN for Enhanced Properties
### Objective
Enhance AlN’s piezoelectric response through alloying strategies, informed by DFT data, and simulate material performance within the Pulser framework.

### Approach
- Modify the Hamiltonian to incorporate alloying effects such as Cr-N pair potentials.
- Utilize DFT-derived parameters to simulate energy changes and piezoelectric response.
- Scale Pulser’s simulation capabilities to test multiple alloying and doping configurations, including Cr, Sc, and Yb.

### Expected Outcome
- AlN alloys with d33 > 22 pC/N.
- Results validated against DFT and experimental benchmarks.

---

## Algorithm Goals
- **Accurate Ground State Energy**: -11 to -13 eV for a 16-atom AlN cluster.
- **Lattice Dynamics**: Simulate phonon-like vibrations to assess piezoelectric response.
- **Piezoelectric Properties**: Estimate e33, C33, and d33 under strain.
- **Alloying/Doping Effects**: Predict improvements in d33 for energy-harvesting materials.

This quantum framework is designed to efficiently screen AlN alloys and dopants for superior energy-harvesting applications.

---

## Code Explanation (Pulser Simulation)
### Overview
The Pulser simulation models a 16-qubit AlN cluster, focusing on optimizing ground state energy.

### Key Components
- **Register**: A 4×4 grid of 16 qubits with 4 μm spacing, encoding atomic displacements (±0.05 Å).
- **Hamiltonian**: Stillinger-Weber two-body potential over 16 nearest-neighbor pairs, scaled to micrometer units and mapped to Å.
- **VQE Optimization**:
  - Two-layer pulse sequence (52 ns runtime).
  - 1000 iterations for energy minimization with debugging outputs.
- **Visualization**: register.draw() and seq.final.draw() to illustrate atomic structure and pulse sequences.
- **Execution**: Iteratively refines energy towards a target range of -11 to -13 eV.

---

## Hamiltonian for Piezoelectric AlN
### Design
Based on the Stillinger-Weber (SW) Two-Body Potential.

### Parameters
- ε = 0.8 eV
- σ = 1.9 Å
- r_cut = 3.5 Å
- A = 7.05, B = 0.60, p = 4, q = 0

### Physical Relevance
- **Phonon Representation**: Qubit displacements (±0.05 Å) model lattice vibrations.
- **Bonding Energy**: Each pair contributes ≈ -0.5 to -0.8 eV, targeting a total energy range of -11 to -13 eV for 16 atoms.
- **Piezoelectric Basis**: Lays the foundation for incorporating strain-polarization coupling in subsequent simulations.

---

## Advantages of Quantum Computing with Pasqal
### Why Use Pasqal?
- **Scalability**: Expand simulations from 16 to 32+ qubits, surpassing DFT’s small-cell limitations.
- **Flexible Interactions**: Rydberg-state couplings naturally simulate dipole interactions.
- **Quantum Optimization**: VQE efficiently searches ground states for alloying and doping configurations.
- **Direct Simulation**: Qubit displacements intrinsically encode phonon modes.

### Advantages for Materials Discovery
- **Alloying/Doping Predictions**: Simulating CrN or ScN doping effects to enhance d33 beyond 22 pC/N.
- **Energy Harvesting Optimization**: Refining C33 and e33 to maximize efficiency.
- **Computational Speed**: Quantum techniques offer faster screening than classical DFT methods.

---

## Next Steps
- Implement three-body and coupling terms to achieve full piezoelectric simulations.
- Expand the simulation to 32 qubits and test CrN/Sc doping effects.

This research provides a scalable quantum approach for advancing the discovery and optimization of next-generation piezoelectric materials.


# PiezoQuantum APP

This repository contains the code and resources for the PiezoQuantum APP, which involves simulations and computations related to piezoelectric materials using Python, Streamlit, and various scientific computing libraries.

## Prerequisites

Before you begin, ensure you have the following installed:

1. **Conda**: A package, dependency, and environment management system.
2. **Python**: A programming language that lets you work quickly and integrate systems more effectively.

## Installation

### Step 1: Install Conda

If you don't have Conda installed, you can download and install it from the [official Conda website](https://docs.conda.io/en/latest/miniconda.html).

### Step 2: Create a Conda Environment

Create a new Conda environment with Python 3.8 (or any compatible version):


conda create -n piezoquantum python=3.8

conda activate piezoquantum

Install Required Packages
Install the necessary Python packages using Conda and pip:

- conda install -c conda-forge 
- pip install streamlit pulser qutip

Running the Programs
Running the Streamlit Application
To run the Streamlit application piezo_app_v4.py, navigate to the PiezoQuantum folder and execute the following command:

streamlit run piezo_app_v4.py

This will start a local web server and open the application in your default web browser. You can interact with the app through the browser interface.

### Running Python Scripts
To run the Python scripts pm_simulation_ain_compute_c33_e33_d33_8A.py, pm_simulation_ain_crn_compute_d33_v4.py, and pm_simulation_ain_two_three_body_coupling.py, use the following commands:

- python pm_simulation_ain_compute_c33_e33_d33_8A.py
- python pm_simulation_ain_crn_compute_d33_v4.py
- python pm_simulation_ain_two_three_body_coupling.py


Each script will execute its respective simulation or computation. Ensure that all dependencies are correctly installed and that the environment is activated before running these scripts.
