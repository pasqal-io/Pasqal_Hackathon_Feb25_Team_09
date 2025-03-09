import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


from pulser import Register, Sequence, Pulse
from pulser.devices import DigitalAnalogDevice
from pulser.waveforms import ConstantWaveform
from pulser_simulation import QutipEmulator
import numpy as np
import time

# Define Register (16 atoms: 8 Al/Cr, 8 N)
positions_eq = {
    "Al1": (0, 0), "N1": (4, 0), "Al2": (0, 6), "N2": (4, 6),
    "Al3": (8, 0), "N3": (12, 0), "Al4": (8, 6), "N4": (12, 6),
    "Al5": (16, 0), "N5": (20, 0), "Al6": (16, 6), "N6": (20, 6),
    "Al7": (24, 0), "N7": (28, 0), "Al8": (24, 6), "N8": (28, 6)
}
register_eq = Register(positions_eq)
positions_strained = {k: (x, y * 1.01) for k, (x, y) in positions_eq.items()}
register_strained = Register(positions_strained)

# SW Potential (Tuned to DFT-like Cr-N softening)
def compute_two_body(r, is_cr_n=False, is_vertical=False):
    epsilon = 1.5 if not is_cr_n else 1.3  # eV, weaker Cr-N bond per DFT
    sigma = 1.9 if not is_cr_n else 2.1    # Å, larger Cr radius
    A, B, p, q, r_cut = 7.049556277, 0.6022245584, 4, 0, 3.5
    if r <= 0 or r >= r_cut:
        return 0
    term1 = A * epsilon * (B * (sigma/r)**p - (sigma/r)**q)
    term2 = np.exp(1.5 * sigma / (r - r_cut))
    return term1 * term2 * (1.2 if is_vertical else 1.0)

def hamiltonian(register, config, cr_sites):
    qubits = list(register.qubits.items())
    pairs = [
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15)
    ]
    scale_factor = 1.9 / 4.0
    energy = 0
    for i, j in pairs:
        pos_i, pos_j = qubits[i][1], qubits[j][1]
        disp_i = -0.005 if int(config[i]) == 0 else 0.005
        disp_j = -0.005 if int(config[j]) == 0 else 0.005
        r_um = np.linalg.norm(np.array(pos_i) - np.array(pos_j))
        r = r_um * scale_factor + (disp_i - disp_j)
        is_cr_n = i in cr_sites and j % 2 == 1  # Cr-N bond if i is Cr and j is N
        is_vertical = (i % 2 == 0 and j == i + 1)
        energy += compute_two_body(r, is_cr_n, is_vertical)
    return energy

# VQE for Lattice Optimization
def evaluate_energy(params, register, cr_sites):
    seq = Sequence(register, DigitalAnalogDevice)
    seq.declare_channel("rydberg_local", "rydberg_local")
    n_qubits = len(register.qubits)
    for i, qubit_id in enumerate(register.qubits.keys()):
        pulse1 = Pulse(ConstantWaveform(52, params[i]), ConstantWaveform(52, 0), 0)  # 20 ns
        pulse2 = Pulse(ConstantWaveform(52, params[i + n_qubits]), ConstantWaveform(52, 0), np.pi/2)
        seq.target(qubit_id, "rydberg_local")
        seq.add(pulse1, "rydberg_local")
        seq.add(pulse2, "rydberg_local")
    sim = QutipEmulator.from_sequence(seq)
    result = sim.run()
    final_state = result.get_final_state()
    raw_probs = np.abs(final_state.full())**2
    probs = raw_probs / np.sum(raw_probs)
    basis_states = [format(i, f'0{n_qubits}b') for i in range(2**n_qubits)]
    sample = np.random.choice(basis_states, size=1, p=probs.flatten())[0]
    return hamiltonian(register, sample, cr_sites), final_state

def optimize_vqe(register, cr_sites, max_iter=2):
    n_qubits = len(register.qubits)
    params = np.random.random(2 * n_qubits) * 0.5
    best_energy, best_params, best_state = float('inf'), params.copy(), None
    start_time = time.time()
    for i in range(max_iter):
        iter_start = time.time()
        new_params = params + np.random.normal(0, 0.05, 2 * n_qubits)  # Smaller step
        new_params = np.clip(new_params, 0, None)
        new_energy, new_state = evaluate_energy(new_params, register, cr_sites)
        if new_energy < best_energy:
            best_energy, best_params, best_state = new_energy, new_params, new_state
            params = new_params
            print(f"Iteration {i+1}, Energy: {new_energy:.4f} eV, Time: {time.time() - iter_start:.2f} s")
    total_time = time.time() - start_time
    print(f"Total lattice simulation time: {total_time:.2f} s")
    return best_params, best_energy, best_state

# Tight-Binding Polarization Model
def polarization_energy(config):
    dipole_strength = 0.25  # eV, tuned for e33 ~ 2-3 C/m²
    energy = 0
    vertical_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]
    for i, j in vertical_pairs:
        if int(config[i]) != int(config[j]):
            energy += dipole_strength
    return energy

def evaluate_polarization(params, register):
    seq = Sequence(register, DigitalAnalogDevice)
    seq.declare_channel("rydberg_local", "rydberg_local")
    n_qubits = len(register.qubits)
    pol_samples = []
    start_time = time.time()
    for i, qubit_id in enumerate(register.qubits.keys()):
        pulse1 = Pulse(ConstantWaveform(52, params[i]), ConstantWaveform(52, 0), 0)
        pulse2 = Pulse(ConstantWaveform(52, params[i + n_qubits]), ConstantWaveform(52, 0), np.pi/2)
        seq.target(qubit_id, "rydberg_local")
        seq.add(pulse1, "rydberg_local")
        seq.add(pulse2, "rydberg_local")
    sim = QutipEmulator.from_sequence(seq)
    result = sim.run()
    final_state = result.get_final_state()
    raw_probs = np.abs(final_state.full())**2
    probs = raw_probs / np.sum(raw_probs)
    basis_states = [format(i, f'0{n_qubits}b') for i in range(2**n_qubits)]
    for _ in range(3):
        sample = np.random.choice(basis_states, size=1, p=probs.flatten())[0]
        pol_samples.append(polarization_energy(sample))
    pol_time = time.time() - start_time
    print(f"Polarization computation time: {pol_time:.2f} s")
    return np.mean(pol_samples)

# Simulate Multiple Cr Concentrations
cr_percentages = [6.25, 12.5, 25, 31.25]  # 1, 2, 4, 5 Cr atoms out of 16
results = {}

for cr_pct in cr_percentages:
    n_cr = int(cr_pct / 100 * 8)  # Number of Cr atoms
    cr_sites = np.random.choice([0, 2, 4, 6, 8, 10, 12, 14], n_cr, replace=False).tolist()
    print(f"\nSimulating CrN-alloyed AlN ({cr_pct}% Cr substitution)...")
    
    # Equilibrium State
    best_params_eq, energy_eq, state_eq = optimize_vqe(register_eq, cr_sites)
    # Strained State
    best_params_strained, energy_strained, state_strained = optimize_vqe(register_strained, cr_sites)
    
    # Polarization
    pol_eq = evaluate_polarization(best_params_eq, register_eq)
    pol_strained = evaluate_polarization(best_params_strained, register_strained)
    delta_pol = abs(pol_strained - pol_eq)

    # Piezoelectric Coefficients
    epsilon_33 = 0.01
    delta_E = energy_strained - energy_eq
    volume = (3.11e-10)**2 * (4.98e-10) * 16
    delta_V = volume * epsilon_33
    sigma_33 = (delta_E * 1.6e-19) / delta_V
    C_33 = sigma_33 / epsilon_33
    area = (3.11e-10 * 4)**2
    e = 1.6e-19
    calibration_factor = e / (area * epsilon_33)
    delta_Pz = delta_pol * calibration_factor
    e33_0 = 0.2
    e33_internal = delta_Pz
    e33 = e33_0 + e33_internal * 1.5  # Cr enhancement factor
    d_33 = e33 / C_33 * 1e12

    results[cr_pct] = {
        "C33": C_33 / 1e9,
        "e33": e33,
        "d33": d_33,
        "delta_Pz": delta_Pz,
        "energy_eq": energy_eq,
        "energy_strained": energy_strained
    }
    print(f"C33: {C_33 / 1e9:.2f} GPa")
    print(f"e33: {e33:.2f} C/m²")
    print(f"d33: {d_33:.2f} pC/N")
    print(f"delta_Pz: {delta_Pz:.6f} C/m²")

# Summary
print("\nSummary of Results:")
for cr_pct, res in results.items():
    print(f"Cr {cr_pct}%: C33 = {res['C33']:.2f} GPa, e33 = {res['e33']:.2f} C/m², d33 = {res['d33']:.2f} pC/N")