

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from pulser import Register, Sequence, Pulse
from pulser.devices import DigitalAnalogDevice
from pulser.waveforms import ConstantWaveform
from pulser_simulation import QutipEmulator
import numpy as np
import time

# Define Register
positions_eq = {
    "Al1": (0, 0), "N1": (4, 0),
    "Al2": (0, 6), "N2": (4, 6),
    "Al3": (8, 0), "N3": (12, 0),
    "Al4": (8, 6), "N4": (12, 6)
}
register_eq = Register(positions_eq)
positions_strained = {k: (x, y * 1.01) for k, (x, y) in positions_eq.items()}
register_strained = Register(positions_strained)

# SW Potential
def compute_two_body(r, is_vertical=False):
    epsilon = 1.1
    sigma = 1.9
    A, B, p, q, r_cut = 7.049556277, 0.6022245584, 4, 0, 3.5
    if r <= 0 or r >= r_cut:
        return 0
    term1 = A * epsilon * (B * (sigma/r)**p - (sigma/r)**q)
    term2 = np.exp( 1.3*sigma / (r - r_cut))
    return term1 * term2 * (1.2 if is_vertical else 1.0)

def hamiltonian(register, config):
    qubits = list(register.qubits.items())
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7)]
    scale_factor = 1.9 / 4.0
    energy = 0
    for i, j in pairs:
        pos_i, pos_j = qubits[i][1], qubits[j][1]
        disp_i = -0.05 if int(config[i]) == 0 else 0.05
        disp_j = -0.05 if int(config[j]) == 0 else 0.05
        r_um = np.linalg.norm(np.array(pos_i) - np.array(pos_j))
        r = r_um * scale_factor + (disp_i - disp_j)
        is_vertical = (i % 2 == 0 and j == i + 1)
        energy += compute_two_body(r, is_vertical)
    return energy

# VQE
def evaluate_energy(params, register):
    seq = Sequence(register, DigitalAnalogDevice)
    seq.declare_channel("rydberg_local", "rydberg_local")
    n_qubits = len(register.qubits)
    qubits = list(register.qubits.items())
    for i, (qubit_id, pos_i) in enumerate(qubits):
        pulse1 = Pulse(ConstantWaveform(52, params[i]), ConstantWaveform(52, 0), 0)
        pulse2_amplitude = params[i + n_qubits] * (1 + 1.0 * pos_i[1] / 6)
        pulse2 = Pulse(ConstantWaveform(60, pulse2_amplitude), ConstantWaveform(60, 0), np.pi/2)
        pulse3 = Pulse(ConstantWaveform(52, params[i + 2 * n_qubits]), ConstantWaveform(52, 0), np.pi)
        pulse4 = Pulse(ConstantWaveform(60, params[i + 3 * n_qubits]), ConstantWaveform(60, 0), -np.pi/2)
        seq.target(qubit_id, "rydberg_local")
        seq.add(pulse1, "rydberg_local")
        seq.add(pulse2, "rydberg_local")
        seq.add(pulse3, "rydberg_local")
        seq.add(pulse4, "rydberg_local")
    sim = QutipEmulator.from_sequence(seq)
    result = sim.run()
    final_state = result.get_final_state()
    raw_probs = np.abs(final_state.full())**2
    probs = raw_probs / np.sum(raw_probs)
    basis_states = [format(i, f'0{n_qubits}b') for i in range(2**n_qubits)]
    top_configs = sorted(zip(basis_states, probs), key=lambda x: x[1], reverse=True)[:50]
    
    sample = np.random.choice(basis_states, size=1, p=probs.flatten())[0]
    return hamiltonian(register, sample), final_state

def optimize_vqe(register, max_iter=10):
    n_qubits = len(register.qubits)
    params = np.random.random(4 * n_qubits) * 0.5
    best_energy, best_params, best_state = float('inf'), params.copy(), None
    start_time = time.time()
    for _ in range(max_iter):
        iter_start = time.time()
        new_params = params + np.random.normal(0, 0.1, 4 * n_qubits)
        new_params = np.clip(new_params, 0, None)
        new_energy, new_state = evaluate_energy(new_params, register)
        if new_energy < best_energy or _ == 0:
            best_energy, best_params, best_state = new_energy, new_params, new_state
            params = new_params
            print(f"Iteration {_+1}, Energy: {new_energy:.4f} eV, Time: {time.time() - iter_start:.2f} s")
    print(f"Total simulation time: {time.time() - start_time:.2f} s")
    return best_energy, best_params, best_state

# Polarization
def polarization_energy(config, register):
    dipole_strength = 0.20
    strain_factor = 1.0 if register is register_eq else 1.2  # Boost in strained
    energy = 0
    vertical_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    for i, j in vertical_pairs:
        if int(config[i]) != int(config[j]):
            energy += dipole_strength * strain_factor
    return energy

def evaluate_polarization(params, register):
    seq = Sequence(register, DigitalAnalogDevice)
    seq.declare_channel("rydberg_local", "rydberg_local")
    n_qubits = len(register.qubits)
    qubits = list(register.qubits.items())
    for i, (qubit_id, pos_i) in enumerate(qubits):
        pulse1 = Pulse(ConstantWaveform(52, params[i]), ConstantWaveform(52, 0), 0)
        pulse2_amplitude = params[i + n_qubits] * (1 + 1.0 * pos_i[1] / 6)
        pulse2 = Pulse(ConstantWaveform(60, pulse2_amplitude), ConstantWaveform(60, 0), np.pi/2)
        pulse3 = Pulse(ConstantWaveform(52, params[i + 2 * n_qubits]), ConstantWaveform(52, 0), np.pi)
        pulse4 = Pulse(ConstantWaveform(60, params[i + 3 * n_qubits]), ConstantWaveform(60, 0), -np.pi/2)
        seq.target(qubit_id, "rydberg_local")
        seq.add(pulse1, "rydberg_local")
        seq.add(pulse2, "rydberg_local")
        seq.add(pulse3, "rydberg_local")
        seq.add(pulse4, "rydberg_local")
    sim = QutipEmulator.from_sequence(seq)
    result = sim.run()
    final_state = result.get_final_state()
    raw_probs = np.abs(final_state.full())**2
    probs = raw_probs / np.sum(raw_probs)
    print(f"Max probability: {probs.max():.4f}")
    basis_states = [format(i, f'0{n_qubits}b') for i in range(2**n_qubits)]
    top_configs = sorted(zip(basis_states, probs), key=lambda x: x[1], reverse=True)[:5]
    samples = [config[0] for config in top_configs]
    pol_samples = [polarization_energy(sample, register) for sample in samples]
    
    return np.mean(pol_samples)

# u Parameter
def compute_u_avg(register):
    qubits = list(register.qubits.items())
    scale_factor = 1.9 / 4.0
    lc_list = []
    strain_pairs = [(0, 2), (4, 6)]
    
    for i, j in strain_pairs:
        pos_i, pos_j = qubits[i][1], qubits[j][1]
        r = abs(pos_j[1] - pos_i[1]) * scale_factor
        lc_list.append(float(r))
    lc_avg = np.mean(lc_list)
    lab_avg = 1.9
    u = lc_avg / (2 * lab_avg)
    
    return u

# Simulate
print("Simulating Pure AlN (8 atoms)...")
energy_eq, best_params_eq, state_eq = optimize_vqe(register_eq)
energy_strained, best_params_strained, state_strained = optimize_vqe(register_strained)
print(f"Equilibrium Energy: {energy_eq:.4f} eV")
print(f"Strained Energy: {energy_strained:.4f} eV")

pol_eq = evaluate_polarization(best_params_eq, register_eq)
pol_strained = evaluate_polarization(best_params_strained, register_strained)
delta_pol = pol_strained - pol_eq


epsilon_33 = 0.01
delta_E = energy_strained - energy_eq
volume = (3.11e-10)**2 * (4.98e-10) * 8
delta_V = volume * epsilon_33
sigma_33 = (delta_E * 1.6e-19) / delta_V
C_33 = sigma_33 / epsilon_33
print(f"C33: {C_33 / 1e9:.2f} GPa")

u_eq = compute_u_avg(register_eq)
u_strained = compute_u_avg(register_strained)
delta_u = u_strained - u_eq


area = (3.11e-10 * 2)**2
e = 1.6e-19
calibration_factor = e / (area * epsilon_33)
delta_Pz = delta_pol * calibration_factor

e33_0 = 0.2
e33_internal = delta_Pz
e33 = e33_0 + e33_internal

print(f"delta_pol: {delta_pol:.6f} eV")
print(f"delta_Pz: {delta_Pz:.6f} C/m²")
print(f"e33: {e33:.2f} C/m²")
d_33 = e33 / C_33 * 1e12
print(f"Predicted d33 (Pure AlN): {d_33:.2f} pC/N")