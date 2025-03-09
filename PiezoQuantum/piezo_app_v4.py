

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import streamlit as st
from pulser import Register, Sequence, Pulse
from pulser.devices import DigitalAnalogDevice
from pulser.waveforms import ConstantWaveform
from pulser_simulation import QutipEmulator
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import time
from streamlit import session_state

# Material Database with Benchmarks
materials = {
    "AlN": {"epsilon": 1.1, "sigma": 1.9, "dipole_strength": 0.21, "a_lat": 3.11e-10, "C33": 395, "e33": 1.55, "d33": 5.0},
    "ZnO": {"epsilon": 1.8, "sigma": 1.95, "dipole_strength": 0.15, "a_lat": 3.25e-10, "C33": 210, "e33": 1.2, "d33": 5.9},
    "PZT": {"epsilon": 2.5, "sigma": 2.1, "dipole_strength": 0.5, "a_lat": 4.0e-10, "C33": 120, "e33": 15.0, "d33": 225.0}
}

dopants = {
    "None": {"epsilon_factor": 1.0, "dipole_factor": 1.0, "e33_boost": 1.0},
    "Cr": {"epsilon_factor": 1.3, "dipole_factor": 1.4, "e33_boost": 1.5},
    "Sc": {"epsilon_factor": 1.2, "dipole_factor": 1.3, "e33_boost": 1.4}
}

# Dynamic Register Definition
def create_register(num_atoms):
    if num_atoms == 4:
        return Register({"A1": (0, 0), "B1": (4, 0), "A2": (0, 6), "B2": (4, 6)})
    elif num_atoms == 8:
        positions_eq = {
            "Al1": (0, 0), "N1": (4, 0),
            "Al2": (0, 6), "N2": (4, 6),
            "Al3": (8, 0), "N3": (12, 0),
            "Al4": (8, 6), "N4": (12, 6)
        }
        return Register(positions_eq)
    else:  # 16 atoms
        return Register({
            "A1": (0, 0), "B1": (4, 0), "A2": (0, 6), "B2": (4, 6),
            "A3": (8, 0), "B3": (12, 0), "A4": (8, 6), "B4": (12, 6),
            "A5": (16, 0), "B5": (20, 0), "A6": (16, 6), "B6": (20, 6),
            "A7": (24, 0), "B7": (28, 0), "A8": (24, 6), "B8": (28, 6)
        })

# Backend Computation Functions
def compute_two_body(r, is_doped=False, is_vertical=False):
    epsilon = material_params["epsilon"] * (dopant_params["epsilon_factor"] if is_doped else 1.0)
    sigma = material_params["sigma"]
    A, B, p, q, r_cut = 7.049556277, 0.6022245584, 4, 0, 3.5
    if r <= 0 or r >= r_cut:
        return 0
    term1 = A * epsilon * (B * (sigma/r)**p - (sigma/r)**q)
    term2 = np.exp(1.3 * sigma / (r - r_cut))
    return term1 * term2 * (1.2 if is_vertical else 1.0)

def hamiltonian(register, config, dopant_sites):
    qubits = list(register.qubits.items())
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7)] if len(qubits) == 8 else \
            [(i, i + 1) for i in range(0, len(qubits) - 1, 2)] + \
            [(i, i + 2) for i in range(0, len(qubits) - 2, 4)] + \
            [(i + 1, i + 3) for i in range(0, len(qubits) - 2, 4)]
    scale_factor = 1.9 / 4.0
    energy = 0
    for i, j in pairs:
        pos_i, pos_j = qubits[i][1], qubits[j][1]
        disp_i = -0.05 if int(config[i]) == 0 else 0.05
        disp_j = -0.05 if int(config[j]) == 0 else 0.05
        r_um = np.linalg.norm(np.array(pos_i) - np.array(pos_j))
        r = r_um * scale_factor + (disp_i - disp_j)
        is_doped = i in dopant_sites
        is_vertical = (i % 2 == 0 and j == i + 1)
        energy += compute_two_body(r, is_doped, is_vertical)
    return energy

def evaluate_energy(params, register, dopant_sites, pulse_duration, progress_bar, log_container):
    # Ensure pulse_duration is a multiple of 4 ns
    pulse_duration = max(52, round(pulse_duration / 4) * 4)
    seq = Sequence(register, DigitalAnalogDevice)
    seq.declare_channel("rydberg_local", "rydberg_local")
    n_qubits = len(register.qubits)
    qubits = list(register.qubits.items())
    if num_atoms == 8:
        for i, (qubit_id, pos_i) in enumerate(qubits):
            pulse1 = Pulse(ConstantWaveform(pulse_duration, params[i]), ConstantWaveform(pulse_duration, 0), 0)
            pulse2_amplitude = params[i + n_qubits] * (1 + 1.0 * pos_i[1] / 6)
            pulse2 = Pulse(ConstantWaveform(pulse_duration, pulse2_amplitude), ConstantWaveform(pulse_duration, 0), np.pi/2)
            pulse3 = Pulse(ConstantWaveform(pulse_duration, params[i + 2 * n_qubits]), ConstantWaveform(pulse_duration, 0), np.pi)
            pulse4 = Pulse(ConstantWaveform(pulse_duration, params[i + 3 * n_qubits]), ConstantWaveform(pulse_duration, 0), -np.pi/2)
            seq.target(qubit_id, "rydberg_local")
            seq.add(pulse1, "rydberg_local")
            seq.add(pulse2, "rydberg_local")
            seq.add(pulse3, "rydberg_local")
            seq.add(pulse4, "rydberg_local")
    else:
        for i, (qubit_id, _) in enumerate(qubits):
            pulse1 = Pulse(ConstantWaveform(pulse_duration, params[i]), ConstantWaveform(pulse_duration, 0), 0)
            pulse2 = Pulse(ConstantWaveform(pulse_duration, params[i + n_qubits]), ConstantWaveform(pulse_duration, 0), np.pi/2)
            seq.target(qubit_id, "rydberg_local")
            seq.add(pulse1, "rydberg_local")
            seq.add(pulse2, "rydberg_local")
    sim = QutipEmulator.from_sequence(seq)
    progress_bar.progress(0.5, "Running VQE Simulation...")
    if not st.session_state.get("stop_simulation", False):
        result = sim.run()
        final_state = result.get_final_state()
        raw_probs = np.abs(final_state.full())**2
        probs = raw_probs / np.sum(raw_probs)
        basis_states = [format(i, f'0{n_qubits}b') for i in range(2**n_qubits)]
        sample = np.random.choice(basis_states, size=1, p=probs.flatten())[0]
        energy = hamiltonian(register, sample, dopant_sites)
        return energy, final_state, sample
    return None, None, None

def optimize_vqe(register, dopant_sites, max_iter, pulse_duration, progress_bar=None, log_container=None, energy_container=None, vibration_container=None):
    n_qubits = len(register.qubits)
    params = np.random.random(4 * n_qubits if num_atoms == 8 else 2 * n_qubits) * 0.5
    best_energy, best_params, best_state = float('inf'), params.copy(), None
    energies = []
    start_time = time.time()
    for i in range(max_iter):
        if st.session_state.get("stop_simulation", False):
            log_container.write("Simulation stopped by user.")
            return None, None, None, 0, energies
        new_params = params + np.random.normal(0, 0.1, len(params))
        new_params = np.clip(new_params, 0, None)
        progress_bar.progress((i + 1) / (max_iter + 3), f"Optimizing Lattice (Iteration {i+1}/{max_iter})...")
        energy, state, config = evaluate_energy(new_params, register, dopant_sites, pulse_duration, progress_bar, log_container)
        if energy is None:
            return None, None, None, 0, energies
        if energy < best_energy:
            best_energy, best_params, best_state = energy, new_params, state
            log_container.write(f"Iteration {i+1}: Energy = {best_energy:.4f} eV")
        energies.append(best_energy)
        # Fix: Use np.array_equal for coordinate comparison
        is_equilibrium = register.qubits.keys() == register_eq.qubits.keys() and all(np.array_equal(register.qubits[k], register_eq.qubits[k]) for k in register.qubits)
        update_energy_plot(energy_container, energies, "Equilibrium" if is_equilibrium else "Strained")
        update_vibration_plot(vibration_container, register, config, dopant_sites, i + 1, max_iter, best_energy)
        params = new_params
    total_time = time.time() - start_time
    return best_params, best_energy, best_state, total_time, energies

def polarization_energy(config, register):
    dipole_strength = material_params["dipole_strength"] * dopant_params["dipole_factor"]
    # Fix: Use np.array_equal for coordinate comparison
    is_equilibrium = register.qubits.keys() == register_eq.qubits.keys() and all(np.array_equal(register.qubits[k], register_eq.qubits[k]) for k in register.qubits)
    strain_factor = 1.0 if is_equilibrium else 1.2
    energy = 0
    vertical_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)] if len(config) == 8 else \
                     [(i, i + 1) for i in range(0, len(config) - 1, 2)]
    for i, j in vertical_pairs:
        if int(config[i]) != int(config[j]):
            energy += dipole_strength * strain_factor
    return energy

def evaluate_polarization(params, register, pulse_duration, progress_bar):
    # Ensure pulse_duration is a multiple of 4 ns
    pulse_duration = max(52, round(pulse_duration / 4) * 4)
    seq = Sequence(register, DigitalAnalogDevice)
    seq.declare_channel("rydberg_local", "rydberg_local")
    n_qubits = len(register.qubits)
    qubits = list(register.qubits.items())
    if num_atoms == 8:
        for i, (qubit_id, pos_i) in enumerate(qubits):
            pulse1 = Pulse(ConstantWaveform(pulse_duration, params[i]), ConstantWaveform(pulse_duration, 0), 0)
            pulse2_amplitude = params[i + n_qubits] * (1 + 1.0 * pos_i[1] / 6)
            pulse2 = Pulse(ConstantWaveform(pulse_duration, pulse2_amplitude), ConstantWaveform(pulse_duration, 0), np.pi/2)
            pulse3 = Pulse(ConstantWaveform(pulse_duration, params[i + 2 * n_qubits]), ConstantWaveform(pulse_duration, 0), np.pi)
            pulse4 = Pulse(ConstantWaveform(pulse_duration, params[i + 3 * n_qubits]), ConstantWaveform(pulse_duration, 0), -np.pi/2)
            seq.target(qubit_id, "rydberg_local")
            seq.add(pulse1, "rydberg_local")
            seq.add(pulse2, "rydberg_local")
            seq.add(pulse3, "rydberg_local")
            seq.add(pulse4, "rydberg_local")
    else:
        for i, (qubit_id, _) in enumerate(qubits):
            pulse1 = Pulse(ConstantWaveform(pulse_duration, params[i]), ConstantWaveform(pulse_duration, 0), 0)
            pulse2 = Pulse(ConstantWaveform(pulse_duration, params[i + n_qubits]), ConstantWaveform(pulse_duration, 0), np.pi/2)
            seq.target(qubit_id, "rydberg_local")
            seq.add(pulse1, "rydberg_local")
            seq.add(pulse2, "rydberg_local")
    sim = QutipEmulator.from_sequence(seq)
    progress_bar.progress(0.75, "Computing Polarization...")
    if not st.session_state.get("stop_simulation", False):
        result = sim.run()
        final_state = result.get_final_state()
        raw_probs = np.abs(final_state.full())**2
        probs = raw_probs / np.sum(raw_probs)
        basis_states = [format(i, f'0{n_qubits}b') for i in range(2**n_qubits)]
        pol_samples = []
        start_time = time.time()
        for _ in range(3):
            sample = np.random.choice(basis_states, size=1, p=probs.flatten())[0]
            pol_samples.append(polarization_energy(sample, register))
        pol_time = time.time() - start_time
        return np.mean(pol_samples), pol_time
    return None, 0

def compute_u_avg(register):
    qubits = list(register.qubits.items())
    scale_factor = 1.9 / 4.0
    lc_list = []
    strain_pairs = [(0, 2), (4, 6)] if len(qubits) == 8 else [(i, i + 2) for i in range(0, len(qubits) - 2, 4)]
    for i, j in strain_pairs:
        pos_i, pos_j = qubits[i][1], qubits[j][1]
        r = abs(pos_j[1] - pos_i[1]) * scale_factor
        lc_list.append(r)
    lc_avg = np.mean(lc_list)
    lab_avg = material_params["sigma"]
    return lc_avg / (2 * lab_avg)

def update_energy_plot(container, energies, state):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, len(energies) + 1)), y=energies, mode='lines+markers', name=f'{state} Energy',
                             line=dict(color='#1f77b4'), marker=dict(size=8)))
    fig.update_layout(title=f"{state} Energy During VQE", xaxis_title="Iteration", yaxis_title="Energy (eV)",
                      template="plotly_white", height=300)
    container.plotly_chart(fig, use_container_width=True)

def update_vibration_plot(container, register, config, dopant_sites, iteration, max_iter, energy):
    qubits = list(register.qubits.items())
    x_coords, y_coords = [], []
    colors = []
    t = iteration / max_iter * 2 * np.pi
    amplitude = 0.05
    for i, (qid, (x, y)) in enumerate(qubits):
        disp = -amplitude if int(config[i]) == 0 else amplitude
        y_vib = y + disp * np.sin(t)
        x_coords.append(x)
        y_coords.append(y_vib)
        colors.append('#ff0000' if i in dopant_sites else '#0000ff' if i % 2 == 0 else '#00ff00')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_coords, y=y_coords, mode='markers', marker=dict(size=10, color=colors),
                             name='Atoms', text=[f"Atom {i+1}" for i in range(len(qubits))]))
    fig.update_layout(title=f"Atomic Vibrations (Iteration {iteration}/{max_iter}, Energy: {energy:.4f} eV)",
                      xaxis_title="X Position (µm)", yaxis_title="Y Position (µm)",
                      template="plotly_white", height=400, width=600,
                      xaxis_range=[-2, max(x_coords) + 2], yaxis_range=[-2, max(y_coords) + 2])
    container.plotly_chart(fig, use_container_width=True)

def estimate_simulation_time(num_atoms, max_iter, pulse_duration_ns):
    base_time_per_iter_atom = 0.5
    pulse_factor = pulse_duration_ns / 52
    total_atoms = num_atoms
    lattice_time = 2 * max_iter * total_atoms * base_time_per_iter_atom * pulse_factor
    pol_time = 2 * total_atoms * base_time_per_iter_atom * pulse_factor
    return lattice_time + pol_time

# Streamlit UI
st.set_page_config(page_title="PiezoQuantum Explorer", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    body {background-color: #f0f2f6; font-family: 'Arial', sans-serif;}
    .main-title {color: #1f77b4; font-size: 36px; font-weight: bold; text-align: center; margin-bottom: 20px;}
    .sidebar .sidebar-content {background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);}
    .stButton>button {background-color: #1f77b4; color: white; border-radius: 5px; padding: 10px 20px; font-weight: bold;}
    .stButton>button:hover {background-color: #155d87;}
    .metric-box {background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center;}
    .terminal {background-color: #1a1a1a; color: #00ff00; padding: 10px; border-radius: 5px; height: 200px; overflow-y: scroll; font-family: 'Courier New', monospace;}
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Simulation Settings")
base_material = st.sidebar.selectbox("Select Piezoelectric Material", list(materials.keys()))
dopant = st.sidebar.selectbox("Select Dopant", list(dopants.keys()))
num_atoms = st.sidebar.selectbox("Number of Atoms", [4, 8, 16])

max_iter_slider = st.sidebar.slider("Number of VQE Iterations", 1, 100, 1)
max_iter_input = st.sidebar.text_input("Enter Iterations (1-100)", value=str(max_iter_slider))
try:
    max_iter = int(max_iter_input)
    if not 1 <= max_iter <= 100:
        st.sidebar.error("Iterations must be between 1 and 100.")
        max_iter = max_iter_slider
except ValueError:
    st.sidebar.error("Please enter a valid integer.")
    max_iter = max_iter_slider

pulse_scale = st.sidebar.selectbox("Pulse Scale", ["ns", "µs"])
if pulse_scale == "ns":
    pulse_duration_slider = st.sidebar.slider("Pulse Duration (ns)", 52, 5000, 52, step=4)
    pulse_duration_input = st.sidebar.text_input("Enter Pulse Duration (ns, 52-5000, multiples of 4)", value=str(pulse_duration_slider))
    try:
        pulse_duration = int(pulse_duration_input)
        if pulse_duration % 4 != 0:
            pulse_duration = round(pulse_duration / 4) * 4
            st.sidebar.warning(f"Pulse duration adjusted to {pulse_duration} ns (multiple of 4 ns).")
        if not 52 <= pulse_duration <= 5000:
            st.sidebar.error("Pulse duration must be between 52 and 5000 ns.")
            pulse_duration = pulse_duration_slider
    except ValueError:
        st.sidebar.error("Please enter a valid integer.")
        pulse_duration = pulse_duration_slider
else:
    pulse_duration_slider = st.sidebar.slider("Pulse Duration (µs)", 0.052, 1.0, 0.052, step=0.004)
    pulse_duration_input = st.sidebar.text_input("Enter Pulse Duration (µs, 0.052-1.0, multiples of 0.004)", value=str(pulse_duration_slider))
    try:
        pulse_duration_us = float(pulse_duration_input)
        pulse_duration = int(pulse_duration_us * 1000)
        if pulse_duration % 4 != 0:
            pulse_duration = round(pulse_duration / 4) * 4
            st.sidebar.warning(f"Pulse duration adjusted to {pulse_duration / 1000:.3f} µs ({pulse_duration} ns, multiple of 4 ns).")
        if not 52 <= pulse_duration <= 1000000:
            st.sidebar.error("Pulse duration must be between 0.052 and 1.0 µs.")
            pulse_duration = int(pulse_duration_slider * 1000)
    except ValueError:
        st.sidebar.error("Please enter a valid number.")
        pulse_duration = int(pulse_duration_slider * 1000)

compute_button = st.sidebar.button("Start Simulation")
stop_button = st.sidebar.button("Stop Simulation")

est_time = estimate_simulation_time(num_atoms, max_iter, pulse_duration)
st.sidebar.write(f"Estimated Simulation Time: {est_time:.2f} s (~{est_time / 60:.1f} min)")

if stop_button:
    st.session_state["stop_simulation"] = True
    st.sidebar.write("Simulation will stop after current step.")

# Main Content
st.markdown('<div class="main-title">PiezoQuantum Explorer</div>', unsafe_allow_html=True)
st.write("Select a piezoelectric material, dopant, atom count, iterations, and pulse duration to compute coefficients using PASQAL’s quantum simulator.")

if "stop_simulation" not in st.session_state:
    st.session_state["stop_simulation"] = False

energy_eq_container = st.empty()
energy_strained_container = st.empty()
vibration_container = st.empty()

if compute_button:
    st.session_state["stop_simulation"] = False
    progress_bar = st.progress(0)
    log_container = st.empty()
    log_container.markdown('<div class="terminal">Simulation Log:<br></div>', unsafe_allow_html=True)

    with st.spinner("Initializing Simulation..."):
        material_params = materials[base_material]
        dopant_params = dopants[dopant]
        dopant_sites = [0, 2, 4, 6, 8][:num_atoms // 2] if dopant != "None" else []
        register_eq = create_register(num_atoms)
        register_strained = Register({k: (x, y * 1.01) for k, (x, y) in register_eq.qubits.items()})

        # Simulate Equilibrium State
        progress_bar.progress(0.1, "Simulating Equilibrium State...")
        params_eq, energy_eq, _, lattice_time_eq, energies_eq = optimize_vqe(register_eq, dopant_sites, max_iter, pulse_duration, progress_bar, log_container, energy_eq_container, vibration_container)
        if energy_eq is None:
            st.error("Simulation stopped.")
        else:
            # Simulate Strained State
            progress_bar.progress(0.3, "Simulating Strained State...")
            params_strained, energy_strained, _, lattice_time_strained, energies_strained = optimize_vqe(register_strained, dopant_sites, max_iter, pulse_duration, progress_bar, log_container, energy_strained_container, vibration_container)
            if energy_strained is None:
                st.error("Simulation stopped.")
            else:
                # Polarization
                progress_bar.progress(0.5, "Computing Equilibrium Polarization...")
                pol_eq, pol_time_eq = evaluate_polarization(params_eq, register_eq, pulse_duration, progress_bar)
                progress_bar.progress(0.7, "Computing Strained Polarization...")
                pol_strained, pol_time_strained = evaluate_polarization(params_strained, register_strained, pulse_duration, progress_bar)
                if pol_eq is None or pol_strained is None:
                    st.error("Simulation stopped.")
                else:
                    delta_pol = abs(pol_strained - pol_eq)

                    # Additional Metrics
                    u_eq = compute_u_avg(register_eq)
                    u_strained = compute_u_avg(register_strained)
                    delta_u = u_strained - u_eq

                    # Compute Coefficients
                    epsilon_33 = 0.01
                    delta_E = energy_strained - energy_eq
                    volume = (material_params["a_lat"] * (num_atoms // 4))**2 * (4.98e-10) * num_atoms
                    delta_V = volume * epsilon_33
                    sigma_33 = (delta_E * 1.6e-19) / delta_V
                    C_33 = sigma_33 / epsilon_33

                    area = (material_params["a_lat"] * (num_atoms // 4))**2
                    e = 1.6e-19
                    calibration_factor = e / (area * epsilon_33)
                    delta_Pz = delta_pol * calibration_factor
                    e33_0 = 0.2
                    e33_internal = delta_Pz
                    e33 = e33_0 + e33_internal * dopant_params["e33_boost"]
                    d_33 = e33 / C_33 * 1e12 if C_33 != 0 else float('inf')

                    progress_bar.progress(1.0, "Simulation Complete!")

                    # Results Display
                    st.subheader(f"Results for {base_material} {'with ' + dopant + ' doping' if dopant != 'None' else ''} ({num_atoms} atoms, {max_iter} iterations, {pulse_duration} ns pulse)")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                        st.metric("C33 (GPa)", f"{C_33 / 1e9:.2f}", f"{C_33 / 1e9 - material_params['C33']:.2f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                        st.metric("e33 (C/m²)", f"{e33:.2f}", f"{e33 - material_params['e33']:.2f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col3:
                        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                        st.metric("d33 (pC/N)", f"{d_33:.2f}", f"{d_33 - material_params['d33']:.2f}")
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Additional Metrics
                    st.subheader("Additional Metrics")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                        st.metric("Δu", f"{delta_u:.6f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                        st.metric("ΔPz (C/m²)", f"{delta_Pz:.6f}")
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Benchmark Table
                    benchmark_data = {
                        "Material": [base_material, f"{base_material} ({dopant})"],
                        "C33 (GPa)": [material_params["C33"], f"{C_33 / 1e9:.2f}"],
                        "e33 (C/m²)": [material_params["e33"], f"{e33:.2f}"],
                        "d33 (pC/N)": [material_params["d33"], f"{d_33:.2f}"]
                    }
                    df = pd.DataFrame(benchmark_data)
                    st.subheader("Benchmark Comparison")
                    st.table(df.style.set_properties(**{'background-color': '#f0f2f6', 'border-color': '#1f77b4', 'text-align': 'center'}))

                    # Timing Info
                    st.subheader("Simulation Times")
                    st.write(f"Lattice (Equilibrium): {lattice_time_eq:.2f} s")
                    st.write(f"Lattice (Strained): {lattice_time_strained:.2f} s")
                    st.write(f"Polarization (Equilibrium): {pol_time_eq:.2f} s")
                    st.write(f"Polarization (Strained): {pol_time_strained:.2f} s")
                    total_time = lattice_time_eq + lattice_time_strained + pol_time_eq + pol_time_strained
                    st.write(f"Total Time: {total_time:.2f} s (~{total_time / 60:.1f} min)")