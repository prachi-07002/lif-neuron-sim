"""Demo / experiment runner for the LIF neuron simulator.

Runs three experiments and writes every result as CSV (always) and as a
PNG plot (when matplotlib is available):

  1. f-I curve: firing rate vs constant input current (0-5 nA)
  2. Parameter sweeps at I = 3 nA: tau_m (5-40 ms) and V_th (-55..-45 mV)
  3. Step response: current steps 0 -> 3 nA at 300 ms (membrane trace +
     spike raster / spike times)

Usage:  python3 demo.py
Outputs: results/*.csv and plots/*.png next to this file.
"""

import csv
import os

import numpy as np

from lif_neuron import (LIFNeuron, fi_curve, firing_rate, step_response,
                        sweep_parameter, theoretical_threshold_current)

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, 'results')
PLOTS = os.path.join(BASE, 'plots')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


def write_csv(path, header, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f'  wrote {path}')


def plot_fi(currents, rates):
    if not HAVE_MPL:
        return
    plt.figure()
    plt.plot(currents, rates, 'o-')
    plt.axvline(1.5, color='gray', linestyle='--',
                label='rheobase 1.5 nA')
    plt.xlabel('Input current (nA)')
    plt.ylabel('Firing rate (Hz)')
    plt.title('LIF neuron f-I curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOTS, 'fi_curve.png'), dpi=150)
    plt.close()
    print('  wrote plots/fi_curve.png')


def plot_sweep(values, rates, param, xlabel):
    if not HAVE_MPL:
        return
    plt.figure()
    plt.plot(values, rates, 'o-')
    plt.xlabel(xlabel)
    plt.ylabel('Firing rate (Hz)')
    plt.title(f'Firing rate vs {param} (I = 3 nA)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOTS, f'sweep_{param}.png'), dpi=150)
    plt.close()
    print(f'  wrote plots/sweep_{param}.png')


def plot_step(times, V, spikes):
    if not HAVE_MPL:
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
    ax1.plot(times, V)
    ax1.axhline(-50, color='gray', linestyle='--', label='threshold -50 mV')
    ax1.set_ylabel('Membrane V (mV)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.eventplot(spikes, lineoffsets=0.5, linelengths=0.8, colors='black')
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Spikes')
    ax2.set_title('Spike raster (step 0 -> 3 nA at 300 ms)')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, 'step_response.png'), dpi=150)
    plt.close(fig)
    print('  wrote plots/step_response.png')


def main():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(PLOTS, exist_ok=True)
    neuron = LIFNeuron()
    print(f'Default neuron: tau_m={neuron.tau_m} ms, R={neuron.R} MOhm, '
          f'V_rest={neuron.V_rest} mV, V_th={neuron.V_th} mV, '
          f't_ref={neuron.t_ref} ms')
    print(f'Theoretical rheobase: '
          f'{theoretical_threshold_current(neuron):.2f} nA')

    # 1. f-I curve
    print('\nExperiment 1: f-I curve (1000 ms per current)')
    currents = np.arange(0.0, 5.01, 0.5)
    currents, rates, counts = fi_curve(currents)
    for I, r, c in zip(currents, rates, counts):
        print(f'  I = {I:4.1f} nA -> {c:3d} spikes, {r:6.1f} Hz')
    write_csv(os.path.join(RESULTS, 'fi_curve.csv'),
              ['current_nA', 'spike_count', 'firing_rate_hz'],
              list(zip(currents, counts, rates)))
    plot_fi(currents, rates)

    # 2a. tau_m sweep
    print('\nExperiment 2a: tau_m sweep (I = 3 nA)')
    taus, rates_tau, counts_tau = sweep_parameter(
        'tau_m', [5.0, 10.0, 20.0, 30.0, 40.0], current_nA=3.0)
    for tau, r, c in zip(taus, rates_tau, counts_tau):
        print(f'  tau_m = {tau:5.1f} ms -> {c:3d} spikes, {r:6.1f} Hz')
    write_csv(os.path.join(RESULTS, 'sweep_tau_m.csv'),
              ['tau_m_ms', 'spike_count', 'firing_rate_hz'],
              list(zip(taus, counts_tau, rates_tau)))
    plot_sweep(taus, rates_tau, 'tau_m', 'Membrane time constant tau_m (ms)')

    # 2b. V_th sweep
    print('\nExperiment 2b: V_th sweep (I = 3 nA)')
    ths, rates_th, counts_th = sweep_parameter(
        'V_th', [-55.0, -52.5, -50.0, -47.5, -45.0], current_nA=3.0)
    for th, r, c in zip(ths, rates_th, counts_th):
        print(f'  V_th = {th:6.1f} mV -> {c:3d} spikes, {r:6.1f} Hz')
    write_csv(os.path.join(RESULTS, 'sweep_V_th.csv'),
              ['V_th_mV', 'spike_count', 'firing_rate_hz'],
              list(zip(ths, counts_th, rates_th)))
    plot_sweep(ths, rates_th, 'V_th', 'Threshold V_th (mV)')

    # 3. Step response
    print('\nExperiment 3: step response (0 -> 3 nA at 300 ms)')
    times, V, spikes = step_response(I_low=0.0, I_high=3.0, switch_ms=300.0)
    print(f'  {len(spikes)} spikes, all after 300 ms; '
          f'rate after step: {firing_rate(spikes, 700.0):.1f} Hz '
          f'(over 700 ms post-step)')
    print(f'  first spike at {spikes[0]:.1f} ms, '
          f'mean ISI {np.diff(spikes).mean():.1f} ms')
    write_csv(os.path.join(RESULTS, 'step_response_trace.csv'),
              ['time_ms', 'V_mV'],
              list(zip(times[::10], V[::10])))  # every 1 ms
    write_csv(os.path.join(RESULTS, 'step_response_spikes.csv'),
              ['spike_time_ms'], [[s] for s in spikes])
    plot_step(times, V, spikes)

    print(f'\nDone. CSVs in results/, plots in plots/ '
          f'({"matplotlib available" if HAVE_MPL else "no matplotlib: CSV only"}).')


if __name__ == '__main__':
    main()
