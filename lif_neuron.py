"""Leaky integrate-and-fire (LIF) neuron simulator, from scratch in NumPy.

Model (single compartment, Euler integration):

    tau_m * dV/dt = -(V - V_rest) + R * I(t)
    if V >= V_th: emit a spike, V <- V_reset, hold for t_ref (refractory)

Units: V in mV, I in nA, R in MOhm (so R*I is in mV), time in ms.
Defaults mirror a textbook cortical neuron and the board the user will
build for her Spring 2027 Honors College project (spiking artificial
neuron, coin-cell powered, potentiometers to adjust weights manually):
the potentiometer on the board maps to R (input resistance / synaptic
weight) and to V_th here, so the sweeps in this repo are the same knobs
she will turn by hand on hardware.

Only NumPy is required. Matplotlib is used only by demo.py for plots;
every result is also written as CSV.
"""

import numpy as np


class LIFNeuron:
    """A single leaky integrate-and-fire neuron."""

    def __init__(self, tau_m=20.0, R=10.0, V_rest=-65.0, V_reset=-65.0,
                 V_th=-50.0, t_ref=2.0):
        self.tau_m = float(tau_m)      # membrane time constant (ms)
        self.R = float(R)              # input resistance (MOhm)
        self.V_rest = float(V_rest)    # resting potential (mV)
        self.V_reset = float(V_reset)  # reset potential after a spike (mV)
        self.V_th = float(V_th)        # spike threshold (mV)
        self.t_ref = float(t_ref)      # absolute refractory period (ms)
        self.reset_state()

    def reset_state(self):
        self.V = self.V_rest
        self.refractory_until = -np.inf  # time (ms) refractory period ends
        self.spike_times = []

    def step(self, I, t, dt):
        """Advance one Euler step at time t with input current I (nA).

        Returns True if a spike was emitted on this step.
        """
        if t < self.refractory_until:
            self.V = self.V_reset
            return False
        dV = (-(self.V - self.V_rest) + self.R * I) * dt / self.tau_m
        self.V += dV
        if self.V >= self.V_th:
            self.spike_times.append(float(t))
            self.V = self.V_reset
            self.refractory_until = t + self.t_ref
            return True
        return False

    def simulate(self, I, duration_ms=1000.0, dt=0.1, V0=None):
        """Simulate for ``duration_ms``.

        ``I`` may be a scalar (constant current, nA), a 1-D array of per-step
        currents, or a callable ``I(t_ms)``. Returns ``(times, V_trace,
        spike_times)`` as NumPy arrays.
        """
        self.reset_state()
        if V0 is not None:
            self.V = float(V0)
        n_steps = int(round(duration_ms / dt))
        times = np.arange(n_steps) * dt
        V_trace = np.empty(n_steps)
        if np.isscalar(I):
            I_arr = np.full(n_steps, float(I))
        elif callable(I):
            I_arr = np.array([float(I(t)) for t in times])
        else:
            I_arr = np.asarray(I, dtype=float)
            if I_arr.shape[0] != n_steps:
                raise ValueError(
                    f"I array length {I_arr.shape[0]} != steps {n_steps}")
        for i, t in enumerate(times):
            self.step(float(I_arr[i]), float(t), dt)
            V_trace[i] = self.V
        return times, V_trace, np.array(self.spike_times)


def firing_rate(spike_times, duration_ms):
    """Mean firing rate in Hz from spike times over a duration in ms."""
    return 1000.0 * len(spike_times) / float(duration_ms)


def theoretical_threshold_current(neuron):
    """Rheobase: minimum constant current (nA) that can ever reach threshold."""
    return (neuron.V_th - neuron.V_rest) / neuron.R


def fi_curve(currents_nA, duration_ms=1000.0, dt=0.1, **neuron_kwargs):
    """Firing rate (Hz) vs constant input current.

    Returns (currents, rates_hz, spike_counts). ``currents_nA`` is any
    iterable of currents in nA.
    """
    currents = np.asarray(list(currents_nA), dtype=float)
    rates = np.empty(len(currents))
    counts = np.empty(len(currents), dtype=int)
    for j, I in enumerate(currents):
        neuron = LIFNeuron(**neuron_kwargs)
        _, _, spikes = neuron.simulate(I, duration_ms=duration_ms, dt=dt)
        counts[j] = len(spikes)
        rates[j] = firing_rate(spikes, duration_ms)
    return currents, rates, counts


def sweep_parameter(param, values, current_nA=3.0, duration_ms=1000.0,
                    dt=0.1, **base_kwargs):
    """Firing rate vs one neuron parameter at fixed input current.

    ``param`` is a LIFNeuron constructor keyword (e.g. 'tau_m', 'V_th',
    't_ref', 'R'). Returns (values, rates_hz, spike_counts).
    """
    values = np.asarray(list(values), dtype=float)
    rates = np.empty(len(values))
    counts = np.empty(len(values), dtype=int)
    for j, v in enumerate(values):
        kwargs = dict(base_kwargs)
        kwargs[param] = float(v)
        neuron = LIFNeuron(**kwargs)
        _, _, spikes = neuron.simulate(current_nA, duration_ms=duration_ms,
                                       dt=dt)
        counts[j] = len(spikes)
        rates[j] = firing_rate(spikes, duration_ms)
    return values, rates, counts


def step_response(I_low=0.0, I_high=3.0, switch_ms=300.0, duration_ms=1000.0,
                  dt=0.1, **neuron_kwargs):
    """Response to a step current: I_low until switch_ms, then I_high.

    Returns (times, V_trace, spike_times). Models a potentiometer being
    turned up mid-experiment (or a synaptic input switching on).
    """
    def I_of_t(t):
        return I_low if t < switch_ms else I_high

    neuron = LIFNeuron(**neuron_kwargs)
    return neuron.simulate(I_of_t, duration_ms=duration_ms, dt=dt)
