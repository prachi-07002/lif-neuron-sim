"""Unit tests for the LIF neuron simulator (lif_neuron.py).

Run:  python3 -m unittest test_lif -v
"""

import unittest

import numpy as np

from lif_neuron import (LIFNeuron, fi_curve, firing_rate,
                        step_response, sweep_parameter,
                        theoretical_threshold_current)


class TestLIFBasics(unittest.TestCase):
    def test_no_spikes_below_threshold(self):
        """Sub-rheobase current never spikes (I = 1.0 nA < rheobase 1.5 nA)."""
        neuron = LIFNeuron()
        _, _, spikes = neuron.simulate(1.0, duration_ms=1000.0)
        self.assertEqual(len(spikes), 0)

    def test_no_spikes_with_zero_input(self):
        neuron = LIFNeuron()
        _, _, spikes = neuron.simulate(0.0, duration_ms=1000.0)
        self.assertEqual(len(spikes), 0)

    def test_spikes_above_threshold(self):
        """Supra-rheobase current produces regular spikes (I = 3.0 nA)."""
        neuron = LIFNeuron()
        _, _, spikes = neuron.simulate(3.0, duration_ms=1000.0)
        self.assertGreater(len(spikes), 10)
        # Regular firing: inter-spike intervals roughly equal (within 20%).
        isi = np.diff(spikes)
        self.assertLess(isi.std() / isi.mean(), 0.2)

    def test_membrane_decays_toward_rest_with_no_input(self):
        """With I = 0, V started above rest decays exponentially toward V_rest."""
        neuron = LIFNeuron()
        _, V, _ = neuron.simulate(0.0, duration_ms=200.0, V0=-55.0)
        # After 10 tau_m the membrane is within ~0.01 mV of rest.
        self.assertAlmostEqual(V[-1], neuron.V_rest, places=1)
        # Monotonic decay toward rest (never below rest, never rising).
        self.assertTrue(np.all(np.diff(V) <= 1e-9))
        self.assertGreater(V[0], V[-1])

    def test_reset_after_spike(self):
        """V is reset to V_reset immediately after a spike."""
        neuron = LIFNeuron()
        times, V, spikes = neuron.simulate(3.0, duration_ms=1000.0)
        self.assertGreater(len(spikes), 0)
        dt = times[1] - times[0]
        for s in spikes:
            idx = int(round(s / dt))
            self.assertAlmostEqual(V[idx], neuron.V_reset, places=6)

    def test_refractory_period_blocks_spikes(self):
        """No two spikes closer together than t_ref, even at huge current."""
        neuron = LIFNeuron(t_ref=5.0)
        _, _, spikes = neuron.simulate(100.0, duration_ms=500.0)
        self.assertGreater(len(spikes), 1)
        isi = np.diff(spikes)
        self.assertTrue(np.all(isi >= 5.0 - 1e-9),
                        f"spikes closer than t_ref: min ISI {isi.min():.3f} ms")

    def test_refractory_caps_max_firing_rate(self):
        """With t_ref = 5 ms, rate cannot exceed 1000/5 = 200 Hz."""
        neuron = LIFNeuron(t_ref=5.0)
        _, _, spikes = neuron.simulate(100.0, duration_ms=1000.0)
        self.assertLessEqual(firing_rate(spikes, 1000.0), 200.0 + 1e-9)

    def test_firing_rate_increases_with_input_current(self):
        """f-I curve is monotonically non-decreasing."""
        currents, rates, _ = fi_curve([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertTrue(np.all(np.diff(rates) >= 0),
                        f"rates not monotonic: {rates}")
        self.assertEqual(rates[0], 0.0)
        self.assertGreater(rates[-1], rates[2])

    def test_rheobase_matches_theory(self):
        """First spiking current sits just above (V_th - V_rest) / R = 1.5 nA."""
        neuron = LIFNeuron()
        self.assertAlmostEqual(theoretical_threshold_current(neuron), 1.5)
        _, _, below = neuron.simulate(1.4, duration_ms=1000.0)
        self.assertEqual(len(below), 0)
        _, _, above = LIFNeuron().simulate(1.6, duration_ms=1000.0)
        self.assertGreater(len(above), 0)

    def test_longer_tau_m_fires_slower_at_fixed_current(self):
        """Larger membrane time constant -> slower charging -> lower rate."""
        _, rates, _ = sweep_parameter('tau_m', [5.0, 10.0, 20.0, 40.0],
                                      current_nA=3.0)
        self.assertTrue(np.all(np.diff(rates) <= 0),
                        f"rates vs tau_m not decreasing: {rates}")
        self.assertGreater(rates[0], rates[-1])

    def test_higher_threshold_fires_slower(self):
        """Raising V_th lowers the firing rate at fixed current."""
        _, rates, _ = sweep_parameter('V_th', [-55.0, -50.0, -45.0],
                                      current_nA=3.0)
        self.assertTrue(np.all(np.diff(rates) <= 0),
                        f"rates vs V_th not decreasing: {rates}")

    def test_step_response_spikes_only_after_step(self):
        """Step from 0 to 3 nA at 300 ms: no spikes before, spikes after."""
        _, _, spikes = step_response(I_low=0.0, I_high=3.0, switch_ms=300.0)
        self.assertTrue(np.all(spikes >= 300.0))
        self.assertGreater(len(spikes), 5)

    def test_steady_state_voltage_below_threshold(self):
        """Sub-threshold steady state is V_rest + R*I (leaky integrator)."""
        neuron = LIFNeuron()
        _, V, _ = neuron.simulate(1.0, duration_ms=500.0)
        self.assertAlmostEqual(V[-1], neuron.V_rest + neuron.R * 1.0,
                               delta=0.05)


if __name__ == '__main__':
    unittest.main()
