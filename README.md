# lif-neuron-sim

A leaky integrate-and-fire (LIF) neuron simulator built from scratch in
NumPy — the software twin of the low-power analog spiking-neuron board
for my Spring 2027 Honors College research project with Prof. Walter
Daniel Leon-Salas (coin-cell powered board emulating a spiking neuron,
potentiometers to adjust weights manually, proof-of-concept prototype,
tested, then a small network used to control a robot).

The board's knobs map directly onto this model: the potentiometer that
sets a synaptic weight / input drive maps to the input current `I` and
input resistance `R` here, and the comparator threshold on the board
maps to `V_th`. The sweeps in this repo are the same knobs turned by
hand on hardware — so the simulation predicts what the board should do
before it is soldered, and the measured board behaviour can be checked
against these curves afterwards.

**Application talking point:** "I built a from-scratch LIF neuron simulator in NumPy and used it to predict my analog spiking-neuron board's firing behaviour — its f-I curve, rheobase, and how the threshold and membrane time constant set the firing rate — before building the hardware."

## Model

Single-compartment LIF, Euler integration (`dt = 0.1 ms`):

```
tau_m * dV/dt = -(V - V_rest) + R * I(t)
if V >= V_th: emit a spike, V <- V_reset, hold for t_ref (refractory)
```

Defaults: `tau_m = 20 ms`, `R = 10 MOhm`, `V_rest = V_reset = -65 mV`,
`V_th = -50 mV`, `t_ref = 2 ms`. Units: V in mV, I in nA, time in ms.

## How to run

Requires Python 3 + NumPy only. Matplotlib is optional — every result is
written as CSV, and plots are PNGs only when matplotlib is installed.

```bash
python3 demo.py                 # runs all experiments, writes results/*.csv and plots/*.png
python3 -m unittest test_lif -v # runs the test suite
```

Or use it as a library:

```python
from lif_neuron import LIFNeuron, fi_curve

neuron = LIFNeuron(tau_m=20.0, V_th=-50.0)
times, V, spikes = neuron.simulate(3.0, duration_ms=1000.0)  # constant 3 nA
print(f"{len(spikes)} spikes")  # 63 spikes -> 63 Hz

currents, rates, counts = fi_curve([0, 1, 2, 3, 4, 5])
```

`simulate()` also accepts a time-varying current: a per-step array, or a
callable `I(t_ms)` — see `step_response()` for a 0 -> 3 nA step at 300 ms.

## Headline results (measured by running `demo.py`)

All rates are over 1000 ms of simulation unless noted.

**Theoretical rheobase** (minimum current that can ever reach threshold,
`(V_th - V_rest) / R`): **1.5 nA** — confirmed: 1.4 nA never spikes,
1.6 nA spikes.

**f-I curve** (default neuron):

| Input current | Spikes / 1000 ms | Firing rate |
|---|---|---|
| 0.0–1.5 nA | 0 | 0 Hz |
| 2.0 nA | 33 | 33 Hz |
| 3.0 nA | 63 | 63 Hz |
| 4.0 nA | 88 | 88 Hz |
| 5.0 nA | 110 | 110 Hz |

Firing rate rises monotonically with input current, roughly linearly
above rheobase (~35 Hz per nA).

**Membrane time constant sweep** (I = 3 nA): tau_m 5 ms -> **185 Hz**,
10 ms -> 113 Hz, 20 ms -> 63 Hz, 30 ms -> 44 Hz, 40 ms -> **33 Hz**.
A faster (leakier) membrane charges to threshold sooner, so rate falls
~5.6x as tau_m goes 5 -> 40 ms.

**Threshold sweep** (I = 3 nA): V_th -55 mV -> **100 Hz**, -50 mV ->
63 Hz, -45 mV -> **41 Hz**. Raising the threshold ~2.4x lowers the rate.

**Step response** (0 -> 3 nA at 300 ms): **44 spikes**, all after the
step; first spike 13.8 ms after the step (313.8 ms), mean inter-spike
interval 15.8 ms, post-step rate 62.9 Hz — matching the 63 Hz constant-
current rate at 3 nA, as it should.

**Refractory limit:** with `t_ref = 5 ms` the rate caps at 200 Hz even
at 100 nA input (verified in tests).

## Files

| File | What it is |
|---|---|
| `lif_neuron.py` | The simulator: `LIFNeuron`, `fi_curve`, `sweep_parameter`, `step_response`, `firing_rate` |
| `demo.py` | Experiment runner: f-I curve, tau_m / V_th sweeps, step response; writes CSVs + PNGs |
| `test_lif.py` | 13 unit tests (see below) |
| `results/fi_curve.csv` | Firing rate vs input current |
| `results/sweep_tau_m.csv` | Firing rate vs membrane time constant |
| `results/sweep_V_th.csv` | Firing rate vs spike threshold |
| `results/step_response_trace.csv` | Membrane voltage trace, 1 ms samples |
| `results/step_response_spikes.csv` | Spike times for the step response |
| `plots/fi_curve.png` | f-I curve plot |
| `plots/sweep_tau_m.png`, `plots/sweep_V_th.png` | Parameter-sweep plots |
| `plots/step_response.png` | Membrane trace + spike raster |

All source and data files are plain text; the PNGs are generated plots
(regenerable from the CSVs by re-running `demo.py`).

## Tests — 13/13 pass

`python3 -m unittest test_lif -v` — 13 tests, 0 failures, ~0.1 s.
Coverage of the required behaviours:

- no spikes below threshold (0 nA, 1.0 nA) and no spikes at zero input
- spikes above threshold (3 nA, >10 spikes, regular inter-spike intervals)
- refractory period blocks spikes (no ISI < t_ref even at 100 nA; rate
  capped at 1000/t_ref Hz)
- membrane decays toward rest with no input (exponential, monotonic,
  settles at V_rest)
- firing rate increases monotonically with input current
- reset after spike (V = V_reset on the spike step)
- rheobase matches theory (1.4 nA silent, 1.6 nA spiking)
- longer tau_m and higher V_th both lower the firing rate
- step response spikes only after the step
- sub-threshold steady state equals V_rest + R*I
