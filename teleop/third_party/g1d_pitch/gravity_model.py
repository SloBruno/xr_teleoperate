"""Gravity-torque feedforward model for the G1-D waist pitch joint, built
from a REAL URDF RNEA computation (see compute_gravity_table.py -- same
technique the TEACH gravity compensation elsewhere in this codebase uses:
q_dot=0, q_ddot=0, external force=0 => tau = g(q)), run once offline on the
robot's Pinocchio-equipped env and baked in here as a plain lookup table so
the control loop and test suite never need Pinocchio at runtime.

KNOWN LIMITATION (do not remove this comment without re-validating): the
table was computed with every OTHER joint locked at q=0 (a nominal pose,
not the arms' live pose), because this isolated pitch tool does not
subscribe to full-body lowstate. Cross-checked against two real measured
holding-torque samples:
  q=+0.0715 rad: table/interp predicts ~-10.6 N.m, measured tau_est was
    ~-9.8 to -10.5 N.m (good agreement).
  q=+0.9574 rad: table/interp predicts ~-40.9 N.m, measured tau_est was
    ~-27.5 N.m (~33% overestimate).
Given that measured gap, GRAVITY_FF_SCALE below defaults to a conservative
<1.0 so the feedforward UNDER-compensates rather than over-pushes past the
real gravity torque -- the position/damping terms (kp/kd) and the existing
safety nets (dq_abort, tracking_error, max_lead) still do the rest of the
work and remain the actual protection. This is an aid to reduce how hard
the PD loop has to fight gravity, not a certified torque source.
"""
from math import isfinite

# (q_rad, tau_gravity_Nm) computed via Pinocchio RNEA on the real G1-D URDF,
# Yaw_Joint (the URDF's name for the waist PITCH axis -- limits match
# rt/waistpitchcmd's vendor range exactly), all other joints at q=0.
GRAVITY_TABLE = [(-0.0436, -5.8528), (0.0064, -8.0844), (0.0564, -10.2959), (0.1064, -12.4816), (0.1564, -14.6362), (0.2064, -16.7541), (0.2564, -18.8302), (0.3064, -20.8592), (0.3564, -22.8361), (0.4064, -24.7558), (0.4564, -26.6138), (0.5064, -28.4052), (0.5564, -30.1255), (0.6064, -31.7706), (0.6564, -33.3363), (0.7064, -34.8187), (0.7564, -36.214), (0.8064, -37.5188), (0.8564, -38.7299), (0.9064, -39.8441), (0.9564, -40.8588), (1.0064, -41.7713), (1.0564, -42.5794), (1.1064, -43.2811), (1.1564, -43.8746), (1.2064, -44.3584), (1.2564, -44.7314), (1.3064, -44.9926), (1.3564, -45.1413), (1.4064, -45.1772), (1.4564, -45.1001), (1.5064, -44.9104), (1.5564, -44.6083), (1.6064, -44.1948), (1.6564, -43.6708), (1.7064, -43.0377), (1.7564, -42.297), (1.8064, -41.4506), (1.8564, -40.5005), (1.9064, -39.4493), (1.9564, -38.2994), (2.0064, -37.0538), (2.0564, -35.7156), (2.1064, -34.2881), (2.1564, -32.7749), (2.2064, -31.1798), (2.2564, -29.5068), (2.3064, -27.76)]

# Conservative default scale (< 1.0, see the measured-vs-modeled gap above).
GRAVITY_FF_SCALE_DEFAULT = 0.6

# Hard ceiling on the feedforward magnitude itself, well under the URDF's
# declared effort=80 N.m for Yaw_Joint and under the table's own peak
# (~45 N.m) -- a defense-in-depth clamp independent of the scale factor.
GRAVITY_FF_HARD_CEILING_NM = 50.0


def gravity_feedforward_tau(q, table=GRAVITY_TABLE, scale=GRAVITY_FF_SCALE_DEFAULT,
                             ceiling=GRAVITY_FF_HARD_CEILING_NM):
    """Linear interpolation over `table`, scaled and clamped. Pure function,
    no I/O, no Pinocchio dependency -- safe to call every control tick.
    Clamps q to the table's own domain (never extrapolates past measured
    range) and clamps the final output to +/- ceiling regardless of scale.
    """
    if not isfinite(q):
        return 0.0
    if not table:
        return 0.0
    lo_q, lo_tau = table[0]
    hi_q, hi_tau = table[-1]
    if q <= lo_q:
        tau = lo_tau
    elif q >= hi_q:
        tau = hi_tau
    else:
        tau = lo_tau
        for i in range(len(table) - 1):
            q0, t0 = table[i]
            q1, t1 = table[i + 1]
            if q0 <= q <= q1:
                frac = 0.0 if q1 == q0 else (q - q0) / (q1 - q0)
                tau = t0 + frac * (t1 - t0)
                break
    tau *= scale
    if ceiling is not None:
        tau = max(-ceiling, min(ceiling, tau))
    return tau
