#!/usr/bin/env python3

# Avoid needing display if plots aren't being shown
import sys

if "--noninteractive" in sys.argv:
    import matplotlib as mpl

    mpl.use("svg")
import bookutil.latex as latex

import frccontrol as fct
import math
import matplotlib.pyplot as plt
import numpy as np

plt.rc("text", usetex=True)


class Elevator(fct.System):
    def __init__(self, dt):
        """Elevator subsystem.

        Keyword arguments:
        dt -- time between model/controller updates
        """
        state_labels = [("Position", "m"), ("Velocity", "m/s")]
        u_labels = [("Voltage", "V")]
        self.set_plot_labels(state_labels, u_labels)

        fct.System.__init__(
            self,
            np.array([[-12.0]]),
            np.array([[12.0]]),
            dt,
            np.zeros((2, 1)),
            np.zeros((1, 1)),
        )

    def create_model(self, states, inputs):
        # Number of motors
        num_motors = 2.0
        # Elevator carriage mass in kg
        m = 6.803886
        # Radius of pulley in meters
        r = 0.02762679089
        # Gear ratio
        G = 42.0 / 12.0 * 40.0 / 14.0

        return fct.models.elevator(fct.models.MOTOR_CIM, num_motors, m, r, G)

    def design_controller_observer(self):
        q = [0.02, 0.4]
        r = [12.0]
        self.design_lqr(q, r)

        q_pos = 0.05
        q_vel = 1.0
        r_pos = 0.0001
        self.design_kalman_filter([q_pos, q_vel], [r_pos])


def main():
    dt = 0.00505
    elevator = Elevator(dt)

    # Set up graphing
    l0 = 0.1
    l1 = l0 + 5.0
    l2 = l1 + 0.1
    t = np.arange(0, l2 + 5.0, dt)

    t, xprof, vprof, aprof = fct.generate_trapezoid_profile(
        max_v=0.5, time_to_max_v=0.5, dt=dt, goal=3
    )

    refs = []

    # Generate references for simulation
    for i in range(len(t)):
        r = np.array([[xprof[i]], [vprof[i]]])
        refs.append(r)

    x_rec, ref_rec, u_rec, y_rec = elevator.generate_time_responses(t, refs)
    latex.plot_time_responses(
        elevator, t, x_rec, ref_rec, u_rec, 2, use_pid_labels=True
    )

    if "--noninteractive" in sys.argv:
        latex.savefig("pd_controller")
    else:
        plt.show()


if __name__ == "__main__":
    main()
