#!/usr/bin/env python3

# Runs Ramsete simulation on decoupled model with right and left wheel
# velocities as states

# Avoid needing display if plots aren't being shown
import sys

if "--noninteractive" in sys.argv:
    import matplotlib as mpl

    mpl.use("svg")
    import bookutil.latex as latex

import control as ct
import frccontrol as fct
import math
import matplotlib.pyplot as plt
import numpy as np

from bookutil.pose2d import Pose2d


def drivetrain(motor, num_motors, m, r, rb, J, Gl, Gr):
    """Returns the state-space model for a drivetrain.

    States: [[left velocity], [right velocity]]
    Inputs: [[left voltage], [right voltage]]
    Outputs: [[left velocity], [right velocity]]

    Keyword arguments:
    motor -- instance of DcBrushedMotor
    num_motors -- number of motors driving the mechanism
    m -- mass of robot in kg
    r -- radius of wheels in meters
    rb -- radius of robot in meters
    J -- moment of inertia of the drivetrain in kg-m^2
    Gl -- gear ratio of left side of drivetrain
    Gr -- gear ratio of right side of drivetrain

    Returns:
    StateSpace instance containing continuous model
    """
    motor = fct.models.gearbox(motor, num_motors)

    C1 = -(Gl ** 2) * motor.Kt / (motor.Kv * motor.R * r ** 2)
    C2 = Gl * motor.Kt / (motor.R * r)
    C3 = -(Gr ** 2) * motor.Kt / (motor.Kv * motor.R * r ** 2)
    C4 = Gr * motor.Kt / (motor.R * r)
    # fmt: off
    A = np.array([[(1 / m + rb**2 / J) * C1, (1 / m - rb**2 / J) * C3],
                  [(1 / m - rb**2 / J) * C1, (1 / m + rb**2 / J) * C3]])
    B = np.array([[(1 / m + rb**2 / J) * C2, (1 / m - rb**2 / J) * C4],
                  [(1 / m - rb**2 / J) * C2, (1 / m + rb**2 / J) * C4]])
    C = np.array([[1, 0],
                  [0, 1]])
    D = np.array([[0, 0],
                  [0, 0]])
    # fmt: on

    return ct.ss(A, B, C, D)


def ramsete(pose_desired, v_desired, omega_desired, pose, b, zeta):
    e = pose_desired.relative_to(pose)

    k = 2 * zeta * math.sqrt(omega_desired ** 2 + b * v_desired ** 2)
    v = v_desired * math.cos(e.theta) + k * e.x
    omega = omega_desired + k * e.theta + b * v_desired * np.sinc(e.theta) * e.y

    return v, omega


def get_diff_vels(v, omega, d):
    """Returns left and right wheel velocities given a central velocity and
    turning rate.

    Keyword arguments:
    v -- center velocity
    omega -- center turning rate
    d -- trackwidth
    """
    return v - omega * d / 2.0, v + omega * d / 2.0


class Drivetrain(fct.System):
    def __init__(self, dt):
        """Drivetrain subsystem.

        Keyword arguments:
        dt -- time between model/controller updates
        """
        state_labels = [("Left velocity", "m/s"), ("Right velocity", "m/s")]
        u_labels = [("Left voltage", "V"), ("Right voltage", "V")]
        self.set_plot_labels(state_labels, u_labels)

        u_min = np.array([[-12.0], [-12.0]])
        u_max = np.array([[12.0], [12.0]])
        fct.System.__init__(self, u_min, u_max, dt, np.zeros((2, 1)), np.zeros((2, 1)))

    def create_model(self, states, inputs):
        self.in_low_gear = False

        # Number of motors per side
        num_motors = 2.0

        # High and low gear ratios of drivetrain
        Glow = 60.0 / 11.0
        Ghigh = 60.0 / 11.0

        # Drivetrain mass in kg
        m = 52
        # Radius of wheels in meters
        r = 0.08255 / 2.0
        # Radius of robot in meters
        self.rb = 0.59055 / 2.0
        # Moment of inertia of the drivetrain in kg-m^2
        J = 6.0

        # Gear ratios of left and right sides of drivetrain respectively
        if self.in_low_gear:
            Gl = Glow
            Gr = Glow
        else:
            Gl = Ghigh
            Gr = Ghigh

        return drivetrain(fct.models.MOTOR_CIM, num_motors, m, r, self.rb, J, Gl, Gr)

    def design_controller_observer(self):
        if self.in_low_gear:
            q_vel = 1.0
        else:
            q_vel = 0.95

        q = [q_vel, q_vel]
        r = [12.0, 12.0]
        self.design_lqr(q, r)

        qff_vel = 0.01
        self.design_two_state_feedforward([qff_vel, qff_vel], [12, 12])

        q_vel = 1.0
        r_vel = 0.01
        self.design_kalman_filter([q_vel, q_vel], [r_vel, r_vel])

        print("ctrb cond =", np.linalg.cond(ct.ctrb(self.sysd.A, self.sysd.B)))


def main():
    dt = 0.00505
    drivetrain = Drivetrain(dt)

    t, xprof, vprof, aprof = fct.generate_s_curve_profile(
        max_v=4.0, max_a=3.5, time_to_max_a=1.0, dt=dt, goal=10.0
    )

    # Generate references for LQR
    refs = []
    for i in range(len(t)):
        r = np.array([[vprof[i]], [vprof[i]]])
        refs.append(r)

    # Run LQR
    state_rec, ref_rec, u_rec, y_rec = drivetrain.generate_time_responses(t, refs)
    drivetrain.plot_time_responses(t, state_rec, ref_rec, u_rec)
    if "--noninteractive" in sys.argv:
        latex.savefig("ramsete_decoupled_vel_lqr_profile")

    # Initial robot pose
    pose = Pose2d(2, 0, np.pi / 2.0)
    desired_pose = Pose2d()

    # Ramsete tuning constants
    b = 2
    zeta = 0.7

    vl = float("inf")
    vr = float("inf")

    x_rec = []
    y_rec = []
    vref_rec = []
    omegaref_rec = []
    v_rec = []
    omega_rec = []
    ul_rec = []
    ur_rec = []

    # Log initial data for plots
    vref_rec.append(0)
    omegaref_rec.append(0)
    x_rec.append(pose.x)
    y_rec.append(pose.y)
    ul_rec.append(drivetrain.u[0, 0])
    ur_rec.append(drivetrain.u[1, 0])
    v_rec.append(0)
    omega_rec.append(0)

    # Run Ramsete
    i = 0
    while i < len(t) - 1:
        desired_pose.x = 0
        desired_pose.y = xprof[i]
        desired_pose.theta = np.pi / 2.0

        # pose_desired, v_desired, omega_desired, pose, b, zeta
        vref, omegaref = ramsete(desired_pose, vprof[i], 0, pose, b, zeta)
        vl, vr = get_diff_vels(vref, omegaref, drivetrain.rb * 2.0)
        next_r = np.array([[vl], [vr]])
        drivetrain.update(next_r)
        vc = (drivetrain.x[0, 0] + drivetrain.x[1, 0]) / 2.0
        omega = (drivetrain.x[1, 0] - drivetrain.x[0, 0]) / (2.0 * drivetrain.rb)

        # Log data for plots
        vref_rec.append(vref)
        omegaref_rec.append(omegaref)
        x_rec.append(pose.x)
        y_rec.append(pose.y)
        ul_rec.append(drivetrain.u[0, 0])
        ur_rec.append(drivetrain.u[1, 0])
        v_rec.append(vc)
        omega_rec.append(omega)

        # Update nonlinear observer
        pose.x += vc * math.cos(pose.theta) * dt
        pose.y += vc * math.sin(pose.theta) * dt
        pose.theta += omega * dt

        if i < len(t) - 1:
            i += 1

    plt.figure(2)
    plt.plot([0] * len(t), xprof, label="Reference trajectory")
    plt.plot(x_rec, y_rec, label="Ramsete controller")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend()

    # Equalize aspect ratio
    xlim = plt.xlim()
    width = abs(xlim[0]) + abs(xlim[1])
    ylim = plt.ylim()
    height = abs(ylim[0]) + abs(ylim[1])
    if width > height:
        plt.ylim([-width / 2, width / 2])
    else:
        plt.xlim([-height / 2, height / 2])

    if "--noninteractive" in sys.argv:
        latex.savefig("ramsete_decoupled_response")

    plt.figure(3)
    num_plots = 4
    plt.subplot(num_plots, 1, 1)
    plt.title("Time domain responses")
    plt.ylabel(
        "Velocity (m/s)",
        horizontalalignment="right",
        verticalalignment="center",
        rotation=45,
    )
    plt.plot(t, vref_rec, label="Reference")
    plt.plot(t, v_rec, label="Estimated state")
    plt.legend()
    plt.subplot(num_plots, 1, 2)
    plt.ylabel(
        "Angular rate (rad/s)",
        horizontalalignment="right",
        verticalalignment="center",
        rotation=45,
    )
    plt.plot(t, omegaref_rec, label="Reference")
    plt.plot(t, omega_rec, label="Estimated state")
    plt.legend()
    plt.subplot(num_plots, 1, 3)
    plt.ylabel(
        "Left voltage (V)",
        horizontalalignment="right",
        verticalalignment="center",
        rotation=45,
    )
    plt.plot(t, ul_rec, label="Control effort")
    plt.legend()
    plt.subplot(num_plots, 1, 4)
    plt.ylabel(
        "Right voltage (V)",
        horizontalalignment="right",
        verticalalignment="center",
        rotation=45,
    )
    plt.plot(t, ur_rec, label="Control effort")
    plt.legend()
    plt.xlabel("Time (s)")

    if "--noninteractive" in sys.argv:
        latex.savefig("ramsete_decoupled_vel_lqr_response")
    else:
        plt.show()


if __name__ == "__main__":
    main()
