#!/usr/bin/env python3

# Runs Extended Rauch-Tung-Striebel controller on differential drive
# https://file.tavsys.net/control/papers/Extended%20Rauch-Tung-Striebel%20Controller%2C%20ZAGSS.pdf

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
import scipy as sp

DT = 0.02


def lerp(a, b, t):
    return a + t * (b - a)


def get_square_refs():
    STATES = 5
    refs = np.zeros((STATES, 0))
    v = 2
    pts = [
        np.array([[0], [0]]),
        np.array([[2], [0]]),
        np.array([[2], [6]]),
        np.array([[-4], [6]]),
        np.array([[-4], [10]]),
        np.array([[10], [10]]),
        np.array([[10], [4]]),
        np.array([[-3], [4]]),
        np.array([[-3], [0]]),
    ]
    for i in range(len(pts)):
        pt0 = pts[i]
        if i + 1 < len(pts):
            pt1 = pts[i + 1]
        else:
            pt1 = pts[0]
        diff = pt1 - pt0
        t = math.hypot(diff[0, 0], diff[1, 0]) / v

        num_pts = int(t / DT)
        for j in range(num_pts):
            mid = lerp(pt0, pt1, j / num_pts)
            ref = np.array(
                [
                    [mid[0, 0]],
                    [mid[1, 0]],
                    [math.atan2(diff[1, 0], diff[0, 0])],
                    [v],
                    [v],
                ]
            )
            refs = np.hstack((refs, ref))
    return refs


def differential_drive(motor, num_motors, m, r, rb, J, Gl, Gr, states):
    """Returns the state-space model for a differential drive.

    States: [[x], [y], [theta], [left velocity], [right velocity]]
    Inputs: [[left voltage], [right voltage]]
    Outputs: [[theta], [left velocity], [right velocity]]

    Keyword arguments:
    motor -- instance of DcBrushedMotor
    num_motors -- number of motors driving the mechanism
    m -- mass of robot in kg
    r -- radius of wheels in meters
    rb -- radius of robot in meters
    J -- moment of inertia of the differential drive in kg-m^2
    Gl -- gear ratio of left side of differential drive
    Gr -- gear ratio of right side of differential drive
    states -- state vector around which to linearize model

    Returns:
    StateSpace instance containing continuous model
    """
    motor = fct.models.gearbox(motor, num_motors)

    C1 = -(Gl ** 2) * motor.Kt / (motor.Kv * motor.R * r ** 2)
    C2 = Gl * motor.Kt / (motor.R * r)
    C3 = -(Gr ** 2) * motor.Kt / (motor.Kv * motor.R * r ** 2)
    C4 = Gr * motor.Kt / (motor.R * r)
    x = states[0, 0]
    y = states[1, 0]
    theta = states[2, 0]
    vl = states[3, 0]
    vr = states[4, 0]
    v = (vr + vl) / 2.0
    if abs(v) < 1e-9:
        vl = 1e-9
        vr = 1e-9
        v = 1e-9
    # fmt: off
    A = np.array([[0, 0, 0, 0.5, 0.5],
                  [0, 0, v, 0, 0],
                  [0, 0, 0, -0.5 / rb, 0.5 / rb],
                  [0, 0, 0, (1 / m + rb**2 / J) * C1, (1 / m - rb**2 / J) * C3],
                  [0, 0, 0, (1 / m - rb**2 / J) * C1, (1 / m + rb**2 / J) * C3]])
    B = np.array([[0, 0],
                  [0, 0],
                  [0, 0],
                  [(1 / m + rb**2 / J) * C2, (1 / m - rb**2 / J) * C4],
                  [(1 / m - rb**2 / J) * C2, (1 / m + rb**2 / J) * C4]])
    C = np.array([[0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]])
    D = np.array([[0, 0],
                  [0, 0],
                  [0, 0]])
    # fmt: on

    return ct.StateSpace(A, B, C, D, remove_useless=False)


def get_diff_vels(v, omega, d):
    """Returns left and right wheel velocities given a central velocity and
    turning rate.
    Keyword arguments:
    v -- center velocity
    omega -- center turning rate
    d -- trackwidth
    """
    return v - omega * d / 2.0, v + omega * d / 2.0


class DifferentialDrive(fct.System):
    def __init__(self, dt, states):
        """Drivetrain subsystem.

        Keyword arguments:
        dt -- time between model/controller updates
        states -- state vector around which to linearize model
        """
        state_labels = [
            ("x position", "m"),
            ("y position", "m"),
            ("Heading", "rad"),
            ("Left velocity", "m/s"),
            ("Right velocity", "m/s"),
        ]
        u_labels = [("Left voltage", "V"), ("Right voltage", "V")]
        self.set_plot_labels(state_labels, u_labels)

        u_min = np.array([[-12.0], [-12.0]])
        u_max = np.array([[12.0], [12.0]])

        f = (
            lambda x, u: np.array(
                [
                    [(x[3, 0] + x[4, 0]) / 2.0 * math.cos(x[2, 0])],
                    [(x[3, 0] + x[4, 0]) / 2.0 * math.sin(x[2, 0])],
                    [(x[4, 0] - x[3, 0]) / (2.0 * self.rb)],
                    [self.sysc.A[3, 3] * x[3, 0] + self.sysc.A[3, 4] * x[4, 0]],
                    [self.sysc.A[4, 3] * x[3, 0] + self.sysc.A[4, 4] * x[4, 0]],
                ]
            )
            + self.sysc.B @ u
        )
        self.f = f
        fct.System.__init__(
            self, u_min, u_max, dt, states, np.zeros((2, 1)), nonlinear_func=f
        )

    def create_model(self, states, inputs):
        """Relinearize model around given state.

        Keyword arguments:
        states -- state vector around which to linearize model
        inputs -- input vector around which to linearize model

        Returns:
        StateSpace instance containing continuous state-space model
        """
        self.in_low_gear = False

        # Number of motors per side
        num_motors = 3.0

        # High and low gear ratios of differential drive
        Glow = 60.0 / 11.0
        Ghigh = 60.0 / 11.0

        # Drivetrain mass in kg
        m = 52
        # Radius of wheels in meters
        r = 0.08255 / 2.0
        # Radius of robot in meters
        self.rb = 0.59055 / 2.0
        # Moment of inertia of the differential drive in kg-m^2
        J = 6.0

        # Gear ratios of left and right sides of differential drive respectively
        if self.in_low_gear:
            Gl = Glow
            Gr = Glow
        else:
            Gl = Ghigh
            Gr = Ghigh

        return differential_drive(
            fct.models.MOTOR_CIM,
            num_motors,
            m,
            r,
            self.rb,
            J,
            Gl,
            Gr,
            np.asarray(states),
        )

    def design_controller_observer(self):
        STATES = self.x_hat.shape[0]
        INPUTS = 2

        q_x = 0.0625
        q_y = 0.125
        q_heading = 10.0
        q_vel = 0.95
        q = [q_x, q_y, q_heading, q_vel, q_vel]
        Q = np.diag(1.0 / np.square(q))
        self.Qinv = np.linalg.inv(Q)

        r = [12.0, 12.0]
        R = np.diag(1.0 / np.square(r))
        self.Rinv = np.linalg.inv(R)

        self.dt = DT
        self.t = 0

        # Get reference trajectory
        self.refs = np.zeros((STATES, 0))
        with open("ramsete_traj.csv", "r") as trajectory_file:
            import csv

            current_t = 0

            reader = csv.reader(trajectory_file, delimiter=",")
            trajectory_file.readline()
            for row in reader:
                x = float(row[1])
                y = float(row[2])
                theta = float(row[3])
                vl, vr = get_diff_vels(float(row[4]), float(row[5]), self.rb * 2.0)
                ref = np.array([[x], [y], [theta], [vl], [vr]])
                self.refs = np.hstack((self.refs, ref))

        self.refs = get_square_refs()

        q_pos = 0.05
        q_heading = 10.0
        q_vel = 1.0
        r_gyro = 0.0001
        r_vel = 0.01
        self.design_kalman_filter(
            [q_pos, q_pos, q_heading, q_vel, q_vel], [r_gyro, r_vel, r_vel]
        )

        # Initialize matrix storage
        STATES = 5
        INPUTS = 2
        self.x_hat_pre_rec = np.zeros((STATES, 1, self.refs.shape[1]))
        self.x_hat_post_rec = np.zeros((STATES, 1, self.refs.shape[1]))
        self.A_rec = np.zeros((STATES, STATES, self.refs.shape[1]))
        self.B_rec = np.zeros((STATES, INPUTS, self.refs.shape[1]))
        self.P_pre_rec = np.zeros((STATES, STATES, self.refs.shape[1]))
        self.P_post_rec = np.zeros((STATES, STATES, self.refs.shape[1]))
        self.x_hat_smooth_rec = np.zeros((STATES, 1, self.refs.shape[1]))

    def update_controller(self, next_r):
        STATES = self.x_hat.shape[0]
        INPUTS = 2

        # Since this is the last reference, there are no reference dynamics to
        # follow
        if self.t == self.refs.shape[1] - 1:
            self.u = np.zeros((INPUTS, 1))
            return

        x_hat = self.x_hat
        P = np.zeros((x_hat.shape[0], x_hat.shape[0]))

        self.x_hat_pre_rec[:, :, self.t] = x_hat
        self.P_pre_rec[:, :, self.t] = P
        self.x_hat_post_rec[:, :, self.t] = x_hat
        self.P_post_rec[:, :, self.t] = P

        # Linearize model
        Ac = fct.numerical_jacobian_x(
            STATES, STATES, self.f, self.x_hat, np.zeros((INPUTS, 1))
        )
        Bc = fct.numerical_jacobian_u(
            STATES, INPUTS, self.f, np.zeros((STATES, 1)), np.zeros((INPUTS, 1))
        )
        A = sp.linalg.expm(Ac * self.dt)
        B = np.linalg.pinv(Ac) @ (A - np.eye(STATES)) @ Bc
        self.B_rec[:, :, self.t] = B

        C = np.eye(STATES, STATES)

        # Filter
        N = min(self.refs.shape[1] - 1, self.t + 50)
        for k in range(self.t + 1, N + 1):
            old_x_hat = np.copy(x_hat)
            x_hat = fct.runge_kutta(self.f, old_x_hat, np.zeros((INPUTS, 1)), self.dt)
            # Linearize model
            Ac = fct.numerical_jacobian_x(
                STATES, STATES, self.f, old_x_hat, np.zeros((INPUTS, 1))
            )
            A = sp.linalg.expm(Ac * self.dt)
            B = np.linalg.pinv(Ac) @ (A - np.eye(STATES)) @ Bc

            P = A @ P @ A.T + B @ self.Rinv @ B.T

            self.x_hat_pre_rec[:, :, k] = x_hat
            self.P_pre_rec[:, :, k] = P
            self.A_rec[:, :, k] = A
            self.B_rec[:, :, k] = B

            K = P @ C.T @ np.linalg.inv(C @ P @ C.T + C @ self.Qinv @ C.T)
            x_hat += K @ (self.refs[:, k : k + 1] - C @ x_hat)
            P = (np.eye(STATES, STATES) - K @ C) @ P

            self.x_hat_post_rec[:, :, k] = x_hat
            self.P_post_rec[:, :, k] = P

        # Smoother

        # Last estimate is already optimal, so add it to the record
        self.x_hat_smooth_rec[:, :, N] = self.x_hat_post_rec[:, :, N]

        for k in range(N - 1, (self.t + 1) - 1, -1):
            K = (
                self.P_post_rec[:, :, k]
                @ self.A_rec[:, :, k].T
                @ np.linalg.pinv(self.P_pre_rec[:, :, k + 1])
            )
            x_hat = self.x_hat_post_rec[:, :, k] + K @ (
                self.x_hat_smooth_rec[:, :, k + 1] - self.x_hat_pre_rec[:, :, k + 1]
            )

            self.x_hat_smooth_rec[:, :, k] = x_hat

        self.u = np.linalg.pinv(self.B_rec[:, :, self.t]) @ (
            self.x_hat_smooth_rec[:, :, self.t + 1]
            - fct.runge_kutta(self.f, self.x_hat, np.zeros((INPUTS, 1)), self.dt)
        )

        u_cap = np.max(np.abs(self.u))
        if u_cap > 12.0:
            self.u = self.u / u_cap * 12.0
        self.r = next_r
        self.t += 1

        print(f"\riteration: {self.t}/{self.refs.shape[1] - 1}", end="")


def main():
    t = []
    refs = []

    # Radius of robot in meters
    rb = 0.59055 / 2.0

    with open("ramsete_traj.csv", "r") as trajectory_file:
        import csv

        current_t = 0

        reader = csv.reader(trajectory_file, delimiter=",")
        trajectory_file.readline()
        for row in reader:
            t.append(float(row[0]))
            x = float(row[1])
            y = float(row[2])
            theta = float(row[3])
            vl, vr = get_diff_vels(float(row[4]), float(row[5]), rb * 2.0)
            ref = np.array([[x], [y], [theta], [vl], [vr]])
            refs.append(ref)

    refs_tmp = get_square_refs()
    refs = []
    for i in range(refs_tmp.shape[1]):
        refs.append(refs_tmp[:, i : i + 1])
    t = [0]
    for i in range(len(refs) - 1):
        t.append(t[-1] + DT)

    dt = DT
    # x = np.array([[refs[0][0, 0] + 0.5], [refs[0][1, 0] + 0.5], [np.pi / 2], [0], [0]])
    x = np.array([[4], [3], [3 * np.pi / 2], [0], [0]])
    diff_drive = DifferentialDrive(dt, x)

    state_rec, ref_rec, u_rec, y_rec = diff_drive.generate_time_responses(t, refs)
    print("")

    plt.figure(1)
    x_rec = np.squeeze(np.asarray(state_rec[0, :]))
    y_rec = np.squeeze(np.asarray(state_rec[1, :]))
    plt.plot(x_rec, y_rec, label="ERTS controller")
    plt.plot(ref_rec[0, :], ref_rec[1, :], label="Reference trajectory")
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
        latex.savefig("ltv_diff_drive_approx_xy")

    diff_drive.plot_time_responses(t, state_rec, ref_rec, u_rec)

    if "--noninteractive" in sys.argv:
        latex.savefig("ltv_diff_drive_approx_response")
    else:
        plt.show()


if __name__ == "__main__":
    main()
