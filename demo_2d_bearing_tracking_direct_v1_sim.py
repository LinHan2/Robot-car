#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure-Python simulation of demo_2d_bearing_tracking_direct_v1.m.

This file intentionally has no ROS dependency.  It reproduces the MATLAB
target/pursuer plant, RBF identifier, finite-window history stack, Riccati
observer and direct relative-tracking controller before deploying to robots.

Example:
    python3 demo_2d_bearing_tracking_direct_v1_sim.py --duration 30

The default algorithm parameters match the original MATLAB file.  By default
there is no velocity saturation, also matching MATLAB.  Use
``--max-pursuer-speed`` to study the effect of a real chassis speed limit.
"""

import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import yaml


DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "bearing_tracking_params.yaml",
)


def limit_vector_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    """A non-positive maximum means no saturation."""
    norm = float(np.linalg.norm(vector))
    if maximum > 0.0 and norm > maximum:
        return vector * (maximum / norm)
    return vector


@dataclass
class Parameters:
    # Matches the MATLAB parameter block unless overridden from the CLI.
    dt: float = 0.01
    duration: float = 30.0
    rd_radius: float = 0.7
    wd: float = 0.3
    phi0: float = 2.0 * math.pi / 3.0
    kr: float = 0.25

    q: int = 30
    sigma: float = 0.52
    delta_t: float = 1.0
    nh: int = 30
    max_history_execution_error: float = 0.0
    max_history_target_execution_error: float = 0.0
    history_forgetting_factor: float = 1.0
    identifier_use_relative_velocity: bool = False
    gamma: float = 1.0e-7
    ac: float = 1.0e-5
    valid_tol: float = 100.0 * np.finfo(float).eps
    kc: float = 2.0
    kh: float = 1.2
    max_history_condition: float = 1.0e3
    sigma_theta: float = 0.003

    af: float = 5.0
    lambda_s: float = 0.30
    ro: float = 0.02
    p_floor: float = 1.0e-9
    mocap_position_observer_gain: float = 0.0
    mocap_relative_velocity_observer_gain: float = 0.0

    # MATLAB targetInput(b): vd*d + ve*b + vj*J*b.
    vd: float = 0.0
    ve: float = 0.1
    vj: float = 0.1
    direction_deg: float = 155.0

    # Set to 0 to faithfully reproduce MATLAB's unsaturated input.
    max_pursuer_speed: float = 0.0
    max_target_speed: float = 0.0
    # False 保持论文 bearing-only；True 只把 Kr 反馈项换成真实相对位置。
    use_true_relative_feedback: bool = False
    target_soft_boundary: bool = False
    field_x_min: float = -1.5
    field_x_max: float = 1.5
    field_y_min: float = -1.5
    field_y_max: float = 1.5
    target_boundary_margin: float = 0.30
    target_boundary_speed: float = 0.15

    @property
    def nq(self) -> int:
        return 2 * self.q

    @property
    def nw(self) -> int:
        count = int(round(self.delta_t / self.dt))
        if count <= 0 or abs(count * self.dt - self.delta_t) > 1.0e-9:
            raise ValueError("delta_t/dt must be an integer")
        return count


@dataclass
class State:
    p_t: np.ndarray
    p_o: np.ndarray
    theta_hat: np.ndarray
    u_t_hat: np.ndarray
    xi_hat: np.ndarray
    po: np.ndarray


def state_add(state: State, derivative: State, scale: float) -> State:
    return State(
        state.p_t + scale * derivative.p_t,
        state.p_o + scale * derivative.p_o,
        state.theta_hat + scale * derivative.theta_hat,
        state.u_t_hat + scale * derivative.u_t_hat,
        state.xi_hat + scale * derivative.xi_hat,
        state.po + scale * derivative.po,
    )


def project_spd(matrix: np.ndarray, floor_value: float) -> np.ndarray:
    matrix = (matrix + matrix.T) * 0.5
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(np.real(values), floor_value)
    projected = vectors @ np.diag(values) @ vectors.T
    return (projected + projected.T) * 0.5


class HistoryStack:
    """Spectral history stack from MATLAB functions updateHistory/spectrum."""

    def __init__(self, p: Parameters):
        self.p = p
        self.windows: List[np.ndarray] = []
        self.vectors: List[np.ndarray] = []
        self.pi = np.zeros((p.nq, p.nq), dtype=float)
        self.theta_gamma = np.zeros(p.nq, dtype=float)
        self.condition = 0.0
        self.update_gated = False

    @staticmethod
    def spectrum(matrix: np.ndarray) -> np.ndarray:
        matrix = (matrix + matrix.T) * 0.5
        values = np.real(np.linalg.eigvalsh(matrix))
        tolerance = 100.0 * np.finfo(float).eps * max(1.0, np.linalg.norm(matrix, 2))
        values[np.abs(values) < tolerance] = 0.0
        return np.sort(values)

    @staticmethod
    def lex_greater(left: np.ndarray, right: np.ndarray) -> bool:
        for lv, rv in zip(left, right):
            scale = max(1.0, abs(float(lv)), abs(float(rv)))
            if lv > rv + 1.0e-12 * scale:
                return True
            if lv < rv - 1.0e-12 * scale:
                return False
        return False

    def add(self, w_new: np.ndarray, z_new: np.ndarray) -> None:
        if len(self.windows) < self.p.nh:
            self.windows.append(w_new.copy())
            self.vectors.append(z_new.copy())
        else:
            aggregate = np.sum(self.windows, axis=0)
            best = self.spectrum(aggregate)
            replacement = -1
            for index in range(self.p.nh):
                candidate = aggregate - self.windows[index] + w_new
                candidate_spectrum = self.spectrum(candidate)
                if self.lex_greater(candidate_spectrum, best):
                    best = candidate_spectrum
                    replacement = index
            if replacement >= 0:
                self.windows[replacement] = w_new.copy()
                self.vectors[replacement] = z_new.copy()
        if self.p.history_forgetting_factor < 1.0:
            factor = self.p.history_forgetting_factor
            for index in range(len(self.windows)):
                self.windows[index] *= factor
                self.vectors[index] *= factor
        self._recompute()

    def _recompute(self) -> None:
        if not self.windows:
            self.pi.fill(0.0)
            self.theta_gamma.fill(0.0)
            self.condition = 0.0
            self.update_gated = False
            return
        wh = np.sum(self.windows, axis=0)
        wh = (wh + wh.T) * 0.5
        zh = np.sum(self.vectors, axis=0)
        values, vectors = np.linalg.eigh(wh)
        active = np.real(values) >= self.p.gamma
        if not np.any(active):
            self.pi.fill(0.0)
            self.theta_gamma.fill(0.0)
            self.condition = 0.0
            self.update_gated = False
            return
        qa = np.real(vectors[:, active])
        la = np.real(values[active])
        self.condition = float(np.max(la) / np.min(la))
        self.update_gated = (
            self.p.max_history_condition > 0.0
            and self.condition > self.p.max_history_condition
        )
        if self.update_gated:
            return
        self.pi = qa @ qa.T
        self.pi = (self.pi + self.pi.T) * 0.5
        self.theta_gamma = qa @ ((qa.T @ zh) / la)


class CurrentWindow:
    """Sufficient statistics for one MATLAB finite window."""

    def __init__(self, b0: np.ndarray, p: Parameters):
        self.p = p
        self.b0 = b0.copy()
        self.g = np.zeros((2, p.nq), dtype=float)
        self.u = np.zeros(2, dtype=float)
        self.a = 0.0
        self.cf = np.zeros(p.nq, dtype=float)
        self.ch = 0.0
        self.sff = np.zeros((p.nq, p.nq), dtype=float)
        self.sfh = np.zeros(p.nq, dtype=float)
        self.execution_error_integral = 0.0
        self.duration = 0.0
        self.count = 0

    def execution_error_rms(self) -> float:
        if self.duration <= 0.0:
            return 0.0
        return math.sqrt(self.execution_error_integral / self.duration)

    def statistics(self, threshold: float) -> Tuple[np.ndarray, np.ndarray, float]:
        if self.a < threshold:
            return (
                np.zeros((self.p.nq, self.p.nq), dtype=float),
                np.zeros(self.p.nq, dtype=float),
                0.0,
            )
        w = self.sff - np.outer(self.cf, self.cf) / self.a
        z = self.sfh - self.cf * self.ch / self.a
        return (w + w.T) * 0.5, z, 1.0

    def accumulate(self, pb: np.ndarray, phi: np.ndarray,
                   u_ident: np.ndarray, u_o_actual: np.ndarray,
                   u_o_command: np.ndarray, dt: float) -> None:
        f = pb @ self.g
        h = pb @ self.u
        v = pb @ self.b0
        self.a += dt * float(v @ v)
        self.cf += dt * (v @ f)
        self.ch += dt * float(v @ h)
        self.sff += dt * (f.T @ f)
        self.sfh += dt * (f.T @ h)
        self.g += dt * phi
        self.u += dt * u_ident
        execution_error = np.asarray(u_o_command) - np.asarray(u_o_actual)
        self.execution_error_integral += dt * float(execution_error @ execution_error)
        self.duration += dt
        self.count += 1


class BearingTrackingSimulation:
    def __init__(self, p: Parameters):
        self.p = p
        self.i2 = np.eye(2)
        self.i_nq = np.eye(p.nq)
        self.j = np.array([[0.0, -1.0], [1.0, 0.0]])
        angles = 2.0 * math.pi * np.arange(p.q) / p.q
        self.centers = np.vstack((np.cos(angles), np.sin(angles)))
        direction = math.radians(p.direction_deg)
        self.d = np.array([math.cos(direction), math.sin(direction)])

        # Same initial conditions as demo_2d_bearing_tracking_direct_v1.m.
        self.state = State(
            p_t=np.array([0.0, 0.0]),
            p_o=np.array([0.0, -1.5]),
            theta_hat=np.zeros(p.nq),
            u_t_hat=np.zeros(2),
            xi_hat=np.zeros(4),
            po=0.30 * np.eye(4),
        )
        self.history = HistoryStack(p)
        self.window = CurrentWindow(self._bearing(self.state)[2], p)

    def _bearing(self, state: State) -> Tuple[np.ndarray, float, np.ndarray]:
        relative = state.p_t - state.p_o
        rho = float(np.linalg.norm(relative))
        if rho < 1.0e-12:
            raise RuntimeError("bearing is undefined because target and pursuer coincide")
        return relative, rho, relative / rho

    def _rbf_regressor(self, b: np.ndarray) -> np.ndarray:
        error = self.centers - b.reshape(2, 1)
        phi = np.exp(-np.sum(error * error, axis=0) / (2.0 * self.p.sigma ** 2))
        return np.kron(self.i2, phi.reshape(1, -1))

    @staticmethod
    def _smoothstep(value: float) -> float:
        value = min(max(value, 0.0), 1.0)
        return value * value * (3.0 - 2.0 * value)

    def _target_boundary_command(self, position: np.ndarray) -> np.ndarray:
        """与 ROS 实车节点相同的 target 世界坐标软边界。"""
        p = self.p
        correction = np.zeros(2, dtype=float)
        if not p.target_soft_boundary or p.target_boundary_speed <= 0.0:
            return correction
        lower = np.array([p.field_x_min, p.field_y_min], dtype=float)
        upper = np.array([p.field_x_max, p.field_y_max], dtype=float)
        for axis in range(2):
            lower_start = lower[axis] + p.target_boundary_margin
            upper_start = upper[axis] - p.target_boundary_margin
            if position[axis] < lower_start:
                depth = (lower_start - position[axis]) / p.target_boundary_margin
                correction[axis] += (
                    p.target_boundary_speed * self._smoothstep(depth)
                )
            elif position[axis] > upper_start:
                depth = (position[axis] - upper_start) / p.target_boundary_margin
                correction[axis] -= (
                    p.target_boundary_speed * self._smoothstep(depth)
                )
        return correction

    def _desired_trajectory(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        angle = self.p.phi0 + self.p.wd * t
        rd = self.p.rd_radius * np.array([math.cos(angle), math.sin(angle)])
        rd_dot = self.p.rd_radius * self.p.wd * np.array([-math.sin(angle), math.cos(angle)])
        return rd, rd_dot

    def _signals(self, t: float, state: State) -> dict:
        relative, rho, b = self._bearing(state)
        pb = self.i2 - np.outer(b, b)
        phi = self._rbf_regressor(b)
        rd, rd_dot = self._desired_trajectory(t)
        u_t_nominal = self.p.vd * self.d + self.p.ve * b + self.p.vj * (self.j @ b)
        u_t = limit_vector_norm(
            u_t_nominal + self._target_boundary_command(state.p_t),
            self.p.max_target_speed,
        )
        feedback_position = (
            relative if self.p.use_true_relative_feedback
            else state.xi_hat[0:2]
        )
        u_o_raw = (
            state.u_t_hat + state.xi_hat[2:4] - rd_dot
            + self.p.kr * (feedback_position - rd)
        )
        u_o = limit_vector_norm(u_o_raw, self.p.max_pursuer_speed)
        return {
            "r": relative, "rho": rho, "b": b, "pb": pb, "phi": phi,
            "rd": rd, "rd_dot": rd_dot, "u_t": u_t, "u_o": u_o,
        }

    def _rhs(self, t: float, state: State, data: Tuple[np.ndarray, np.ndarray, float]) -> State:
        wc, zc, chi = data
        s = self._signals(t, state)
        p = self.p
        theta_dot = (
            p.kc * chi * (zc - wc @ state.theta_hat)
            + p.kh * (self.history.theta_gamma - self.history.pi @ state.theta_hat)
            - p.sigma_theta * (self.i_nq - self.history.pi) @ state.theta_hat
        )
        u_t_hat_dot = p.af * (s["phi"] @ state.theta_hat - state.u_t_hat)
        co = np.hstack((s["pb"], np.zeros((2, 2))))
        ro_inv = (1.0 / p.ro) * self.i2
        lo = state.po @ co.T @ ro_inv
        go = np.concatenate((state.u_t_hat - s["u_o"], np.zeros(2)))
        ao = np.zeros((4, 4))
        ao[0:2, 2:4] = self.i2
        po_dot = (
            ao @ state.po + state.po @ ao.T + p.lambda_s * state.po
            - state.po @ co.T @ ro_inv @ co @ state.po
        )
        xi_dot = ao @ state.xi_hat + go + lo @ (-co @ state.xi_hat)
        if p.use_true_relative_feedback and p.mocap_position_observer_gain > 0.0:
            xi_dot[0:2] += p.mocap_position_observer_gain * (
                s["r"] - state.xi_hat[0:2]
            )
        if p.mocap_relative_velocity_observer_gain > 0.0:
            predicted_relative_velocity = (
                state.u_t_hat - s["u_o"] + state.xi_hat[2:4]
            )
            xi_dot[2:4] += p.mocap_relative_velocity_observer_gain * (
                s["u_t"] - s["u_o"] - predicted_relative_velocity
            )
        return State(s["u_t"], s["u_o"], theta_dot, u_t_hat_dot, xi_dot, po_dot)

    def run(self) -> dict:
        p = self.p
        steps = int(round(p.duration / p.dt))
        times = np.arange(steps + 1) * p.dt
        target_log = np.zeros((steps + 1, 2))
        pursuer_log = np.zeros((steps + 1, 2))
        target_speed = np.zeros(steps + 1)
        pursuer_speed = np.zeros(steps + 1)
        input_error = np.zeros(steps + 1)
        localization_error = np.zeros(steps + 1)
        tracking_error = np.zeros(steps + 1)
        active_directions = np.zeros(steps + 1)

        for index, t in enumerate(times):
            data = self.window.statistics(p.ac)
            s = self._signals(t, self.state)
            target_log[index] = self.state.p_t
            pursuer_log[index] = self.state.p_o
            target_speed[index] = np.linalg.norm(s["u_t"])
            pursuer_speed[index] = np.linalg.norm(s["u_o"])
            input_error[index] = np.linalg.norm(s["u_t"] - self.state.u_t_hat)
            localization_error[index] = np.linalg.norm(s["r"] - self.state.xi_hat[0:2])
            tracking_error[index] = np.linalg.norm(s["r"] - s["rd"])
            active_directions[index] = np.linalg.matrix_rank(self.history.pi, tol=1.0e-8)
            if index == steps:
                break

            # MATLAB's left-rectangle update of one finite data window.
            # Ideal execution: dr/dt = u_t - u_o, so the corrected window target
            # Pb*(u_o + dr/dt) equals Pb*u_t when the flag is enabled. Execution
            # quality compares command with the actually executed velocity.
            u_ident = (
                s["u_o"] + (s["u_t"] - s["u_o"])
                if p.identifier_use_relative_velocity else s["u_o"]
            )
            self.window.accumulate(s["pb"], s["phi"], u_ident, s["u_o"], s["u_o"], p.dt)

            # RK4 integration of plant, identifier, filter and observer.
            k1 = self._rhs(t, self.state, data)
            k2 = self._rhs(t + p.dt / 2.0, state_add(self.state, k1, p.dt / 2.0), data)
            k3 = self._rhs(t + p.dt / 2.0, state_add(self.state, k2, p.dt / 2.0), data)
            k4 = self._rhs(t + p.dt, state_add(self.state, k3, p.dt), data)
            weighted = State(
                (k1.p_t + 2*k2.p_t + 2*k3.p_t + k4.p_t) / 6.0,
                (k1.p_o + 2*k2.p_o + 2*k3.p_o + k4.p_o) / 6.0,
                (k1.theta_hat + 2*k2.theta_hat + 2*k3.theta_hat + k4.theta_hat) / 6.0,
                (k1.u_t_hat + 2*k2.u_t_hat + 2*k3.u_t_hat + k4.u_t_hat) / 6.0,
                (k1.xi_hat + 2*k2.xi_hat + 2*k3.xi_hat + k4.xi_hat) / 6.0,
                (k1.po + 2*k2.po + 2*k3.po + k4.po) / 6.0,
            )
            self.state = state_add(self.state, weighted, p.dt)
            self.state.po = project_spd(self.state.po, p.p_floor)

            if self.window.count >= p.nw:
                w_new, z_new, valid = self.window.statistics(p.valid_tol)
                execution_error = self.window.execution_error_rms()
                if valid > 0.0 and (
                        p.max_history_execution_error <= 0.0
                        or execution_error <= p.max_history_execution_error):
                    self.history.add(w_new, z_new)
                self.window = CurrentWindow(self._bearing(self.state)[2], p)

        return {
            "time": times,
            "target": target_log,
            "pursuer": pursuer_log,
            "target_speed": target_speed,
            "pursuer_speed": pursuer_speed,
            "input_error": input_error,
            "localization_error": localization_error,
            "tracking_error": tracking_error,
            "active_directions": active_directions,
            "history_count": len(self.history.windows),
        }


def save_plot(results: dict, output: str, show: bool) -> None:
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_axis = results["time"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(results["target"][:, 0], results["target"][:, 1], label="target")
    ax.plot(results["pursuer"][:, 0], results["pursuer"][:, 1], label="pursuer")
    ax.set_title("World trajectory")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.grid(True)
    ax.legend(loc="best")

    ax = axes[0, 1]
    ax.plot(time_axis, results["input_error"], label="||uT-uThat||")
    ax.plot(time_axis, results["localization_error"], label="||r-rhat||")
    ax.plot(time_axis, results["tracking_error"], label="||r-rd||")
    ax.set_title("Errors")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("error")
    ax.grid(True)
    ax.legend(loc="best")

    ax = axes[1, 0]
    ax.plot(time_axis, results["target_speed"], label="||uT||")
    ax.plot(time_axis, results["pursuer_speed"], label="||uO||")
    ax.set_title("Command speed")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("m/s")
    ax.grid(True)
    ax.legend(loc="best")

    ax = axes[1, 1]
    ax.plot(time_axis, results["active_directions"], label="active RBF directions")
    ax.set_title("History-stack excitation")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("count")
    ax.grid(True)
    ax.legend(loc="best")

    figure.tight_layout()
    parent = os.path.dirname(os.path.abspath(output))
    os.makedirs(parent, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    print(f"[SIM] saved figure: {output}")
    if show:
        plt.show()
    plt.close(figure)


def add_boolean_argument(parser: argparse.ArgumentParser, name: str,
                         default: bool, help_text: str) -> None:
    """Add --flag and --no-flag options compatible with Python 3.8."""
    option_name = name[2:]
    destination = option_name.replace("-", "_")
    parser.add_argument(name, dest=destination, action="store_true",
                        default=default, help=help_text)
    parser.add_argument(f"--no-{option_name}", dest=destination,
                        action="store_false", help=argparse.SUPPRESS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pure Python bearing-only tracking simulation")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                        help="YAML parameter file shared with the ROS controller")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="simulation duration in seconds; MATLAB uses 200")
    parser.add_argument("--dt", type=float, default=0.01, help="simulation timestep")
    parser.add_argument("--relative-radius", type=float, default=0.7)
    parser.add_argument("--relative-angular-speed", type=float, default=0.3)
    parser.add_argument("--relative-initial-phase-deg", type=float, default=120.0)
    parser.add_argument("--tracking-gain", type=float, default=0.25)
    parser.add_argument("--rbf-count", type=int, default=30)
    parser.add_argument("--rbf-sigma", type=float, default=0.52)
    parser.add_argument("--window", type=float, default=1.0)
    parser.add_argument("--history-size", type=int, default=30)
    parser.add_argument("--max-history-execution-error", type=float, default=0.0)
    parser.add_argument("--max-history-target-execution-error", type=float, default=0.0)
    parser.add_argument("--history-forgetting-factor", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0e-7)
    parser.add_argument("--current-window-threshold", type=float, default=1.0e-5)
    parser.add_argument("--window-valid-tolerance", type=float,
                        default=100.0 * np.finfo(float).eps)
    parser.add_argument("--current-window-gain", type=float, default=2.0)
    parser.add_argument("--history-gain", type=float, default=1.2)
    parser.add_argument("--max-history-condition", type=float, default=1.0e3)
    parser.add_argument("--sigma-theta", type=float, default=0.003)
    parser.add_argument("--target-estimate-filter-bandwidth", type=float, default=5.0)
    parser.add_argument("--riccati-lambda", type=float, default=0.30)
    parser.add_argument("--observer-noise-weight", type=float, default=0.02)
    parser.add_argument("--riccati-eigenvalue-floor", type=float, default=1.0e-9)
    parser.add_argument("--mocap-position-observer-gain", type=float, default=0.0)
    parser.add_argument("--mocap-relative-velocity-observer-gain", type=float,
                        default=0.0)
    parser.add_argument("--target-vd", type=float, default=0.0)
    parser.add_argument("--target-ve", type=float, default=0.1)
    parser.add_argument("--target-vj", type=float, default=0.1)
    parser.add_argument("--max-pursuer-speed", type=float, default=0.0,
                        help="0 means no saturation, matching MATLAB")
    parser.add_argument("--max-target-speed", type=float, default=0.0,
                        help="0 means no saturation, matching MATLAB")
    add_boolean_argument(
        parser, "--use-true-relative-position", False,
        "use true relative position in Kr feedback (hybrid mocap equivalent)",
    )
    add_boolean_argument(
        parser, "--identifier-use-relative-velocity", False,
        "use u_o plus relative velocity as the identifier window target",
    )
    add_boolean_argument(parser, "--target-soft-boundary", False, "enable target soft boundary")
    parser.add_argument("--field-x-min", type=float, default=-1.5)
    parser.add_argument("--field-x-max", type=float, default=1.5)
    parser.add_argument("--field-y-min", type=float, default=-1.5)
    parser.add_argument("--field-y-max", type=float, default=1.5)
    parser.add_argument("--target-boundary-margin", type=float, default=0.30)
    parser.add_argument("--target-boundary-speed", type=float, default=0.15)
    parser.add_argument("--output", default="python_bearing_tracking_sim.png")
    parser.add_argument("--show", action="store_true", help="also open the Matplotlib figure")
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    config_args, _ = config_parser.parse_known_args()
    config_path = os.path.abspath(os.path.expanduser(config_args.config))
    try:
        with open(config_path, "r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
    except OSError as exc:
        parser.error(f"cannot read config file {config_path}: {exc}")
    except yaml.YAMLError as exc:
        parser.error(f"invalid YAML in {config_path}: {exc}")
    if not isinstance(config, dict):
        parser.error("YAML root must be a mapping of parameter names to values")

    defaults = {
        "dt": 1.0 / config["rate"],
        "relative_radius": config["relative_radius"],
        "relative_angular_speed": config["relative_angular_speed"],
        "relative_initial_phase_deg": config["relative_initial_phase_deg"],
        "tracking_gain": config["tracking_gain"],
        "rbf_count": config["rbf_count"],
        "rbf_sigma": config["rbf_sigma"],
        "window": config["window"],
        "history_size": config["history_size"],
        "max_history_execution_error": config.get("max_history_execution_error", 0.0),
        "max_history_target_execution_error": config.get(
            "max_history_target_execution_error", 0.0
        ),
        "history_forgetting_factor": config.get("history_forgetting_factor", 1.0),
        "identifier_use_relative_velocity": config.get(
            "identifier_use_relative_velocity", False
        ),
        "gamma": config["gamma"],
        "current_window_threshold": config["current_window_threshold"],
        "window_valid_tolerance": config["window_valid_tolerance"],
        "current_window_gain": config["current_window_gain"],
        "history_gain": config["history_gain"],
        "max_history_condition": config.get("max_history_condition", 1.0e3),
        "sigma_theta": config["sigma_theta"],
        "target_estimate_filter_bandwidth": config["target_estimate_filter_bandwidth"],
        "riccati_lambda": config["riccati_lambda"],
        "observer_noise_weight": config["observer_noise_weight"],
        "riccati_eigenvalue_floor": config["riccati_eigenvalue_floor"],
        "mocap_position_observer_gain": config.get(
            "mocap_position_observer_gain", 0.0
        ),
        "mocap_relative_velocity_observer_gain": config.get(
            "mocap_relative_velocity_observer_gain", 0.0
        ),
        "target_vd": config["target_vd"],
        "target_ve": config["target_ve"],
        "target_vj": config["target_vj"],
        "max_pursuer_speed": config["max_speed"],
        "max_target_speed": config["max_speed"],
        "use_true_relative_position": config["use_mocap_relative_position"],
        "target_soft_boundary": config["target_soft_boundary"],
        "field_x_min": config["field_x_min"],
        "field_x_max": config["field_x_max"],
        "field_y_min": config["field_y_min"],
        "field_y_max": config["field_y_max"],
        "target_boundary_margin": config["target_boundary_margin"],
        "target_boundary_speed": config["target_boundary_speed"],
    }
    parser.set_defaults(**defaults, config=config_path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dt <= 0.0 or args.duration <= 0.0:
        raise ValueError("dt and duration must be positive")
    p = Parameters(
        dt=args.dt,
        duration=args.duration,
        rd_radius=args.relative_radius,
        wd=args.relative_angular_speed,
        phi0=math.radians(args.relative_initial_phase_deg),
        kr=args.tracking_gain,
        q=args.rbf_count,
        sigma=args.rbf_sigma,
        delta_t=args.window,
        nh=args.history_size,
        max_history_execution_error=args.max_history_execution_error,
        max_history_target_execution_error=(
            args.max_history_target_execution_error
        ),
        history_forgetting_factor=args.history_forgetting_factor,
        identifier_use_relative_velocity=args.identifier_use_relative_velocity,
        gamma=args.gamma,
        ac=args.current_window_threshold,
        valid_tol=args.window_valid_tolerance,
        kc=args.current_window_gain,
        kh=args.history_gain,
        max_history_condition=args.max_history_condition,
        sigma_theta=args.sigma_theta,
        af=args.target_estimate_filter_bandwidth,
        lambda_s=args.riccati_lambda,
        ro=args.observer_noise_weight,
        p_floor=args.riccati_eigenvalue_floor,
        mocap_position_observer_gain=args.mocap_position_observer_gain,
        mocap_relative_velocity_observer_gain=(
            args.mocap_relative_velocity_observer_gain
        ),
        vd=args.target_vd,
        ve=args.target_ve,
        vj=args.target_vj,
        max_pursuer_speed=args.max_pursuer_speed,
        max_target_speed=args.max_target_speed,
        use_true_relative_feedback=args.use_true_relative_position,
        target_soft_boundary=args.target_soft_boundary,
        field_x_min=args.field_x_min,
        field_x_max=args.field_x_max,
        field_y_min=args.field_y_min,
        field_y_max=args.field_y_max,
        target_boundary_margin=args.target_boundary_margin,
        target_boundary_speed=args.target_boundary_speed,
    )
    _ = p.nw
    print(
        "[SIM] dt=%.3f s, T=%.1f s, Rd=%.2f m, wd=%.2f rad/s, "
        "ve=%.2f, vj=%.2f, pursuer_limit=%s"
        % (
            p.dt, p.duration, p.rd_radius, p.wd, p.ve, p.vj,
            "none" if p.max_pursuer_speed <= 0.0 else f"{p.max_pursuer_speed:.2f} m/s",
        )
    )
    results = BearingTrackingSimulation(p).run()
    save_plot(results, args.output, args.show)
    tail_mask = results["time"] >= max(0.0, results["time"][-1] - 30.0)
    print(
        "[SIM] windows=%d, active=%d, final errors: input=%.4f m/s, "
        "localization=%.4f m, tracking=%.4f m"
        % (
            results["history_count"], int(results["active_directions"][-1]),
            results["input_error"][-1], results["localization_error"][-1],
            results["tracking_error"][-1],
        )
    )
    print(
        "[SIM] last-30s mean/RMS: input=%.4f/%.4f m/s, "
        "localization=%.4f/%.4f m, tracking=%.4f/%.4f m"
        % (
            np.mean(results["input_error"][tail_mask]),
            np.sqrt(np.mean(results["input_error"][tail_mask] ** 2)),
            np.mean(results["localization_error"][tail_mask]),
            np.sqrt(np.mean(results["localization_error"][tail_mask] ** 2)),
            np.mean(results["tracking_error"][tail_mask]),
            np.sqrt(np.mean(results["tracking_error"][tail_mask] ** 2)),
        )
    )
    print(
        "[SIM] peak speeds: target=%.4f m/s, pursuer=%.4f m/s"
        % (np.max(results["target_speed"]), np.max(results["pursuer_speed"]))
    )


if __name__ == "__main__":
    main()
