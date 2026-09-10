"""Read the C++ paper fixtures; no separately implemented random generator."""

import json
import struct
import subprocess

import numpy as np


def cases(executable, suite):
    return json.loads(subprocess.check_output([str(executable), "--suite", suite]))


def problem(executable, suite, index, seed):
    data = subprocess.check_output(
        [str(executable), "--suite", suite, "--index", str(index), "--seed", str(seed)]
    )
    if data[:8] != b"CLQRB001":
        raise ValueError("invalid paper fixture format")
    offset = 8

    def word():
        nonlocal offset
        value = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        return value

    def array(rank):
        nonlocal offset
        shape = tuple(word() for _ in range(rank))
        count = int(np.prod(shape))
        result = np.frombuffer(data, dtype="<f8", count=count, offset=offset).reshape(shape)
        offset += 8 * count
        return result

    horizon = word()
    if word() != seed:
        raise ValueError("fixture seed mismatch")
    p = {"initial_state": array(1), "Q": [], "q": [], "stages": []}
    for _ in range(horizon + 1):
        p["Q"].append(array(2))
        p["q"].append(array(1))
    p["terminal_E"], p["terminal_e"] = array(2), array(1)
    fields = [("A", 2), ("B", 2), ("c", 1), ("R", 2), ("M", 2), ("r", 1),
              ("C", 2), ("D", 2), ("d", 1), ("E", 2), ("e", 1)]
    for _ in range(horizon):
        p["stages"].append({name: array(rank) for name, rank in fields})
    x = [array(1) for _ in range(horizon + 1)]
    u = [array(1) for _ in range(horizon)]
    if offset != len(data):
        raise ValueError("trailing paper fixture data")
    return p, x, u


def objective(p, x, u):
    value = np.longdouble(0)
    for i, xi in enumerate(x):
        xi = np.asarray(xi, dtype=np.longdouble)
        value += 0.5 * xi @ p["Q"][i].astype(np.longdouble) @ xi + p["q"][i] @ xi
        if i < len(u):
            ui = np.asarray(u[i], dtype=np.longdouble)
            s = p["stages"][i]
            value += 0.5 * ui @ s["R"].astype(np.longdouble) @ ui + s["r"] @ ui + xi @ s["M"] @ ui
    return value


def audit(p, solution, expected_x, expected_u):
    """Original unscaled KKT equations, independent of solver diagnostics."""
    def norm(v):
        return float(np.max(np.abs(v), initial=0))

    n = p["initial_state"].size
    x = [np.asarray(row[:n]) for row in solution.states]
    u = [np.asarray(row[:s["B"].shape[1]]) for row, s in zip(solution.controls, p["stages"])]
    if not all(np.all(np.isfinite(a)) for a in x + u):
        raise ValueError("non-finite trajectory")
    primal = max([norm(a - b) for a, b in zip(x, expected_x)] +
                 [norm(a - b) for a, b in zip(u, expected_u)])
    residuals = [norm(x[0] - p["initial_state"])]
    stationarity = []
    for i, xi in enumerate(x):
        gx = p["Q"][i] @ xi + p["q"][i]
        gx += solution.initial_multiplier[:n] if i == 0 else solution.dynamics_multipliers[i - 1, :n]
        if i < len(u):
            s, ui = p["stages"][i], u[i]
            lam = solution.dynamics_multipliers[i, :n]
            mu = solution.mixed_multipliers[i, :s["C"].shape[0]]
            nu = solution.state_multipliers[i, :s["E"].shape[0]]
            residuals += [norm(s["A"] @ xi + s["B"] @ ui + s["c"] - x[i + 1]),
                          norm(s["C"] @ xi + s["D"] @ ui + s["d"]),
                          norm(s["E"] @ xi + s["e"])]
            gx += s["M"] @ ui - s["A"].T @ lam + s["C"].T @ mu + s["E"].T @ nu
            stationarity.append(norm(s["R"] @ ui + s["r"] + s["M"].T @ xi - s["B"].T @ lam + s["D"].T @ mu))
        else:
            residuals.append(norm(p["terminal_E"] @ xi + p["terminal_e"]))
            gx += p["terminal_E"].T @ solution.terminal_state_multiplier[:p["terminal_E"].shape[0]]
        stationarity.append(norm(gx))
    # Do not let Python's max silently swallow NaNs in any individual equation.
    all_values = residuals + stationarity + [primal]
    if not np.all(np.isfinite(all_values)):
        raise ValueError("non-finite primal or KKT residual")
    actual = objective(p, x, u)
    reference = objective(p, expected_x, expected_u)
    obj_error = float(abs(actual - reference) / max(1, abs(reference)))
    if not np.isfinite(obj_error):
        raise ValueError("non-finite objective")
    return dict(primal_error=primal, relative_objective_error=obj_error,
                feasibility_inf=max(residuals), stationarity_inf=max(stationarity),
                kkt_inf=max(residuals + stationarity))
