from flask import Flask, render_template, Response, jsonify, request
from argparse import ArgumentParser
from simulator import Simulator
from utils import load_config
import threading
import time
import json
import numpy as np
import os
import glob
import re

app = Flask(
    __name__,
    template_folder="visualizer/templates",
    static_folder="visualizer/static",
)

TARGET_FPS = 60.0
PANEL_KEYS = ("left", "right")
PANEL_TO_INDEX = {"left": 0, "right": 1}

state_lock = threading.Lock()
sim_lock = threading.Lock()

app_state = {
    "step_index": 0,
    "actual_fps": 0.0,
    "topology_dirty": False,
    "kill_stream": False,
    "left_file": "",
    "right_file": "",
    "paused": False,
    "step_once": False,
    "stream_session": 0,
    "last_error": "",
}

simulator = None
robots_by_panel = {key: None for key in PANEL_KEYS}
n_masses_cached = {key: 0 for key in PANEL_KEYS}
n_springs_cached = {key: 0 for key in PANEL_KEYS}
prev_com = {key: None for key in PANEL_KEYS}
total_dist = {key: 0.0 for key in PANEL_KEYS}
max_steps = 0

robot_files = []
generation_files = []
generation_options = []
fitness_files = []
results_dir = ""
args = None


def reset_motion_metrics():
    for key in PANEL_KEYS:
        prev_com[key] = None
        total_dist[key] = 0.0


def generation_label(path: str) -> str:
    name = os.path.basename(path)
    match = re.match(r"best_robot_gen_(\d+)\.npy$", name)
    if match:
        return f"Generation {int(match.group(1)):03d}"
    return name


def make_generation_option(path: str) -> dict:
    name = os.path.basename(path)
    match = re.match(r"best_robot_gen_(\d+)\.npy$", name)
    return {
        "file": path,
        "label": generation_label(path),
        "generation": int(match.group(1)) if match else None,
    }


def _glob_from_results(pattern: str, recursive: bool = False):
    if results_dir:
        return glob.glob(os.path.join(results_dir, pattern), recursive=recursive)
    return glob.glob(pattern, recursive=recursive)

def discover_saved_files(default_input: str):
    files = sorted(_glob_from_results("robot_*.npy") + _glob_from_results("best_robot_gen_*.npy"))

    if default_input and os.path.exists(default_input) and default_input not in files:
        files.append(default_input)

    files = sorted(set(files))

    generations = sorted(_glob_from_results("best_robot_gen_*.npy"))
    if not generations:
        generations = files.copy()

    return files, generations


def discover_fitness_files():
    if results_dir:
        files = _glob_from_results("*fitness*.npy")
    else:
        files = _glob_from_results("*fitness*.npy") + _glob_from_results("**/*fitness*.npy", recursive=True)
    files = [path for path in files if os.path.isfile(path)]
    return sorted(set(files))


def resolve_saved_file(file_name: str):
    if not file_name:
        return None

    if os.path.exists(file_name):
        return file_name

    for candidate in robot_files:
        if candidate == file_name or os.path.basename(candidate) == os.path.basename(file_name):
            return candidate

    return file_name


def resolve_fitness_file(file_name: str):
    if not file_name:
        return None

    if os.path.exists(file_name):
        return file_name

    for candidate in fitness_files:
        if candidate == file_name or os.path.basename(candidate) == os.path.basename(file_name):
            return candidate

    return file_name


def load_robot_dict(path: str) -> dict:
    robot = np.load(path, allow_pickle=True).item()
    if not isinstance(robot, dict):
        raise ValueError(f"Saved file {path} is not a robot dictionary.")
    return robot


def infer_cpg_count(w1_rows: int, default_cpg: int):
    candidates = []
    for cpg in [default_cpg, 6, 8, 10, 12, 4, 16]:
        if cpg not in candidates:
            candidates.append(cpg)

    for cpg in candidates:
        if (w1_rows - cpg) > 0 and (w1_rows - cpg) % 4 == 0:
            return cpg, (w1_rows - cpg) // 4

    raise ValueError(
        "Cannot infer nn_cpg_count / n_masses from controller dimensions. "
        f"weights1 rows={w1_rows}."
    )


def build_robot_spec(robot: dict, path: str, config: dict) -> dict:
    masses = robot.get("masses", [])
    springs = robot.get("springs", [])

    actual_masses = len(masses)
    actual_springs = len(springs)

    if actual_masses == 0 or actual_springs == 0:
        raise ValueError(f"Saved file {path} is missing valid masses/springs.")

    if "max_n_masses" in robot and "max_n_springs" in robot:
        nm = int(robot["max_n_masses"])
        ns = int(robot["max_n_springs"])
    elif "n_masses" in robot and "n_springs" in robot:
        nm = int(robot["n_masses"])
        ns = int(robot["n_springs"])
    else:
        nm = int(actual_masses)
        ns = int(actual_springs)

    control_params = robot.get("control_params")
    cp_hidden = None
    cp_cpg = None

    if control_params is not None:
        for key in ("weights1", "weights2", "biases1", "biases2"):
            if key not in control_params:
                raise ValueError(f"control_params missing '{key}' in {path}.")

        w1 = np.asarray(control_params["weights1"])
        w2 = np.asarray(control_params["weights2"])
        b1 = np.asarray(control_params["biases1"])
        b2 = np.asarray(control_params["biases2"])

        if w1.ndim != 2 or w2.ndim != 2 or b1.ndim != 1 or b2.ndim != 1:
            raise ValueError(f"Invalid controller array shapes in {path}.")

        w1_rows, hidden = w1.shape
        w2_hidden, w2_springs = w2.shape

        if w2_hidden != hidden or b1.shape[0] != hidden:
            raise ValueError(
                f"Inconsistent hidden dimensions in {path}: "
                f"weights1={w1.shape}, weights2={w2.shape}, biases1={b1.shape}."
            )

        if b2.shape[0] != w2_springs:
            raise ValueError(
                f"Inconsistent spring dimensions in {path}: "
                f"weights2={w2.shape}, biases2={b2.shape}."
            )

        default_cpg = int(config["simulator"].get("nn_cpg_count", 6))
        cp_cpg, nm_from_cp = infer_cpg_count(w1_rows, default_cpg)

        nm = max(nm, int(nm_from_cp))
        ns = max(ns, int(w2_springs))
        cp_hidden = int(hidden)

    nm = max(nm, actual_masses)
    ns = max(ns, actual_springs)

    return {
        "robot": robot,
        "nm": int(nm),
        "ns": int(ns),
        "cp_hidden": cp_hidden,
        "cp_cpg": cp_cpg,
        "control_params": control_params,
    }


def adapt_control_params(control_params: dict, target_mass_rows: int, target_cpg_count: int, target_hidden: int, target_springs: int) -> dict:
    """Pad/crop controller arrays so mixed-generation robots can share one simulator shape."""
    w1_old = np.asarray(control_params["weights1"], dtype=np.float32)
    w2_old = np.asarray(control_params["weights2"], dtype=np.float32)
    b1_old = np.asarray(control_params["biases1"], dtype=np.float32)
    b2_old = np.asarray(control_params["biases2"], dtype=np.float32)

    if w1_old.ndim != 2 or w2_old.ndim != 2 or b1_old.ndim != 1 or b2_old.ndim != 1:
        raise ValueError("control_params arrays must be rank-2/2/1/1.")

    old_total_rows, old_hidden = w1_old.shape
    old_mass_rows = old_total_rows - target_cpg_count
    if old_mass_rows < 0:
        raise ValueError(
            "control_params weights1 rows are smaller than nn_cpg_count; "
            "this file is incompatible with the selected simulator settings."
        )

    mass_rows = max(0, int(target_mass_rows))
    cpg_count = max(0, int(target_cpg_count))
    total_rows = mass_rows + cpg_count
    hidden = max(0, int(target_hidden))
    springs = max(0, int(target_springs))

    w1_new = np.zeros((total_rows, hidden), dtype=np.float32)
    w2_new = np.zeros((hidden, springs), dtype=np.float32)
    b1_new = np.zeros((hidden,), dtype=np.float32)
    b2_new = np.zeros((springs,), dtype=np.float32)

    copy_mass_rows = min(old_mass_rows, mass_rows)
    copy_hidden = min(old_hidden, hidden)
    if copy_mass_rows > 0 and copy_hidden > 0:
        w1_new[:copy_mass_rows, :copy_hidden] = w1_old[:copy_mass_rows, :copy_hidden]

    copy_cpg_rows = min(cpg_count, old_total_rows - old_mass_rows)
    if copy_cpg_rows > 0 and copy_hidden > 0:
        old_cpg_start = old_mass_rows
        new_cpg_start = mass_rows
        w1_new[new_cpg_start:new_cpg_start + copy_cpg_rows, :copy_hidden] = (
            w1_old[old_cpg_start:old_cpg_start + copy_cpg_rows, :copy_hidden]
        )

    copy_w2_hidden = min(w2_old.shape[0], hidden)
    copy_w2_springs = min(w2_old.shape[1], springs)
    if copy_w2_hidden > 0 and copy_w2_springs > 0:
        w2_new[:copy_w2_hidden, :copy_w2_springs] = w2_old[:copy_w2_hidden, :copy_w2_springs]

    copy_b1 = min(b1_old.shape[0], hidden)
    if copy_b1 > 0:
        b1_new[:copy_b1] = b1_old[:copy_b1]

    copy_b2 = min(b2_old.shape[0], springs)
    if copy_b2 > 0:
        b2_new[:copy_b2] = b2_old[:copy_b2]

    return {
        "weights1": w1_new,
        "weights2": w2_new,
        "biases1": b1_new,
        "biases2": b2_new,
    }


def friendly_load_error(exc: Exception) -> str:
    message = str(exc).strip()
    if not message:
        return "Unable to load selected generation files."
    return message


def friendly_step_error(exc: Exception) -> str:
    message = str(exc).strip()
    if "Field with dim" in message or "accessed with indices of dim" in message:
        return (
            "A temporary simulator state mismatch was detected while loading robots. "
            "The stream was paused safely. Please re-select the generation pair."
        )
    if "Simulator is not initialized" in message:
        return "Simulator is not initialized yet. Please choose generation files again."
    return f"Simulation paused due to an internal issue: {message}"


def friendly_fitness_error(exc: Exception) -> str:
    message = str(exc).strip()
    if not message:
        return "Unable to load selected fitness history file."
    return message


def build_sim_bundle(left_file: str, right_file: str) -> dict:
    config = load_config(args.config)

    left_robot = load_robot_dict(left_file)
    right_robot = load_robot_dict(right_file)

    left_spec = build_robot_spec(left_robot, left_file, config)
    right_spec = build_robot_spec(right_robot, right_file, config)

    hidden_values = [v for v in [left_spec["cp_hidden"], right_spec["cp_hidden"]] if v is not None]
    cpg_values = [v for v in [left_spec["cp_cpg"], right_spec["cp_cpg"]] if v is not None]

    default_hidden = int(config["simulator"].get("nn_hidden_size", 128))
    default_cpg = int(config["simulator"].get("nn_cpg_count", 6))

    target_hidden = max([default_hidden] + hidden_values) if hidden_values else default_hidden
    target_cpg = max([default_cpg] + cpg_values) if cpg_values else default_cpg

    config["simulator"]["nn_hidden_size"] = int(target_hidden)
    config["simulator"]["nn_cpg_count"] = int(target_cpg)
    config["simulator"]["n_masses"] = int(max(left_spec["nm"], right_spec["nm"]))
    config["simulator"]["n_springs"] = int(max(left_spec["ns"], right_spec["ns"]))
    config["simulator"]["n_sims"] = 2

    with sim_lock:
        sim = Simulator(
            sim_config=config["simulator"],
            taichi_config=config["taichi"],
            seed=config["seed"],
            needs_grad=False,
        )

        sim.initialize(
            [left_robot["masses"], right_robot["masses"]],
            [left_robot["springs"], right_robot["springs"]],
        )

        cp_indices = []
        cp_values = []

        target_mass_rows = int(config["simulator"]["n_masses"]) * 4
        target_cpg_count = int(config["simulator"]["nn_cpg_count"])
        target_hidden = int(config["simulator"]["nn_hidden_size"])
        target_springs = int(config["simulator"]["n_springs"])

        if left_spec["control_params"] is not None:
            cp_indices.append(0)
            cp_values.append(
                adapt_control_params(
                    left_spec["control_params"],
                    target_mass_rows=target_mass_rows,
                    target_cpg_count=target_cpg_count,
                    target_hidden=target_hidden,
                    target_springs=target_springs,
                )
            )
        if right_spec["control_params"] is not None:
            cp_indices.append(1)
            cp_values.append(
                adapt_control_params(
                    right_spec["control_params"],
                    target_mass_rows=target_mass_rows,
                    target_cpg_count=target_cpg_count,
                    target_hidden=target_hidden,
                    target_springs=target_springs,
                )
            )

        if cp_indices:
            sim.set_control_params(cp_indices, cp_values)

        counts_m = {
            "left": int(sim.n_masses[0]),
            "right": int(sim.n_masses[1]),
        }
        counts_s = {
            "left": int(sim.n_springs[0]),
            "right": int(sim.n_springs[1]),
        }
        max_steps_local = int(sim.steps[None])

    return {
        "simulator": sim,
        "robots": {
            "left": left_robot,
            "right": right_robot,
        },
        "n_masses": counts_m,
        "n_springs": counts_s,
        "max_steps": max_steps_local,
        "left_file": left_file,
        "right_file": right_file,
    }


def apply_sim_bundle(bundle: dict):
    global simulator, robots_by_panel, n_masses_cached, n_springs_cached, max_steps

    with sim_lock:
        simulator = bundle["simulator"]
        robots_by_panel = bundle["robots"]
        n_masses_cached = bundle["n_masses"]
        n_springs_cached = bundle["n_springs"]
        max_steps = bundle["max_steps"]

    reset_motion_metrics()

    with state_lock:
        app_state["step_index"] = 0
        app_state["actual_fps"] = 0.0
        app_state["left_file"] = bundle["left_file"]
        app_state["right_file"] = bundle["right_file"]
        app_state["topology_dirty"] = True
        app_state["kill_stream"] = False
        app_state["step_once"] = False
        app_state["last_error"] = ""


def build_topology_payload() -> dict:
    with sim_lock:
        panel_payload = {}
        for key in PANEL_KEYS:
            robot = robots_by_panel.get(key)
            panel_payload[key] = {
                "springs": robot["springs"].tolist() if robot is not None else [],
                "n_masses": int(n_masses_cached.get(key, 0)),
                "n_springs": int(n_springs_cached.get(key, 0)),
                "robot_file": app_state.get(f"{key}_file", ""),
                "label": generation_label(app_state.get(f"{key}_file", "")),
            }

    with state_lock:
        current = {
            "left": app_state.get("left_file", ""),
            "right": app_state.get("right_file", ""),
        }
        paused = bool(app_state.get("paused", False))
        step_index = int(app_state.get("step_index", 0))

    return {
        "type": "topology",
        "panels": panel_payload,
        "current": current,
        "paused": paused,
        "step": step_index,
    }


def validate_runtime_state(sim: Simulator):
    if sim is None:
        raise ValueError("Simulator is not initialized.")

    n_sims = int(sim.n_sims[None])
    if n_sims < 2:
        raise ValueError("Simulator panel count mismatch. Please reload generations.")

    max_masses = int(sim.max_n_masses[None])
    max_springs = int(sim.max_n_springs[None])
    if max_masses <= 0 or max_springs <= 0:
        raise ValueError("Simulator dimensions are invalid. Please reload generations.")

    counts = {}
    for key in PANEL_KEYS:
        idx = PANEL_TO_INDEX[key]
        masses = int(sim.n_masses[idx])
        springs = int(sim.n_springs[idx])

        if masses <= 0 or springs <= 0:
            raise ValueError(f"{key.title()} robot has invalid masses/springs. Please choose another generation.")

        if masses > max_masses or springs > max_springs:
            raise ValueError(
                f"{key.title()} robot exceeds simulator capacity. Please choose a different generation pair."
            )

        counts[key] = {
            "masses": masses,
            "springs": springs,
        }

    steps = int(sim.steps[None])
    if steps <= 0:
        raise ValueError("Simulator has invalid step count.")

    return counts, steps


def step_once():
    with state_lock:
        t = int(app_state["step_index"])

    with sim_lock:
        sim = simulator
        counts, max_steps_local = validate_runtime_state(sim)

        if t >= max_steps_local:
            sim.reinitialize_robots()
            reset_motion_metrics()
            with state_lock:
                app_state["step_index"] = 0
                t = 0

        sim.compute_com(t)
        sim.nn1(t)
        sim.nn2(t)
        sim.apply_spring_force(t)
        sim.advance(t + 1)

        x_np = sim.x.to_numpy()
        act_np = sim.act.to_numpy()

        panel_frames = {}
        for key in PANEL_KEYS:
            idx = PANEL_TO_INDEX[key]
            m_count = counts[key]["masses"]
            s_count = counts[key]["springs"]

            positions = x_np[idx, t + 1, :m_count]
            activations = act_np[idx, t, :s_count]
            center_of_mass = positions.mean(axis=0) if len(positions) > 0 else np.array([0.0, 0.0], dtype=np.float32)

            if prev_com[key] is None:
                speed = 0.0
            else:
                dx = float(center_of_mass[0] - prev_com[key][0])
                dy = float(center_of_mass[1] - prev_com[key][1])
                step_dist = (dx * dx + dy * dy) ** 0.5
                total_dist[key] += step_dist
                speed = step_dist * TARGET_FPS

            prev_com[key] = center_of_mass.copy()
            n_masses_cached[key] = m_count
            n_springs_cached[key] = s_count

            panel_frames[key] = {
                "positions": positions,
                "activations": activations,
                "center_of_mass": center_of_mass,
                "speed": speed,
                "dist": total_dist[key],
            }

    with state_lock:
        app_state["step_index"] = t + 1
        step_index = int(app_state["step_index"])

    return panel_frames, step_index


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stream")
def stream():
    def event_stream():
        with state_lock:
            app_state["stream_session"] = int(app_state.get("stream_session", 0)) + 1
            my_session = app_state["stream_session"]

        yield f"data: {json.dumps(build_topology_payload())}\n\n"

        fps_samples = []
        last_fps_update = time.perf_counter()
        last_playback_emit = 0.0

        while True:
            frame_start = time.perf_counter()
            target_interval = 1.0 / TARGET_FPS

            with state_lock:
                current_session = int(app_state.get("stream_session", 0))
                if my_session != current_session:
                    break

                if app_state.get("kill_stream", False):
                    app_state["kill_stream"] = False
                    break

                dirty = bool(app_state.get("topology_dirty", False))
                if dirty:
                    app_state["topology_dirty"] = False

                paused = bool(app_state.get("paused", False))
                do_single_step = bool(app_state.get("step_once", False))
                if do_single_step:
                    app_state["step_once"] = False

                current_step = int(app_state.get("step_index", 0))

            if dirty:
                yield f"data: {json.dumps(build_topology_payload())}\n\n"

            if paused and not do_single_step:
                now = time.perf_counter()
                if now - last_playback_emit >= 0.25:
                    playback_payload = {
                        "type": "playback",
                        "paused": True,
                        "step": current_step,
                    }
                    yield f"data: {json.dumps(playback_payload)}\n\n"
                    last_playback_emit = now
                time.sleep(min(target_interval, 0.05))
                continue

            try:
                panel_frames, step_index = step_once()
            except Exception as exc:
                friendly = friendly_step_error(exc)
                with state_lock:
                    app_state["paused"] = True
                    app_state["step_once"] = False
                    app_state["last_error"] = friendly
                error_payload = {
                    "type": "error",
                    "message": friendly,
                }
                yield f"data: {json.dumps(error_payload)}\n\n"
                continue

            with state_lock:
                paused_now = bool(app_state.get("paused", False))

            payload = {
                "type": "step",
                "step": step_index,
                "fps": app_state["actual_fps"],
                "paused": paused_now,
                "panels": {},
            }

            for key in PANEL_KEYS:
                frame = panel_frames[key]
                payload["panels"][key] = {
                    "positions": frame["positions"].tolist(),
                    "activations": frame["activations"].tolist(),
                    "center_of_mass": frame["center_of_mass"].tolist(),
                    "speed": frame["speed"],
                    "dist": frame["dist"],
                }

            yield f"data: {json.dumps(payload)}\n\n"

            work_time = time.perf_counter() - frame_start
            sleep_time = target_interval - work_time
            if sleep_time > 0.001:
                time.sleep(sleep_time)

            total_frame_time = time.perf_counter() - frame_start
            if total_frame_time > 0:
                fps_samples.append(1.0 / total_frame_time)

            now = time.perf_counter()
            if now - last_fps_update >= 0.5 and fps_samples:
                with state_lock:
                    app_state["actual_fps"] = sum(fps_samples) / len(fps_samples)
                fps_samples = []
                last_fps_update = now

    response = Response(event_stream(), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


@app.route("/robots")
def robots():
    return jsonify(
        {
            "robots": robot_files,
            "current": {
                "left": app_state.get("left_file", ""),
                "right": app_state.get("right_file", ""),
            },
        }
    )


@app.route("/generations")
def generations():
    with state_lock:
        paused = bool(app_state.get("paused", False))
    return jsonify(
        {
            "generations": generation_options,
            "current": {
                "left": app_state.get("left_file", ""),
                "right": app_state.get("right_file", ""),
            },
            "paused": paused,
        }
    )


@app.route("/fitness_files")
def list_fitness_files():
    global fitness_files
    fitness_files = discover_fitness_files()
    return jsonify(
        {
            "files": [
                {
                    "file": path,
                    "label": os.path.basename(path),
                }
                for path in fitness_files
            ]
        }
    )


@app.route("/fitness_history", methods=["POST"])
def fitness_history():
    data = request.get_json(force=True, silent=True) or {}
    file_name = data.get("file", "")
    resolved = resolve_fitness_file(file_name)

    try:
        if not resolved:
            raise ValueError("Please choose a fitness history file.")
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"Could not find fitness history file: {resolved}")

        fitness = np.load(resolved, allow_pickle=False)
        fitness = np.asarray(fitness, dtype=np.float32)

        if fitness.ndim == 1:
            fitness = fitness[np.newaxis, :]
        if fitness.ndim != 2:
            raise ValueError("Fitness history file must be a 1D or 2D NumPy array.")
        if fitness.shape[1] == 0:
            raise ValueError("Fitness history file has no training steps.")
        if not np.isfinite(fitness).all():
            raise ValueError("Fitness history contains non-finite values.")

        mean_trajectory = fitness.mean(axis=0)

        return jsonify(
            {
                "ok": True,
                "file": resolved,
                "n_robots": int(fitness.shape[0]),
                "n_steps": int(fitness.shape[1]),
                "fitness_history": fitness.tolist(),
                "mean": mean_trajectory.tolist(),
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": friendly_fitness_error(exc)}), 400

@app.route("/playback", methods=["POST"])
def playback():
    data = request.get_json(force=True, silent=True) or {}
    action = str(data.get("action", "status")).strip().lower()

    with state_lock:
        if action == "pause":
            app_state["paused"] = True
            app_state["step_once"] = False
        elif action == "play":
            app_state["paused"] = False
            app_state["step_once"] = False
        elif action == "toggle":
            app_state["paused"] = not bool(app_state.get("paused", False))
            if not app_state["paused"]:
                app_state["step_once"] = False
        elif action == "step":
            app_state["paused"] = True
            app_state["step_once"] = True
        elif action == "status":
            pass
        else:
            return jsonify({"ok": False, "error": "Unknown playback action."}), 400

        state = {
            "ok": True,
            "paused": bool(app_state.get("paused", False)),
            "step": int(app_state.get("step_index", 0)),
        }

    return jsonify(state)


def validate_selection(left_file: str, right_file: str):
    if not left_file or not right_file:
        raise ValueError("Please choose both a left and right generation file.")

    if not os.path.exists(left_file):
        raise FileNotFoundError(f"Could not find left generation file: {left_file}")

    if not os.path.exists(right_file):
        raise FileNotFoundError(f"Could not find right generation file: {right_file}")


@app.route("/set_generations", methods=["POST"])
def set_generations():
    data = request.get_json(force=True, silent=True) or {}

    left_file = resolve_saved_file(data.get("left", ""))
    right_file = resolve_saved_file(data.get("right", ""))

    try:
        validate_selection(left_file, right_file)
    except Exception as exc:
        return jsonify({"ok": False, "error": friendly_load_error(exc)}), 400

    with state_lock:
        app_state["kill_stream"] = True

    time.sleep(0.06)

    try:
        bundle = build_sim_bundle(left_file, right_file)
    except Exception as exc:
        with state_lock:
            app_state["kill_stream"] = False
            app_state["topology_dirty"] = True
        return jsonify({"ok": False, "error": friendly_load_error(exc)}), 400

    apply_sim_bundle(bundle)

    with state_lock:
        paused = bool(app_state.get("paused", False))

    return jsonify(
        {
            "ok": True,
            "current": {
                "left": left_file,
                "right": right_file,
            },
            "paused": paused,
        }
    )


@app.route("/set_robot", methods=["POST"])
def set_robot():
    data = request.get_json(force=True, silent=True) or {}
    robot_file = resolve_saved_file(data.get("robot", ""))

    if not robot_file or not os.path.exists(robot_file):
        return jsonify({"ok": False, "error": "Robot file not found."}), 400

    with state_lock:
        app_state["kill_stream"] = True

    time.sleep(0.06)

    try:
        bundle = build_sim_bundle(robot_file, robot_file)
    except Exception as exc:
        with state_lock:
            app_state["kill_stream"] = False
            app_state["topology_dirty"] = True
        return jsonify({"ok": False, "error": friendly_load_error(exc)}), 400

    apply_sim_bundle(bundle)

    return jsonify(
        {
            "ok": True,
            "current": {
                "left": robot_file,
                "right": robot_file,
            },
        }
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, default="robot_0.npy", help="Path to saved robot .npy file or basename inside --results")
    parser.add_argument("--results", type=str, default="", help="Directory containing saved results (.npy files)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.results:
        results_dir = os.path.abspath(args.results)
        if not os.path.isdir(results_dir):
            raise RuntimeError(f"Results directory not found: {results_dir}")

    robot_files, generation_files = discover_saved_files(args.input)
    generation_options = [make_generation_option(path) for path in generation_files]
    fitness_files = discover_fitness_files()

    if not robot_files:
        raise RuntimeError("No saved robot files were found. Run training first to generate .npy files.")

    default_left = resolve_saved_file(args.input)
    if default_left is None or not os.path.exists(default_left):
        default_left = generation_files[0]

    default_right = generation_files[-1] if len(generation_files) > 1 else default_left

    validate_selection(default_left, default_right)

    print(f"Loading initial generations: left={default_left}, right={default_right}")
    initial_bundle = build_sim_bundle(default_left, default_right)
    apply_sim_bundle(initial_bundle)

    print(f"\nVisualizer running at http://localhost:{args.port}")
    print("Press Ctrl+C to stop\n")

    app.run(host="0.0.0.0", port=args.port, debug=args.debug, threaded=True, use_reloader=False)
