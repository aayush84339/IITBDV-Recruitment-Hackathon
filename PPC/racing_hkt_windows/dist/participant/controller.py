'''
PPC Hackathon — Participant Boilerplate
You must implement two functions: plan() and control()
'''

# ─── TYPES (for reference) ────────────────────────────────────────────────────

# Path: list of waypoints [{"x": float, "y": float}, ...]
# State: {"x", "y", "yaw", "vx", "vy", "yaw_rate"} 
# CmdFeedback: {"throttle", "steer"}         

# ─── CONTROLLER ───────────────────────────────────────────────────────────────
import numpy as np

# Persistent state across control() calls
_integral = 0.0
_prev_err = 0.0
_nearest = 0


def steering(path: list[dict], state: dict):

    length_of_car = 2.6
    x, y, yaw = state["x"], state["y"], state["yaw"]
    speed = np.sqrt(state["vx"]**2 + state["vy"]**2)

    # Adaptive lookahead: faster → look further ahead (bigger range for stability)
    lookahead = np.clip(0.6 * speed, 3.0, 10.0)

    # Find nearest waypoint — FORWARD ONLY (no backward search = no oscillation)
    global _nearest
    best, best_d = _nearest, float('inf')
    for i in range(_nearest, min(len(path), _nearest + 50)):
        d = (path[i]["x"] - x)**2 + (path[i]["y"] - y)**2
        if d < best_d:
            best_d = d
            best = i
    _nearest = best

    # Walk forward to find lookahead target
    target = min(len(path) - 1, best)
    for i in range(best, min(best + 80, len(path))):
        if np.hypot(path[i]["x"] - x, path[i]["y"] - y) >= lookahead:
            target = i
            break

    # Transform target to vehicle frame
    dx = path[target]["x"] - x
    dy = path[target]["y"] - y
    local_x =  dx * np.cos(yaw) + dy * np.sin(yaw)
    local_y = -dx * np.sin(yaw) + dy * np.cos(yaw)

    # Pure Pursuit formula: δ = arctan(2·L·sin(α) / ld)
    ld_sq = local_x**2 + local_y**2
    if ld_sq < 1e-2:
        steer = 0.0
    else:
        steer = np.arctan2(2.0 * length_of_car * local_y, ld_sq)

    # 0.5 is the max steering angle in radians (about 28.6 degrees)
    return np.clip(steer, -0.5, 0.5)


def throttle_algorithm(target_speed, current_speed, dt):
    global _integral, _prev_err

    error = target_speed - current_speed
    _integral = np.clip(_integral + error * dt, -12.0, 12.0)  # anti-windup
    deriv = (error - _prev_err) / dt if dt > 0 else 0.0
    _prev_err = error

    # Aggressive PID (Kp=3.0, Ki=0.06, Kd=0.15)
    output = 3.0 * error + 0.06 * _integral + 0.15 * deriv

    if output >= 0:
        throttle = 2*output
        brake = 0.0
    else:
        throttle = 0
        brake = -output * 0.7  # softer braking to keep momentum

    # Hard brake if way over target
    if current_speed > target_speed + 5.0:
        throttle = 0
        brake = np.clip((current_speed - target_speed) / 5.0, 0.3, 0.9)

    # clip throttle and brake to [0, 1]
    return np.clip(throttle, 0.0, 1.0), np.clip(brake, 0.0, 1.0)


def _target_speed(path, idx):
    """Curvature-adaptive target speed: slow in turns, fast on straights."""
    max_k = 0.0
    n = len(path)
    for i in range(max(1, idx), min(idx + 18, n - 1)):
        p1 = np.array([path[i-1]["x"], path[i-1]["y"]])
        p2 = np.array([path[i]["x"],   path[i]["y"]])
        p3 = np.array([path[i+1]["x"], path[i+1]["y"]])
        a, b, c = np.linalg.norm(p2-p1), np.linalg.norm(p3-p2), np.linalg.norm(p3-p1)
        area = abs(np.cross(p2-p1, p3-p1)) / 2.0
        denom = a * b * c
        if denom > 1e-9:
            max_k = max(max_k, 4.0 * area / denom)
    # Map curvature → speed: tight turn = slow, straight = fast
    return np.clip(80.0 / (1.0 +  max_k*20), 10.0, 80.0)


def control(
    path: list[dict],
    state: dict,
    cmd_feedback: dict,
    step: int,
) -> tuple[float, float, float]:
    """
    Generate throttle, steer, brake for the current timestep.
    Called every 50ms during simulation.
    """
    global _integral, _prev_err, _nearest

    # Reset on first step
    if step == 0:
        _integral, _prev_err, _nearest = 0.0, 0.0, 0

    if not path or len(path) < 2:
        return 0.0, 0.0, 1.0


    # Steering via Pure Pursuit
    steer = steering(path, state)

    # Curvature-adaptive target speed
    speed = np.sqrt(state["vx"]**2 + state["vy"]**2)
    t_speed = _target_speed(path, _nearest)


    throttle, brake = throttle_algorithm(t_speed, speed, 0.05)

    # Mutual exclusivity: throttle and brake can't both be > 0
    if throttle > 0 and brake > 0:
        throttle, brake = (0.0, brake) if brake > throttle else (throttle, 0.0)

    return float(throttle), float(steer), float(brake)
