'''
PPC Hackathon — Participant Boilerplate
You must implement two functions: plan() and control()
'''

# ─── TYPES (for reference) ────────────────────────────────────────────────────

# Cone: {"x": float, "y": float, "side": "left" | "right", "index": int}
# State: {"x", "y", "yaw", "vx", "vy", "yaw_rate"}  
# CmdFeedback: {"throttle", "steer"}        

# ─── PLANNER ──────────────────────────────────────────────────────────────────
import numpy as np

def plan(cones: list[dict]) -> list[dict]:
    """
    Generate a path from the cone layout.
    Called ONCE before the simulation starts.

    Args:
        cones: List of cone dicts with keys x, y, side ("left"/"right"), index

    Returns:
        path: List of waypoints [{"x": float, "y": float}, ...]
              Ordered from start to finish.
    """
    path = []
    blue = np.array([[cone["x"], cone["y"]] for cone in cones if cone["side"] == "left"])
    yellow = np.array([[cone["x"], cone["y"]] for cone in cones if cone["side"] == "right"])

    if len(blue) == 0 or len(yellow) == 0:
        return path

    # Sort each side by index so they follow track order
    left_cones = sorted([c for c in cones if c["side"] == "left"], key=lambda c: c["index"])
    right_cones = sorted([c for c in cones if c["side"] == "right"], key=lambda c: c["index"])

    # Equal cones — direct 1:1 matching by index
    n = len(left_cones)  # equal count guaranteed
    left_pts = np.array([[c["x"], c["y"]] for c in left_cones])
    right_pts = np.array([[c["x"], c["y"]] for c in right_cones])

    # Midpoints between matched pairs
    midpoints = (left_pts + right_pts) / 2.0

    # Racing line: shift toward inside of turns (apex cutting)
    racing = midpoints.copy()
    for i in range(1, n - 1):
        v1 = midpoints[i] - midpoints[i - 1]
        v2 = midpoints[i + 1] - midpoints[i]
        cross = np.cross(v1, v2)

        if abs(cross) > 0.01:
            if cross > 0:  # left turn → inner side is left
                inner = left_pts[i]
            else:  # right turn → inner side is right
                inner = right_pts[i]
            direction = inner - midpoints[i]
            dlen = np.linalg.norm(direction)
            if dlen > 0.1:
                racing[i] = midpoints[i] + 0.2 * direction  # 20% toward apex

    # Chaikin smoothing (4 iterations for very smooth path)
    pts = racing.copy()
    for _ in range(3):
        if len(pts) < 2:
            break
        new = []
        for i in range(len(pts) - 1):
            new.append(0.75 * pts[i] + 0.25 * pts[i + 1])
            new.append(0.25 * pts[i] + 0.75 * pts[i + 1])
        pts = np.array(new)

    for pt in pts:
        path.append({"x": float(pt[0]), "y": float(pt[1])})

    return path
