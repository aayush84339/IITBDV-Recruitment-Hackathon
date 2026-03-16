import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import distance
import pandas as pd

# ── Load Track from CSV ───────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(_HERE, "small_track.csv"))

BLUE_CONES   = df[df["tag"] == "blue"      ][["x", "y"]].values.astype(float)
YELLOW_CONES = df[df["tag"] == "yellow"    ][["x", "y"]].values.astype(float)
BIG_ORANGE   = df[df["tag"] == "big_orange"][["x", "y"]].values.astype(float)

_cs               = df[df["tag"] == "car_start"].iloc[0]
CAR_START_POS     = np.array([float(_cs["x"]), float(_cs["y"])])
CAR_START_HEADING = float(_cs["direction"])   # radians (0 = east)

MAP_CONES = np.vstack([BLUE_CONES, YELLOW_CONES])


# ── Build Approximate Centerline ──────────────────────────────────────────────
def _build_centerline():
    """
    Pair each blue cone with its nearest yellow cone, take the midpoint,
    then sort CLOCKWISE around the track centroid so pure-pursuit drives CW.
    """
    center = np.mean(MAP_CONES, axis=0)
    D      = distance.cdist(BLUE_CONES, YELLOW_CONES)
    mids   = np.array(
        [(BLUE_CONES[i] + YELLOW_CONES[np.argmin(D[i])]) / 2.0
         for i in range(len(BLUE_CONES))]
    )
    angles = np.arctan2(mids[:, 1] - center[1], mids[:, 0] - center[0])
    return mids[np.argsort(angles)[::-1]]   # descending angle = clockwise


CENTERLINE = _build_centerline()


# ── Simulation Parameters ─────────────────────────────────────────────────────
SENSOR_RANGE = 12.0   # metres – sensor visibility radius
NOISE_STD    = 0.20   # metres – measurement noise std-dev
WHEELBASE    = 3.0    # metres – bicycle model wheelbase
DT           = 0.1    # seconds – time step
SPEED        = 7.0    # m/s
LOOKAHEAD    = 5.5    # pure-pursuit lookahead distance (m)
N_FRAMES     = 130    # ≈ one full lap


# ── Utility Functions ─────────────────────────────────────────────────────────
def angle_wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def pure_pursuit(pos: np.ndarray, heading: float, path: np.ndarray) -> float:
    """Compute steering angle (rad) to follow *path* via pure-pursuit."""
    dists   = np.linalg.norm(path - pos, axis=1)
    nearest = int(np.argmin(dists))
    n       = len(path)
    target  = path[(nearest + 5) % n]       # fallback lookahead
    for k in range(nearest, nearest + n):
        pt = path[k % n]
        if np.linalg.norm(pt - pos) >= LOOKAHEAD:
            target = pt
            break
    alpha = angle_wrap(
        np.arctan2(target[1] - pos[1], target[0] - pos[0]) - heading
    )
    steer = np.arctan2(2.0 * WHEELBASE * np.sin(alpha), LOOKAHEAD)
    return float(np.clip(steer, -0.6, 0.6))


def local_to_global(local_pts: np.ndarray,
                    pos: np.ndarray, heading: float) -> np.ndarray:
    """Rotate + translate points from the car's local frame to world frame."""
    c, s = np.cos(heading), np.sin(heading)
    R    = np.array([[c, -s], [s, c]])       # local → world rotation
    return (R @ local_pts.T).T + pos


def get_measurements(pos: np.ndarray, heading: float) -> np.ndarray:
    """
    Simulate a 2-D lidar: return visible cone positions as noisy
    measurements in the car's LOCAL frame (x = forward, y = left).
    """
    dists   = np.linalg.norm(MAP_CONES - pos, axis=1)
    visible = MAP_CONES[dists < SENSOR_RANGE]
    if len(visible) == 0:
        return np.zeros((0, 2))
    c, s = np.cos(heading), np.sin(heading)
    R    = np.array([[c, s], [-s, c]])       # world → local (transpose of above)
    local = (R @ (visible - pos).T).T
    return local + np.random.normal(0, NOISE_STD, local.shape)


def draw_track(ax, alpha_b: float = 0.4, alpha_y: float = 0.4) -> None:
    ax.scatter(BLUE_CONES[:, 0],   BLUE_CONES[:, 1],
               c="royalblue", marker="^", s=65,  alpha=alpha_b,
               zorder=2, label="Blue cones")
    ax.scatter(YELLOW_CONES[:, 0], YELLOW_CONES[:, 1],
               c="gold",      marker="^", s=65,  alpha=alpha_y,
               zorder=2, label="Yellow cones")
    ax.scatter(BIG_ORANGE[:, 0],   BIG_ORANGE[:, 1],
               c="darkorange", marker="s", s=100, alpha=0.7,
               zorder=2, label="Start gate")


def draw_car(ax, pos: np.ndarray, heading: float) -> None:
    ax.scatter(pos[0], pos[1], c="red", s=160, zorder=7, label="Car")
    ax.arrow(pos[0], pos[1],
             2.2 * np.cos(heading), 2.2 * np.sin(heading),
             head_width=0.8, fc="red", ec="red", zorder=8)


def setup_ax(ax, subtitle: str = "") -> None:
    ax.set_xlim(-28, 28)
    ax.set_ylim(-22, 22)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, linestyle="--")
    if subtitle:
        ax.set_title(subtitle, fontsize=10)


# ── Abstract Base ─────────────────────────────────────────────────────────────
class Bot:
    def __init__(self):
        self.pos     = CAR_START_POS.copy()   # (2,) float64
        self.heading = CAR_START_HEADING      # radians

    def data_association(self, measurements, current_map):
        raise NotImplementedError

    def localization(self, velocity, steering):
        raise NotImplementedError

    def mapping(self, measurements):
        raise NotImplementedError


# ──  Solution ──────────────────────────────────────────────────────────
class Solution(Bot):
    """
    Extended Kalman Filter (EKF) localization.

    State vector: [x, y, ψ]  (position + heading)
    Predict:  bicycle kinematic model + Jacobian covariance propagation
    Update:   range-bearing observations to known map landmarks
    """
    def __init__(self):
        super().__init__()
        self.learned_map  = []
        self._global_meas = np.zeros((0, 2))
        self._assoc       = np.array([], dtype=int)

        # ── EKF state ──
        self.P = np.diag([0.1, 0.1, 0.01])          # initial covariance
        self.Q = np.diag([0.05**2, 0.05**2, 0.005**2])  # process noise
        self.R_obs = np.diag([0.3**2, 0.05**2])      # observation noise [range, bearing]

    # ------------------------------------------------------------------
    def localization(self, velocity, steering, measurements=None):
        """
        EKF-based localization with bicycle model predict + landmark update.

        Parameters
        ----------
        velocity  : float — forward speed (m/s)
        steering  : float — steering angle (rad)
        measurements : np.ndarray (M,2) | None
            Noisy local-frame measurements to known map cones.
            If provided, an EKF observation update is performed.
        """
        # ── PREDICT ──────────────────────────────────────────────────
        x, y, psi = self.pos[0], self.pos[1], self.heading

        x_new   = x + velocity * np.cos(psi) * DT
        y_new   = y + velocity * np.sin(psi) * DT
        psi_new = angle_wrap(psi + (velocity / WHEELBASE) * np.tan(steering) * DT)

        # Jacobian of motion model w.r.t. state
        F = np.array([
            [1, 0, -velocity * np.sin(psi) * DT],
            [0, 1,  velocity * np.cos(psi) * DT],
            [0, 0,  1],
        ])

        self.pos[0]  = x_new
        self.pos[1]  = y_new
        self.heading  = psi_new
        self.P = F @ self.P @ F.T + self.Q

        # ── UPDATE (observation correction) ───────────────────────────
        if measurements is not None and len(measurements) > 0 and len(MAP_CONES) > 0:
            # Transform local measurements to global frame for matching
            gm = local_to_global(measurements, self.pos, self.heading)

            for i, z_global in enumerate(gm):
                # Find nearest known landmark
                dists = np.linalg.norm(MAP_CONES - z_global, axis=1)
                j = int(np.argmin(dists))
                if dists[j] > 3.0:
                    continue   # skip outlier

                lm = MAP_CONES[j]

                # Expected range and bearing to landmark j
                dx = lm[0] - self.pos[0]
                dy = lm[1] - self.pos[1]
                r_exp  = np.sqrt(dx**2 + dy**2)
                b_exp  = angle_wrap(np.arctan2(dy, dx) - self.heading)

                if r_exp < 1e-6:
                    continue

                # Actual range and bearing from local measurement
                z_local = measurements[i]
                r_meas  = np.linalg.norm(z_local)
                b_meas  = np.arctan2(z_local[1], z_local[0])

                # Innovation
                innovation = np.array([
                    r_meas - r_exp,
                    angle_wrap(b_meas - b_exp),
                ])

                # Observation Jacobian H (range-bearing w.r.t. [x, y, psi])
                H = np.array([
                    [-dx / r_exp, -dy / r_exp,  0],
                    [ dy / r_exp**2, -dx / r_exp**2, -1],
                ])

                S = H @ self.P @ H.T + self.R_obs
                K = self.P @ H.T @ np.linalg.inv(S)

                update = K @ innovation
                self.pos[0]  += update[0]
                self.pos[1]  += update[1]
                self.heading  = angle_wrap(self.heading + update[2])
                self.P = (np.eye(3) - K @ H) @ self.P


# ── Problem 2 – Localization ───────────────────────────────────────────────────
def make_problem2():
    """
    Visualise dead-reckoning: the magenta trail is the car's estimated
    trajectory built purely from the kinematic model and steering commands.
    """
    sol     = Solution()
    path_x  = [float(sol.pos[0])]
    path_y  = [float(sol.pos[1])]
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle("Problem 2 – Localization  (Dead Reckoning / Kinematic Model)",
                 fontsize=13, fontweight="bold")

    def update(frame):
        ax.clear()
        steer = pure_pursuit(sol.pos, sol.heading, CENTERLINE)
        meas  = get_measurements(sol.pos, sol.heading)   # sensor observations
        sol.localization(SPEED, steer, measurements=meas)
        path_x.append(float(sol.pos[0]))
        path_y.append(float(sol.pos[1]))

        draw_track(ax)
        ax.plot(path_x, path_y, color="magenta", lw=2.0,
                alpha=0.85, zorder=4, label="Dead-reckoning path")
        draw_car(ax, sol.pos, sol.heading)
        setup_ax(ax,
            f"Frame {frame+1}/{N_FRAMES}  –  "
            f"pos=({sol.pos[0]:.1f}, {sol.pos[1]:.1f})  "
            f"ψ={np.degrees(sol.heading):.1f}°")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    ani = FuncAnimation(fig, update, frames=N_FRAMES, interval=100, repeat=True)
    return fig, ani


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Driverless Car Hackathon – SLAM Visualisation ===")
    print(f"  Blue cones   : {len(BLUE_CONES)}")
    print(f"  Yellow cones : {len(YELLOW_CONES)}")
    print(f"  Big orange   : {len(BIG_ORANGE)}")
    print(f"  Car start    : {CAR_START_POS}  "
          f"heading={np.degrees(CAR_START_HEADING):.1f}°")
    print(f"  Centerline   : {len(CENTERLINE)} waypoints (clockwise)")
    print("\nOpening 1 animation window …")

    # Keep references to prevent garbage collection of FuncAnimation objects.
    fig2, ani2 = make_problem2()

    plt.show()
