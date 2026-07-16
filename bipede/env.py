"""MuJoCo side of the biped: model loading, PD controller, observation and reward.

No dependency on TensorFlow, so this stays cheap to import and easy to unit-test.
"""

import numpy as np
import mujoco


def load_model(xml_path='bipede.xml'):
    "Build a fresh (model, data) pair from the MuJoCo XML."
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


class Controller():
    "Hand-tuned PD+I controller: turns the actor's reference joint angles into motor torques."

    def __init__(self):
        self.integral = 0

    def get_motor_cmd(self, data, ref_pos):
        state = 180 * data.qpos[7:] / np.pi
        vel = data.qvel[6:]
        erreur = state - ref_pos
        self.integral = np.clip(self.integral + erreur, -10, 10)
        cmd = np.clip(-5.5 * erreur - 2 * vel - 0.05 * self.integral, -100, 100)
        return cmd

    def reset(self):
        self.__init__()


def get_hip_info(data):
    hip_x = data.geom_xpos[1][0].copy()
    hip_y = data.geom_xpos[1][1].copy()
    hip_z = data.geom_xpos[1][2].copy()
    hip_y_vel = data.cvel[1][4].copy()
    return hip_x, hip_y, hip_z, hip_y_vel


def symlog(x):
    return np.sign(x) * np.log10(np.sign(x) * x + 1)


# Reward shaping knobs, mutable so the trainer can rebalance without editing code.
# TARGET_VELOCITY: forward speed (m/s) at which velocity reward saturates -- lower means
#   the policy is not pushed to lunge to earn its reward.
# ALIVE_BONUS: per-step reward just for staying upright. Must stay below the saturated
#   velocity reward (1.5*TARGET_VELOCITY) so walking still beats standing still.
# JOINT_VEL_COST: penalty on sum of squared joint velocities. Kills the "buzz the feet to
#   skate forward" exploit: a vibrating policy hits joint-vel^2 ~ 400+ vs ~0 standing and
#   tens for a real gait, so at this coef vibration costs >1/step while walking costs <0.2.
TARGET_VELOCITY = 0.5
ALIVE_BONUS = 0.15
JOINT_VEL_COST = 0.003


def get_reward(model, data, ref_pos):
    hip_x, hip_y, hip_z, hip_y_vel = get_hip_info(data)
    hip_z_initial = model.geom_pos[1][2]
    hip_z_vel = data.cvel[1][5]                         # vertical velocity of the hip

    fwd_vel = -hip_y_vel                                # robot advances in the -y direction
    target_v = TARGET_VELOCITY

    # every term is bounded to ~O(1) so no single penalty can swamp the walking signal.
    # forward progress, clipped both ways (no overshoot bonus, bounded backward penalty)
    # vel_reward = 1.5 * np.clip(fwd_vel, -0.5, target_v)         # in [-0.75, 1.5*target_v]
    vel_reward = 1.5 * fwd_vel        # in [-0.75, 1.5*target_v]
    # anti-jump terms: punish leaving the ground / bobbing vertically
    height_pen = -2.0 * (hip_z - hip_z_initial) ** 2           # ~[-0.7, 0]
    # vvel_pen = -np.minimum(0.1 * hip_z_vel ** 2, 1.0)          # in [-1, 0]
    # ctrl_cost = -1e-5 * np.sum(np.square(data.ctrl))          # ~[-1, 0] at torque saturation
    smooth_pen = -JOINT_VEL_COST * np.sum(data.qvel[6:] ** 2)  # kills the foot-vibration exploit
    alive = ALIVE_BONUS                                       # pays for staying up; balance must be worthwhile
    dead_penalty = -FALL_PENALTY if is_fallen(data) else 0.0

    reward = vel_reward + height_pen + smooth_pen + alive + dead_penalty
    return reward


# --- gait shaping: reward actual steps, not toppling ---------------------------------------
# Forward-velocity reward alone is gamed by leaning past the feet and falling forward (stiff-legged,
# feet planted). To make the policy STEP, reward foot air-time: a foot that lifts, swings, and lands
# earns ~its airborne duration on touchdown. A planted (toppling) foot earns nothing, so this term
# only pays for genuine steps. Geom ids from bipede.xml: floor=0, left foot={6,7}, right foot={12,13}.
_FLOOR_GEOM = 0
_LEFT_FOOT_GEOMS = (6, 7)
_RIGHT_FOOT_GEOMS = (12, 13)
AIR_TIME_WEIGHT = 5.0     # reward per second of foot air-time credited at touchdown
AIR_TIME_TARGET = 0.4     # cap: a good step is ~0.4 s of swing (don't reward hopping forever)
# Dense swing-foot clearance: air-time alone only pays after a full 0.1 s swing the policy never
# samples, so it is undiscoverable. This pays for the swing foot's height every step (even a mm),
# giving PPO a gradient toward lifting a foot from the first noisy action. Gated on single support.
_FOOT_Z_BASELINE = 0.025  # geom_xpos z of a grounded foot (measured); clearance = current - this
CLEARANCE_WEIGHT = 3.0    # reward per metre of swing-foot clearance, per step
CLEARANCE_TARGET = 0.08   # cap clearance reward at an 8 cm lift


class GaitShaper:
    "Stateful: tracks per-foot air-time and credits a bonus when a foot lands after a real swing."

    def __init__(self):
        self.reset()

    def reset(self):
        self.air = [0.0, 0.0]     # seconds airborne, [left, right]

    def _contacts(self, data):
        left = right = False
        for i in range(data.ncon):
            c = data.contact[i]
            pair = (c.geom1, c.geom2)
            if _FLOOR_GEOM not in pair:
                continue
            other = pair[0] if pair[1] == _FLOOR_GEOM else pair[1]
            if other in _LEFT_FOOT_GEOMS:
                left = True
            elif other in _RIGHT_FOOT_GEOMS:
                right = True
        return (left, right)

    def step(self, data, dt=0.01):
        "Advance one env step and return the gait bonus: dense swing-foot clearance + air-time on landing."
        left, right = self._contacts(data)
        bonus = 0.0
        # dense (discoverable): reward the swing foot's clearance during single support
        lz = 0.5 * (data.geom_xpos[6][2] + data.geom_xpos[7][2]) - _FOOT_Z_BASELINE
        rz = 0.5 * (data.geom_xpos[12][2] + data.geom_xpos[13][2]) - _FOOT_Z_BASELINE
        if left and not right:
            bonus += CLEARANCE_WEIGHT * min(max(rz, 0.0), CLEARANCE_TARGET)
        elif right and not left:
            bonus += CLEARANCE_WEIGHT * min(max(lz, 0.0), CLEARANCE_TARGET)
        # sparse: credit a completed step's air-time at touchdown
        for i, in_contact in enumerate((left, right)):
            if in_contact:
                if self.air[i] > 0.1:                       # landed after a genuine swing
                    bonus += AIR_TIME_WEIGHT * min(self.air[i], AIR_TIME_TARGET)
                self.air[i] = 0.0
            else:
                self.air[i] += dt
        return bonus


# Fixed per-block observation scaling so every input lands ~[-1, 1]: the tanh hidden layers
# saturate (vanishing gradient) on large inputs, and unscaled blocks would dominate the small
# ones (raw joint velocities reach +-30 while joint angles stay within +-1.6 rad).
ANG_VEL_SCALE = 0.25     # base angular velocity: +-4 rad/s -> +-1
JOINT_VEL_SCALE = 0.1    # joint velocities: +-10 rad/s -> +-1 (fall transients clip beyond)


def get_observation(data):
    """Observation (32-dim): height, base orientation, joint angles, scaled velocities.

    Orientation is the gravity direction in the body frame (bounded, smooth, no wraparound
    -- upright = (0,0,-1)) plus sin/cos of yaw. Yaw must stay observable because the reward
    is world-frame directional (forward = -y), but sin/cos keeps it continuous.
    Absolute x,y are dropped so the policy is translation-invariant.
    """
    height = data.qpos[2:3]                            # ~[0, 0.35] in practice
    R = data.xmat[1].reshape(3, 3)                     # base body -> world rotation
    gravity_body = -R[2, :]                            # world "down" seen from the body
    yaw = np.arctan2(R[1, 0], R[0, 0])
    heading = np.array([np.sin(yaw), np.cos(yaw)])
    return np.concatenate([
        height,
        gravity_body,
        heading,
        data.qpos[7:],                                 # joint angles, rad, +-1.6 max
        data.qvel[0:3],                                # base linear velocity, ~walking speed already O(1)
        data.qvel[3:6] * ANG_VEL_SCALE,
        data.qvel[6:] * JOINT_VEL_SCALE,
    ], axis=0)


# Max episode length in seconds. Mutable so the trainer can run a curriculum that
# lengthens the horizon as the policy learns to stay up (see bipede/ppo.py).
EPISODE_TIME_LIMIT = 2.0


# One-off penalty applied when an episode ends by falling (not by the time limit).
# Makes "lunge forward then faceplant" net-negative vs. staying upright, so the policy
# can't farm velocity reward by sacrificing balance. Standing still is trivially stable
# (holding the nominal pose stands >30 s), so this is the missing cost, not a crutch.
FALL_PENALTY = 20.0


def is_fallen(data):
    "The fall half of is_done: hip left the safe band. Ignores the time limit."
    _, hip_y, hip_z, _ = get_hip_info(data)
    return not (hip_y < 1 and hip_z > 0.4 and hip_z < 1.5)


def is_done(data):
    "Episode terminates on timeout or a fall (hip out of the safe height band)."
    if data.time > EPISODE_TIME_LIMIT:                  # time is up
        return True
    return is_fallen(data)                              # fall
