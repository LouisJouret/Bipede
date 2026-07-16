"""Watch a trained PPO policy walk in a live MuJoCo viewer.

Loads the deterministic (no exploration noise) policy from a checkpoint and
runs episodes back-to-back in an interactive window. Same env, controller and
termination as training -- only the exploration noise is dropped.

    python -m bipede.eval                       # latest networks_ppo checkpoint
    python -m bipede.eval --networks networks_jumping_backup
    python -m bipede.eval --video gait.mp4       # headless: record instead of viewer
"""

import argparse
import time

import numpy as np
import mujoco

from . import env as E
from .ppo import PPOAgent

# PD reference ranges, must match bipede/ppo.py main().
LOW = np.array([-30, -30, -90, 0, -30, -30, -30, -90, 0, -30], dtype=np.float32)
HIGH = np.array([30, 30, 90, 90, 30, 30, 30, 90, 90, 30], dtype=np.float32)


def build_agent(model, data, networks, control='position'):
    obs_dim = len(E.get_observation(data))
    act_dim = model.nu
    agent = PPOAgent(obs_dim, act_dim, LOW, HIGH, control=control)
    if not agent.load(networks):
        raise SystemExit(f"No usable checkpoint in '{networks}'.")
    return agent


def standing(data):
    "True while the robot is still upright -- the fall test from is_done, minus the 2 s timeout."
    _, hip_y, hip_z, _ = E.get_hip_info(data)
    return (hip_y < 1) and (0.4 < hip_z < 1.5)


def run_episode(agent, model, data, on_step=None, done_fn=E.is_done):
    """Reset and roll out the deterministic policy until done_fn. Returns (distance, reward, steps)."""
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    controller = E.Controller()
    gait = E.GaitShaper()
    R, steps = 0.0, 0
    while not done_fn(data):
        obs = E.get_observation(data)
        a_env = agent.deterministic_action(obs)        # no exploration noise
        data.ctrl = a_env if agent.control == 'torque' else controller.get_motor_cmd(data, a_env)
        mujoco.mj_step(model, data, nstep=2)
        R += float(E.get_reward(model, data, a_env)) + gait.step(data)
        steps += 1
        if on_step is not None:
            on_step()
    _, hip_y, _, _ = E.get_hip_info(data)
    return -float(hip_y), R, steps


def watch(agent, model, data, realtime=True):
    import sys
    import mujoco.viewer
    if sys.platform == 'darwin' and 'mjpython' not in sys.executable:
        raise SystemExit(
            "The live viewer needs mjpython on macOS. Run:\n"
            "    .venv/bin/mjpython -m bipede.eval\n"
            "or record a video instead:\n"
            "    .venv/bin/python -m bipede.eval --video gait.mp4")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # follow the hip so the robot stays centred as it walks
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = 1
        viewer.cam.distance = 3.0
        ep = 0
        while viewer.is_running():
            dt = model.opt.timestep * 2                # one env step advances nstep=2

            def on_step(_dt=dt):
                viewer.sync()
                if realtime:
                    time.sleep(_dt)

            # run each episode until the robot actually falls, not the 2 s training cap
            dist, R, steps = run_episode(agent, model, data, on_step=on_step,
                                         done_fn=lambda d: not standing(d))
            print(f"episode {ep}: walked {dist:.2f} m, reward {R:.1f}, {steps} steps, stayed up {data.time:.1f}s")
            ep += 1


def tracking_camera(distance=3.0, azimuth=150, elevation=-20, trackbodyid=1):
    """A camera that follows a body (default: the hip / free-joint root, body 1)."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = trackbodyid
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def record(agent, model, data, path, max_seconds=100.0, fps=30):
    """Record one episode that runs until the robot falls, hard-capped at max_seconds."""
    import mediapy as media
    renderer = mujoco.Renderer(model, 480, 640)
    cam = tracking_camera()
    frames = []

    def on_step():
        if len(frames) < data.time * fps:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())

    done_fn = lambda d: (not standing(d)) or (d.time > max_seconds)
    dist, R, steps = run_episode(agent, model, data, on_step=on_step, done_fn=done_fn)
    reason = 'hit cap' if data.time > max_seconds else 'fell'
    print(f"stayed up {data.time:.1f}s ({reason}): walked {dist:.2f} m, reward {R:.1f}, {steps} steps")
    media.write_video(path, frames, fps=fps)
    print(f"wrote {len(frames)} frames ({len(frames)/fps:.1f}s) -> {path}")


def main():
    p = argparse.ArgumentParser(description='Watch or record a trained PPO policy.')
    p.add_argument('--xml', default='bipede.xml')
    p.add_argument('--networks', default='networks_ppo')
    p.add_argument('--video', default=None, help='write an mp4 instead of opening a live viewer')
    p.add_argument('--max-seconds', type=float, default=100.0,
                   help='video: hard cap; the episode otherwise runs until the robot falls')
    p.add_argument('--fast', action='store_true', help='live viewer: run as fast as possible, not real time')
    p.add_argument('--control', choices=['torque', 'position'], default='position', help='must match how the checkpoint was trained')
    args = p.parse_args()

    model, data = E.load_model(args.xml)
    agent = build_agent(model, data, args.networks, control=args.control)

    if args.video:
        record(agent, model, data, args.video, max_seconds=args.max_seconds)
    else:
        watch(agent, model, data, realtime=not args.fast)


if __name__ == '__main__':
    main()
