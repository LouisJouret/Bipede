"""PPO trainer for the biped walking controller.

Uses env.py for everything environment-side: observation, reward, PD-target
action space, and termination.

    python -m bipede.ppo --fresh --watch

Actions are handled in a normalised [-1, 1] space per joint and mapped to the
PD reference ranges for the controller. The policy is a diagonal Gaussian with a
state-independent (learnable) log-std; training uses GAE(lambda) advantages and
the clipped surrogate objective. A fraction of episodes (--rsi) start from
mid-gait snapshots of past healthy rollouts (Reference State Initialization)
instead of the standing pose, so the policy also trains on continuing a gait.
"""

import os
import argparse
from collections import deque

import numpy as np
from matplotlib.figure import Figure
import tensorflow as tf

from . import env as E

# This action space feeds joint-angle references (up to +-90 deg) to a stiff PD controller at
# 100 Hz, so exploration noise is violently destabilising: a robot that stands >30 s under clean
# control falls in <1 s at std 0.2, and in 0.65 s at std 0.5. Exploration must therefore be GENTLE
# -- start modest and allow it to anneal to a small floor so the policy can actually stay upright
# during training and experience the reward landscape.
LOG_STD_INIT = -2.5   # exp(-2.5) ~ 0.08 -> survives ~5 s from the standing pose, so episodes give signal
LOG_STD_MIN = -3.0    # exp(-3.0) ~ 0.05 -> ~+-4.5 deg jitter, low enough to balance through


def mlp(obs_dim, sizes, out_dim, out_activation=None, zero_output=False):
    inp = tf.keras.Input(shape=(obs_dim,))
    x = inp
    for s in sizes:
        x = tf.keras.layers.Dense(s, activation='tanh')(x)
    # zero_output: a fresh actor then outputs 0 -> the neutral (standing) action, so an
    # untrained policy starts upright instead of in a random, usually-falling pose.
    out_kernel = 'zeros' if zero_output else 'glorot_uniform'
    out = tf.keras.layers.Dense(out_dim, activation=out_activation, kernel_initializer=out_kernel)(x)
    return tf.keras.Model(inp, out)


class PPOAgent:
    def __init__(self, obs_dim, act_dim, action_low, action_high,
                 clip=0.2, ent_coef=0.0, actor_lr=3e-4, critic_lr=1e-3,
                 control='torque', torque_scale=100.0, log_std_init=LOG_STD_INIT):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        self.control = control                 # 'torque': action IS the motor torque; 'position': PD reference angle
        self.torque_scale = float(torque_scale)
        self.actor = mlp(obs_dim, [512, 512], act_dim, out_activation='tanh', zero_output=True)   # mean in [-1,1]
        self.critic = mlp(obs_dim, [512, 512], 1)
        self.log_std = tf.Variable(np.full(act_dim, log_std_init, dtype=np.float32), name='log_std')
        self.actor_opt = tf.keras.optimizers.Adam(actor_lr)
        self.critic_opt = tf.keras.optimizers.Adam(critic_lr)
        self.clip = clip
        # a tf.Variable so a decay schedule can lower it between updates without retracing
        self.ent_coef = tf.Variable(ent_coef, dtype=tf.float32, trainable=False, name='ent_coef')

    # --- action space mapping (normalised [-1,1] -> what the actuators receive) ---
    # torque mode:  a_norm maps directly to joint torque (|a|=1 -> +-torque_scale, the motor
    #   ctrlrange). No PD layer -- the policy commands forces itself. a_norm=0 -> 0 torque (limp),
    #   so there is no free static "stand" and the policy must learn to hold itself up.
    # position mode: a_norm=0 -> 0 deg (standing pose); +/-1 -> the joint limits. The two half-
    #   ranges are scaled independently so 0 lands on the standing pose despite range asymmetry.
    def _to_env(self, a_norm):
        a = np.clip(a_norm, -1.0, 1.0)
        if self.control == 'torque':
            return a * self.torque_scale
        return np.where(a >= 0, a * self.action_high, -a * self.action_low)

    @staticmethod
    def _logp(a, mean, std):
        var = std ** 2
        logp = -0.5 * (((a - mean) ** 2) / var + 2.0 * tf.math.log(std) + tf.math.log(2.0 * np.pi))
        return tf.reduce_sum(logp, axis=-1)

    @tf.function(reduce_retracing=True)
    def _sample(self, obs):
        "Compiled hot path (one graph call per env step): eager Keras calls here leak memory and are ~4x slower."
        mean = self.actor(obs)
        std = tf.exp(self.log_std)
        a_norm = mean + tf.random.normal(tf.shape(mean)) * std
        logp = self._logp(a_norm, mean, std)
        value = self.critic(obs)[0, 0]
        return a_norm, logp, value

    def act(self, obs):
        "Sample an action. Returns (env_action, norm_action, logp, value) as numpy/floats."
        obs = tf.constant(obs.reshape(1, -1).astype(np.float32))
        a_norm, logp, value = self._sample(obs)
        a_norm = a_norm.numpy()[0]
        return self._to_env(a_norm), a_norm, float(logp.numpy()[0]), float(value.numpy())

    def deterministic_action(self, obs):
        obs = obs.reshape(1, -1).astype(np.float32)
        return self._to_env(self.actor(obs).numpy()[0])

    def value(self, obs):
        return float(self.critic(obs.reshape(1, -1).astype(np.float32))[0, 0].numpy())

    @tf.function(reduce_retracing=True)
    def _update_actor(self, obs, act, logp_old, adv):
        with tf.GradientTape() as tape:
            mean = self.actor(obs)
            std = tf.exp(self.log_std)
            logp = self._logp(act, mean, std)
            ratio = tf.exp(logp - logp_old)
            clipped = tf.clip_by_value(ratio, 1.0 - self.clip, 1.0 + self.clip)
            pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv, clipped * adv))
            entropy = tf.reduce_sum(self.log_std + 0.5 * np.log(2.0 * np.pi * np.e))
            loss = pi_loss - self.ent_coef * entropy
        variables = self.actor.trainable_variables + [self.log_std]
        grads = tape.gradient(loss, variables)
        grads, _ = tf.clip_by_global_norm(grads, 0.5)
        self.actor_opt.apply_gradients(zip(grads, variables))
        self.log_std.assign(tf.maximum(self.log_std, LOG_STD_MIN))   # keep exploration alive
        approx_kl = tf.reduce_mean(logp_old - logp)                  # how far this batch moved the policy
        return pi_loss, approx_kl

    @tf.function(reduce_retracing=True)
    def _update_critic(self, obs, ret):
        with tf.GradientTape() as tape:
            v = self.critic(obs)[:, 0]
            v_loss = tf.reduce_mean((ret - v) ** 2)
        grads = tape.gradient(v_loss, self.critic.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 0.5)
        self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_variables))
        return v_loss

    def update(self, obs, act, logp_old, adv, ret, epochs=10, minibatch=256, target_kl=0.03):
        n = len(obs)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        obs = tf.constant(obs, tf.float32); act = tf.constant(act, tf.float32)
        logp_old = tf.constant(logp_old, tf.float32)
        adv = tf.constant(adv, tf.float32); ret = tf.constant(ret, tf.float32)
        idx = np.arange(n)
        for _ in range(epochs):
            np.random.shuffle(idx)
            kls = []
            for start in range(0, n, minibatch):
                mb = idx[start:start + minibatch]
                _, kl = self._update_actor(tf.gather(obs, mb), tf.gather(act, mb),
                                           tf.gather(logp_old, mb), tf.gather(adv, mb))
                self._update_critic(tf.gather(obs, mb), tf.gather(ret, mb))
                kls.append(float(kl))
            # trust region: stop updating on this batch once the policy has drifted too far,
            # so one over-eager update can't knock a good policy into a bad basin (PPO's key guard)
            if np.mean(kls) > target_kl:
                break

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.actor.save_weights(os.path.join(path, 'actor.weights.h5'))
        self.critic.save_weights(os.path.join(path, 'critic.weights.h5'))
        np.save(os.path.join(path, 'log_std.npy'), self.log_std.numpy())

    def load(self, path):
        af = os.path.join(path, 'actor.weights.h5')
        cf = os.path.join(path, 'critic.weights.h5')
        if not (os.path.exists(af) and os.path.exists(cf)):
            print(f"No PPO checkpoint in '{path}'. Starting from scratch.")
            return False
        try:
            self.actor.load_weights(af)
            self.critic.load_weights(cf)
        except Exception as e:
            reason = str(e).splitlines()[0] if str(e) else type(e).__name__
            print(f"Checkpoint in '{path}' is incompatible (saved with an older observation "
                  f"format? {reason}). Starting from scratch.")
            return False
        ls = os.path.join(path, 'log_std.npy')
        if os.path.exists(ls):
            self.log_std.assign(np.load(ls))
        print(f"Loaded PPO checkpoint from '{path}'.")
        return True


def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.append(np.asarray(values, dtype=np.float32), last_value)
    dones = np.asarray(dones, dtype=np.float32)
    adv = np.zeros_like(rewards)
    lastgae = 0.0
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
        adv[t] = lastgae = delta + gamma * lam * nonterminal * lastgae
    return adv, adv + values[:-1]


def save_progress_png(reward_history, path='progress_ppo.png'):
    # bare Figure, not pyplot: the pyplot state machine leaks ~0.3 MB per figure in long runs
    fig = Figure()
    ax = fig.add_subplot()
    ax.plot(reward_history, label='episode reward')
    if len(reward_history) >= 20:
        m = np.convolve(reward_history, np.ones(20) / 20, mode='valid')
        ax.plot(range(19, len(reward_history)), m, label='mean (20)')
    ax.set_xlabel('episode'); ax.set_ylabel('total reward'); ax.legend()
    fig.savefig(path, dpi=100, bbox_inches='tight')


def horizon_for(steps_done, total_steps, start, end, ramp_frac=0.5):
    "Episode time-limit curriculum: ramp linearly start->end over the first ramp_frac of training, then hold."
    if total_steps <= 0 or start >= end:
        return end
    frac = min(1.0, steps_done / (total_steps * ramp_frac))
    return start + (end - start) * frac


class StateBank:
    """Reference State Initialization (RSI): a pool of mid-gait (qpos, qvel) snapshots
    harvested from the policy's own healthy rollouts. Starting a fraction of episodes
    from these states lets the policy practice steps 3, 4, 5... instead of spending
    almost all its data on steps 1-2 from the standing pose."""

    CAPACITY = 2000      # oldest states rotate out, keeping the pool near the current policy's distribution
    SNAPSHOT_EVERY = 5   # env steps between snapshots (50 ms) -- avoids near-duplicate states
    EXCLUDE_LAST_S = 1.0 # states this close to a fall are already toppling; restarting from them is unrecoverable
    MIN_DISTANCE = 0.3   # only episodes that actually walked donate states (others are ~the standing pose)

    def __init__(self):
        self.states = deque(maxlen=self.CAPACITY)
        self.pending = []   # (time, qpos, qvel) of the running episode, filtered at commit
        self._step = 0

    def start_episode(self):
        self.pending = []
        self._step = 0

    def record(self, data):
        self._step += 1
        if self._step % self.SNAPSHOT_EVERY == 0:
            self.pending.append((data.time, data.qpos.copy(), data.qvel.copy()))

    def commit(self, fell, end_time, distance):
        "Episode over: move the healthy snapshots into the pool, drop the doomed tail."
        if distance >= self.MIN_DISTANCE:
            cutoff = end_time - self.EXCLUDE_LAST_S if fell else np.inf
            self.states.extend((q, v) for t, q, v in self.pending if t < cutoff)
        self.start_episode()

    def sample(self):
        if not self.states:
            return None
        return self.states[np.random.randint(len(self.states))]


def record_gait(agent, model, renderer, path, max_seconds=10.0, fps=30):
    """Roll out the deterministic policy in a FRESH sim and write a tracking-camera mp4.
    Uses its own MjData so it never touches the training rollout state."""
    import mujoco
    import mediapy as media
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data); mujoco.mj_forward(model, data)
    controller = E.Controller()
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = 1; cam.distance = 3.0; cam.azimuth = 150; cam.elevation = -20
    frames = []
    while data.time < max_seconds:
        a_env = agent.deterministic_action(E.get_observation(data))
        data.ctrl = a_env if agent.control == 'torque' else controller.get_motor_cmd(data, a_env)
        mujoco.mj_step(model, data, nstep=2)
        if E.is_fallen(data):
            break
        if len(frames) < data.time * fps:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())
    if frames:
        media.write_video(path, frames, fps=fps)
    return len(frames)


def train_ppo(agent, model, data, total_steps, steps_per_update=4096,
              networks_dir='networks_ppo', log_path='training_log_ppo.csv',
              save_every_updates=5, horizon_start=3.0, horizon_end=10.0,
              ent_start=0.01, ent_end=0.0, video_every=0, video_dir='gait_videos', gamma=0.997,
              rsi=0.5):
    import mujoco
    controller = E.Controller()
    gait = E.GaitShaper()
    bank = StateBank()
    mujoco.mj_resetData(model, data)
    home_xy = data.qpos[:2].copy()   # the floor is flat and the obs is translation-invariant, so restored states are re-homed here

    renderer = None
    if video_every:
        try:
            renderer = mujoco.Renderer(model, 480, 640)
            os.makedirs(video_dir, exist_ok=True)
        except Exception as e:
            print(f"[video] disabled -- renderer init failed: {e}")
            video_every = 0

    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        with open(log_path, 'w') as f:
            f.write('episode,distance_m,total_reward,steps\n')

    def reset():
        mujoco.mj_resetData(model, data)
        snap = bank.sample() if np.random.rand() < rsi else None
        if snap is not None:
            qpos, qvel = snap
            data.qpos[:] = qpos
            data.qvel[:] = qvel
            data.qpos[:2] = home_xy   # keep logged distance (-hip_y) and the walked-backward cutoff meaningful
        else:
            data.qpos[7:] += np.random.uniform(-0.05, 0.05, size=data.qpos[7:].shape)
        mujoco.mj_forward(model, data)
        controller.reset()
        gait.reset()
        bank.start_episode()

    reset()
    reward_history = []
    ep_reward, ep_steps, episode = 0.0, 0, 0
    steps_done, update_i = 0, 0
    best_reward = -1e18                        # track the best policy so a later collapse can't erase it

    while steps_done < total_steps:
        # lengthen the episode horizon as training progresses so staying up long pays off
        E.EPISODE_TIME_LIMIT = horizon_for(steps_done, total_steps, horizon_start, horizon_end)
        # anneal exploration: high entropy early to explore, decaying so the mean policy can sharpen
        frac = min(1.0, steps_done / total_steps) if total_steps > 0 else 1.0
        agent.ent_coef.assign(ent_start + (ent_end - ent_start) * frac)
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
        for _ in range(steps_per_update):
            obs = E.get_observation(data)
            env_a, a_norm, logp, value = agent.act(obs)
            # torque mode: env_a IS the motor torque; position mode: PD servos to the reference
            data.ctrl = env_a if agent.control == 'torque' else controller.get_motor_cmd(data, env_a)
            mujoco.mj_step(model, data, nstep=2)
            reward = E.get_reward(model, data, env_a) + gait.step(data)   # + reward for real steps
            done = E.is_done(data)
            # NOTE: get_reward already charges FALL_PENALTY on the fallen step -- do not subtract it
            # again here (it used to be double-counted, making every fall cost 2x the configured value)
            if not done:
                bank.record(data)                      # candidate RSI snapshot; doomed tail filtered at commit

            obs_buf.append(obs); act_buf.append(a_norm); logp_buf.append(logp)
            rew_buf.append(reward); val_buf.append(value); done_buf.append(float(done))
            ep_reward += reward; ep_steps += 1; steps_done += 1

            if done:
                _, hip_y, _, _ = E.get_hip_info(data)
                distance = -hip_y
                bank.commit(E.is_fallen(data), data.time, float(distance))
                reward_history.append(ep_reward)
                with open(log_path, 'a') as f:
                    f.write(f"{episode},{float(distance):.4f},{float(ep_reward):.4f},{ep_steps}\n")
                if episode % 20 == 0:
                    print(f"episode {episode}: walked {np.round(distance,2)}m  reward {np.round(ep_reward,2)}  steps {ep_steps}")
                episode += 1; ep_reward, ep_steps = 0.0, 0
                reset()
                if video_every and episode % video_every == 0:
                    path = os.path.join(video_dir, f"ep_{episode:06d}.mp4")
                    try:
                        n = record_gait(agent, model, renderer, path)
                        print(f"[video] episode {episode}: wrote {n} frames -> {path}")
                    except Exception as e:
                        print(f"[video] episode {episode}: failed -- {e}")

        last_value = 0.0 if done_buf[-1] else agent.value(E.get_observation(data))
        adv, ret = compute_gae(rew_buf, val_buf, done_buf, last_value, gamma=gamma)
        agent.update(np.array(obs_buf, dtype=np.float32), np.array(act_buf, dtype=np.float32),
                     np.array(logp_buf, dtype=np.float32), adv, ret)

        update_i += 1
        if update_i % save_every_updates == 0:
            agent.save(networks_dir)
            save_progress_png(reward_history)
            recent = float(np.mean(reward_history[-20:])) if len(reward_history) >= 20 else -1e18
            if recent > best_reward:           # keep a copy of the best-so-far policy
                best_reward = recent
                agent.save(networks_dir + '_best')
            print(f"[update {update_i}] steps {steps_done}/{total_steps}  episodes {episode}  "
                  f"last-20 reward {np.mean(reward_history[-20:]):.1f} (best {best_reward:.1f})  std {np.round(np.exp(agent.log_std.numpy()),2).mean():.2f}  "
                  f"horizon {E.EPISODE_TIME_LIMIT:.1f}s  ent {float(agent.ent_coef.numpy()):.4f}  rsi bank {len(bank.states)}")

    agent.save(networks_dir)
    save_progress_png(reward_history)


def main():
    p = argparse.ArgumentParser(description='Train the biped walking controller with PPO.')
    p.add_argument('--steps', type=int, default=100_000_000, help='total environment steps')
    p.add_argument('--xml', default='bipede.xml')
    p.add_argument('--networks', default='networks_ppo')
    p.add_argument('--log', default='training_log_ppo.csv')
    p.add_argument('--rollout', type=int, default=4096, help='env steps collected per PPO update')
    p.add_argument('--gamma', type=float, default=0.997, help='discount factor; ~1/(1-gamma) steps of horizon (0.997 ~= 3 s)')
    p.add_argument('--ent-coef', type=float, default=0.01, help='entropy coefficient at start of training (anneals to --ent-coef-end; keeps the std from collapsing to the floor early)')
    p.add_argument('--ent-coef-end', type=float, default=0.0, help='entropy coefficient after annealing (std floor still prevents collapse)')
    p.add_argument('--horizon-start', type=float, default=3.0, help='episode time-limit at start of training (s)')
    p.add_argument('--horizon-end', type=float, default=10.0, help='episode time-limit after the curriculum ramp (s)')
    p.add_argument('--alive', type=float, default=E.ALIVE_BONUS, help='per-step reward for staying upright')
    p.add_argument('--target-v', type=float, default=E.TARGET_VELOCITY, help='forward speed (m/s) where velocity reward saturates')
    p.add_argument('--fall-penalty', type=float, default=E.FALL_PENALTY, help='one-off cost when an episode ends by falling')
    p.add_argument('--joint-vel-cost', type=float, default=E.JOINT_VEL_COST, help='penalty on sum of squared joint velocities (anti-vibration)')
    p.add_argument('--air-time-weight', type=float, default=E.AIR_TIME_WEIGHT, help='reward per second of foot air-time at touchdown (rewards completed steps)')
    p.add_argument('--clearance-weight', type=float, default=E.CLEARANCE_WEIGHT, help='dense reward per metre of swing-foot clearance (makes foot-lifting discoverable)')
    p.add_argument('--rsi', type=float, default=0.5, help='fraction of episodes started from a mid-gait snapshot of a past healthy rollout (Reference State Initialization); 0 disables')
    p.add_argument('--control', choices=['torque', 'position'], default='position', help='action space: PD position reference (default, robot is statically stable), or direct joint torque')
    p.add_argument('--torque-scale', type=float, default=100.0, help='torque mode: |action|=1 maps to this torque (motor ctrlrange ~100)')
    p.add_argument('--init-std', type=float, default=None, help='initial exploration std (default: 0.3 for torque, 0.08 for position)')
    p.add_argument('--fresh', action='store_true')
    p.add_argument('--seed', type=int, default=69)
    p.add_argument('--watch', action='store_true')
    p.add_argument('--video-every', type=int, default=1000, help='record a tracking-camera gait mp4 every N episodes (0 = off)')
    p.add_argument('--video-dir', default='gait_videos', help='folder collecting one ep_XXXXXX.mp4 per recording (kept, not overwritten)')
    args = p.parse_args()

    import random
    np.random.seed(args.seed); tf.random.set_seed(args.seed); random.seed(args.seed)

    E.ALIVE_BONUS = args.alive
    E.TARGET_VELOCITY = args.target_v
    E.FALL_PENALTY = args.fall_penalty
    E.JOINT_VEL_COST = args.joint_vel_cost
    E.AIR_TIME_WEIGHT = args.air_time_weight
    E.CLEARANCE_WEIGHT = args.clearance_weight

    model, data = E.load_model(args.xml)
    obs_dim = len(E.get_observation(data)); act_dim = model.nu
    low = np.array([-30, -30, -90, 0, -30, -30, -30, -90, 0, -30], dtype=np.float32)
    high = np.array([30, 30, 90, 90, 30, 30, 30, 90, 90, 30], dtype=np.float32)
    init_std = args.init_std if args.init_std is not None else (0.3 if args.control == 'torque' else float(np.exp(LOG_STD_INIT)))
    print(f"PPO: obs dim = {obs_dim}, act dim = {act_dim}  |  control={args.control}  "
          f"torque_scale={args.torque_scale}  init_std={init_std:.2f}  |  alive={E.ALIVE_BONUS}  target_v={E.TARGET_VELOCITY}")

    agent = PPOAgent(obs_dim, act_dim, low, high, ent_coef=args.ent_coef,
                     control=args.control, torque_scale=args.torque_scale, log_std_init=float(np.log(init_std)))
    if not args.fresh:
        agent.load(args.networks)

    watcher = None
    if args.watch:
        import sys, subprocess
        watcher = subprocess.Popen([sys.executable, '-m', 'bipede.watch', '--log', args.log])

    try:
        train_ppo(agent, model, data, args.steps, steps_per_update=args.rollout,
                  networks_dir=args.networks, log_path=args.log,
                  horizon_start=args.horizon_start, horizon_end=args.horizon_end,
                  ent_start=args.ent_coef, ent_end=args.ent_coef_end,
                  video_every=args.video_every, video_dir=args.video_dir, gamma=args.gamma,
                  rsi=args.rsi)
    finally:
        if watcher is not None:
            watcher.terminate()


if __name__ == '__main__':
    main()
