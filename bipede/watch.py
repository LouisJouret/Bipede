"""Live training monitor: a matplotlib window that re-reads training_log_ppo.csv and
redraws every few seconds while training runs in another process.

    python -m bipede.watch                 # watch training_log_ppo.csv, refresh every 3 s
    python -m bipede.watch --log other.csv --interval 5
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def rolling(x, w=20):
    if len(x) < w:
        return np.array([]), np.array([])
    m = np.convolve(x, np.ones(w) / w, mode='valid')
    return np.arange(w - 1, len(x)), m


def read_log(path):
    """Return (episode, total_reward, distance_m); empty arrays if not ready yet.

    Parses by column position and drops any non-numeric rows, so it tolerates a
    missing/duplicated header and half-written lines from a concurrent writer.
    Columns: episode, distance_m, total_reward, steps.
    """
    empty = (np.array([]), np.array([]), np.array([]))
    try:
        d = np.genfromtxt(path, delimiter=',', usecols=(0, 1, 2), invalid_raise=False)
    except OSError:
        return empty
    d = np.atleast_2d(d)
    d = d[~np.isnan(d).any(axis=1)]          # drop header / partial rows
    if d.size == 0:
        return empty
    return d[:, 0], d[:, 2], d[:, 1]         # episode, total_reward, distance_m


def main():
    parser = argparse.ArgumentParser(description='Live plot of the training log.')
    parser.add_argument('--log', default='training_log_ppo.csv')
    parser.add_argument('--interval', type=float, default=3.0, help='refresh seconds')
    parser.add_argument('--window', type=int, default=20, help='rolling-mean window')
    args = parser.parse_args()

    plt.ion()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    fig.canvas.manager.set_window_title('Bipede training')

    while plt.fignum_exists(fig.number):
        ep, rew, dist = read_log(args.log)
        for a in ax:
            a.clear()
        if len(ep):
            ax[0].plot(ep, rew, alpha=0.35, label='per episode')
            ax[0].plot(*rolling(rew, args.window), label=f'mean ({args.window})')
            ax[1].plot(ep, dist, alpha=0.35, label='per episode')
            ax[1].plot(*rolling(dist, args.window), label=f'mean ({args.window})')
            ax[0].set_title(f'episode {int(ep[-1])}')
        ax[0].set_xlabel('episode'); ax[0].set_ylabel('total reward'); ax[0].legend(loc='upper left')
        ax[1].set_xlabel('episode'); ax[1].set_ylabel('distance walked (m)'); ax[1].legend(loc='upper left')
        ax[1].axhline(0, color='k', lw=0.5)
        fig.tight_layout()
        plt.pause(args.interval)


if __name__ == '__main__':
    main()
