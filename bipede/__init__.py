"""Bipede: PPO walking controller for a MuJoCo biped.

Modules:
    ppo    -- PPO agent and headless trainer + CLI (TensorFlow)
    env    -- MuJoCo model loading, PD controller, observation and reward (NumPy/MuJoCo)
    eval   -- watch a trained policy in the live viewer, or record it to an mp4
    watch  -- live matplotlib monitor that re-reads the training log while training runs

The heavy training runs headless via ``python -m bipede.ppo``; use ``python -m
bipede.eval`` to load a checkpoint and watch the policy walk.
"""
