import pytest
import warnings
import hypothesis
import hypothesis.strategies as st

import os
import subprocess

EXAMPLE_DIR = "gym"
SCRIPTS = ["ddpg_gym_pendulum.py",
           "cem_gym_cartpole.py",
           "dqn_gym_cartpole.py",
           "q_learning_gym_frozen_lake.py"]
EXAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "source", "examples"))
COMMANDS = [f"python {os.path.join(EXAMPLES_DIR, EXAMPLE_DIR, script)}" for script in SCRIPTS]


@pytest.mark.parametrize("command", COMMANDS)
def test_scripts(capsys, command):
    try:
        import gym
    except ImportError as e:
        warnings.warn(f"\n\nUnable to import gym ({e}).\nThis test will be skipped\n")
        return

    subprocess.run(command, shell=True, check=True)
