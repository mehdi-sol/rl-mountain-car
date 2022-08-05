import numpy as np
from agent import MountainCarContinuousAgent


if __name__ == "__main__":
    agent = MountainCarContinuousAgent()
    agent.train(20)
    agent.show_off()