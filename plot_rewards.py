import matplotlib.pyplot as plt

import pickle
# list of rewards from the ddpo_test.py script
with open("vla_rollout_rewards.pkl", "rb") as f:
    rewards = pickle.load(f)

plt.figure(figsize=(10, 5))
plt.plot(rewards, marker='o')
plt.title("Mean Rewards per Update during DDPO Training")
plt.xlabel("Update Number")
plt.ylabel("Mean Reward")
plt.grid()
# plt.savefig("ddpo_rewards.png")
plt.show()