import matplotlib.pyplot as plt

import pickle
# list of rewards from the ddpo_test.py script
with open("past_metrics.pkl", "rb") as f:
    metrics = pickle.load(f)

rewards = [m["mean_reward"] for m in metrics]
avg_return = [m["episode_return"] for m in metrics]

# two side by side plots
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
# plot rewards
axs[0].plot(rewards)
axs[0].set_title("Rewards over Time")
axs[0].set_xlabel("Episode")
axs[0].set_ylabel("Reward")
# plot average return
axs[1].plot(avg_return)
axs[1].set_title("Average Return over Time")
axs[1].set_xlabel("Episode")
axs[1].set_ylabel("Average Return")
plt.tight_layout()
plt.show()
