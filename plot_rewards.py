import matplotlib.pyplot as plt

import pickle
# list of rewards from the ddpo_test.py script
# with open("past_metrics.pkl", "rb") as f:
#     metrics = pickle.load(f)

# rewards = [m["mean_reward"] for m in metrics]
# avg_return = [m["episode_return"] for m in metrics]

# # two side by side plots
# fig, axs = plt.subplots(1, 2, figsize=(12, 5))
# # plot rewards
# axs[0].plot(rewards)
# axs[0].set_title("Rewards over Time")
# axs[0].set_xlabel("Episode")
# axs[0].set_ylabel("Reward")
# # plot average return
# axs[1].plot(avg_return)
# axs[1].set_title("Average Return over Time")
# axs[1].set_xlabel("Episode")
# axs[1].set_ylabel("Average Return")
# plt.tight_layout()
# plt.show()

with open("sft_training_metrics.pkl", "rb") as f:
    metrics = pickle.load(f)

loss = metrics["loss"]
# moving average over loss with window of 64
loss = [sum(loss[max(0, i-64):i+1])/(i - max(0, i-64) + 1) for i in range(len(loss))]
avg_returns = {k: v for k, v in metrics.items() if k != "loss"}

# two side by side plots
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
# plot loss
axs[0].plot(loss)
axs[0].set_title("Loss over Time")
axs[0].set_xlabel("Batch")
axs[0].set_ylabel("Loss")
# plot average return for each objective
for objective, returns in avg_returns.items():
    axs[1].plot(returns, label=objective)
axs[1].set_title("Average Return over Time")
axs[1].set_xlabel("Evaluation Point")
axs[1].set_ylabel("Average Return")
axs[1].legend()
plt.tight_layout()
plt.show()
