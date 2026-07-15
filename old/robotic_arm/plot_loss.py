import matplotlib.pyplot as plt
import pickle

with open("training_metrics.pkl", "rb") as f:
    metrics = pickle.load(f)
with open("training_metrics_2.pkl", "rb") as f:
    metrics2 = pickle.load(f)
loss = metrics["loss"]
# loss2 = metrics2["loss"]
wnd = 32*64
loss = [sum(loss[max(0, i-wnd):i+1])/(i - max(0, i-wnd) + 1) for i in range(len(loss))]
# loss2 = [sum(loss2[max(0, i-wnd):i+1])/(i - max(0, i-wnd) + 1) for i in range(len(loss2))]

plt.plot(loss, label="loss")
# plt.plot(loss2, label="loss2")
plt.legend()
plt.tight_layout()
plt.show()
