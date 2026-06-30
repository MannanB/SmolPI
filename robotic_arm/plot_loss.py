import matplotlib.pyplot as plt
import pickle

with open("training_metrics.pkl", "rb") as f:
    metrics = pickle.load(f)
loss = metrics["loss"]
wnd = 32*8
loss = [sum(loss[max(0, i-wnd):i+1])/(i - max(0, i-wnd) + 1) for i in range(len(loss))]

plt.plot(loss)
plt.tight_layout()
plt.show()
