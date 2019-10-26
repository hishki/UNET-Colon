import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('train_loss.txt',sep=':', header=None)

train_loss = data[2]

plt.figure()
plt.title('UNet training loss', fontsize=20)
plt.xlabel('epoch', fontsize=15)
plt.ylabel('loss', fontsize=15)
plt.plot(train_loss, linewidth=2)
plt.show()
