import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('val_loss.txt',sep=':', header=None)

validation_loss = data[2]

plt.figure()
plt.title('UNet Validation loss', fontsize=20)
plt.xlabel('epoch', fontsize=15)
plt.ylabel('loss', fontsize=15)
plt.plot(validation_loss, linewidth=2)
plt.show()
