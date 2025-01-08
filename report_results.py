import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
data = pd.read_csv('results_ablation.csv')

epochs = data['epoch']


plt.figure(figsize=(10, 6))
plt.plot(epochs, data['train/cls_loss'], label='Train Classification Loss')
plt.plot(epochs, data['val/cls_loss'], label='Validation Classification Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Classification Loss')
plt.legend()
plt.grid()
plt.show()

# Plot box loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, data['train/box_loss'], label='Train Box Loss')
plt.plot(epochs, data['val/box_loss'], label='Validation Box Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Box Loss')
plt.legend()
plt.grid()
plt.show()
#
# Plot DFL loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, data['train/dfl_loss'], label='Train DFL Loss')
plt.plot(epochs, data['val/dfl_loss'], label='Validation DFL Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Distribution Focal Loss (DFL)')
plt.legend()
plt.grid()
plt.show()

# # Plot mAP@50
plt.figure(figsize=(10, 6))
plt.plot(epochs, data['metrics/mAP50(B)'], label='mAP@50')
plt.plot(epochs, data['metrics/mAP50-95(B)'], label='mAP@50-95')
plt.xlabel('Epoch')
plt.ylabel('mAP')
plt.title('mAP Metrics')
plt.legend()
plt.grid()
plt.show()

# Extract precision and recall values
precision = data['metrics/precision(B)']
recall = data['metrics/recall(B)']

# Plot Precision-Recall curve
# plt.figure(figsize=(10, 6))
# plt.plot(recall, precision, marker='o')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.grid()
# plt.show()
