import matplotlib.pyplot as plt
import json

# Replace with your actual training history if you have it
# Or use this dummy example if you don't have saved history

history_data = {
    'accuracy': [0.60, 0.68, 0.72, 0.76, 0.78, 0.80, 0.82, 0.85, 0.87, 0.90],
    'val_accuracy': [0.13, 0.14, 0.16, 0.18, 0.19, 0.20, 0.22, 0.23, 0.25, 0.26],
    'loss': [0.92, 0.73, 0.67, 0.58, 0.50, 0.45, 0.39, 0.33, 0.28, 0.24],
    'val_loss': [3.5, 3.4, 3.3, 3.0, 2.8, 2.6, 2.5, 2.3, 2.1, 2.0]
}

epochs = range(1, len(history_data['accuracy']) + 1)

plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, history_data['accuracy'], 'b-o', label='Training Accuracy')
plt.plot(epochs, history_data['val_accuracy'], 'r-o', label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(epochs, history_data['loss'], 'b-o', label='Training Loss')
plt.plot(epochs, history_data['val_loss'], 'r-o', label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("training_performance.png")
plt.show()
