import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

# ======================
# CONFIG
# ======================
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20

train_dir = 'dataset/train'
val_dir = 'dataset/val'
unlabeled_dir = 'dataset/unlabeled_dataset'
pseudo_labeled_dir = 'pseudo_labeled_dataset'

# ======================
# ENABLE MIXED PRECISION (FP16 for Tensor Cores on RTX GPUs)
# ======================
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
print("✅ Mixed Precision enabled:", mixed_precision.global_policy())

# ======================
# LOAD BASE MODEL
# ======================
model = load_model("models/acne_transfer_model_best.h5")

# ======================
# CREATE PSEUDO-LABELED DATASET
# ======================
os.makedirs(pseudo_labeled_dir, exist_ok=True)
for i in range(4):
    os.makedirs(os.path.join(pseudo_labeled_dir, f"level {i}"), exist_ok=True)

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

for img_name in os.listdir(unlabeled_dir):
    img_path = os.path.join(unlabeled_dir, img_name)
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    img = tf.keras.utils.load_img(img_path, target_size=IMAGE_SIZE)
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x, verbose=0)
    pred_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)
    if confidence > 0.85:
        shutil.copy(img_path, os.path.join(pseudo_labeled_dir, f"level {pred_class}", img_name))

print("✅ Pseudo-labeling complete.")

# ======================
# TRAINING FUNCTION (WITH FALLBACK)
# ======================
def run_training(batch_size):
    print(f"\n🚀 Starting training with batch_size={batch_size} ...")

    # ----------------------
    # Load datasets using tf.data
    # ----------------------
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="categorical"
    )

    pseudo_dataset = tf.keras.utils.image_dataset_from_directory(
        pseudo_labeled_dir,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="categorical"
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="categorical"
    )

    # ----------------------
    # Combine train + pseudo, optimize pipeline
    # ----------------------
    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.concatenate(pseudo_dataset)
    train_dataset = (
        train_dataset
        .map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y), num_parallel_calls=AUTOTUNE)
        .shuffle(1000)
        .prefetch(buffer_size=AUTOTUNE)
    )

    val_dataset = (
        val_dataset
        .map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y), num_parallel_calls=AUTOTUNE)
        .prefetch(buffer_size=AUTOTUNE)
    )

    # ----------------------
    # Fine-tune model
    # ----------------------
    base_model = model.layers[0]
    base_model.trainable = True

    optimizer = mixed_precision.LossScaleOptimizer(
        tf.keras.optimizers.Adam(learning_rate=1e-5)
    )

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(filepath='models/acne_transfer_model_finetuned.h5',
                                           save_best_only=True, monitor='val_accuracy', mode='max')
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    return history


# ======================
# RUN TRAINING WITH AUTO-FALLBACK
# ======================
try:
    history = run_training(BATCH_SIZE)
except tf.errors.ResourceExhaustedError:
    print("⚠️ VRAM exhausted! Retrying with smaller batch size...")
    history = run_training(BATCH_SIZE // 2)

# ======================
# PLOT RESULTS
# ======================
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, 'b-o', label='Train Accuracy')
plt.plot(epochs_range, val_acc, 'r-o', label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, 'b-o', label='Train Loss')
plt.plot(epochs_range, val_loss, 'r-o', label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.grid(True)

plt.tight_layout()
plt.savefig('outputs/training_performance_finetuned.png')
plt.show()
