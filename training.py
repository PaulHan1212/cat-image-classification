import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.utils import load_img, img_to_array
from pathlib import Path
import os


from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import RandomFlip, RandomRotation
BATCH_SIZE = 4
IMG_SIZE = (224, 224)
directory = str(Path.home() / "Desktop" / "cat")
train_dataset, remaining_dataset = image_dataset_from_directory(
    directory,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    validation_split=0.30,
    subset="both",
    seed=42
)

remaining_dataset = remaining_dataset.unbatch()
val_size = int(remaining_dataset.cardinality().numpy()) // 2
validation_dataset = remaining_dataset.take(val_size).batch(BATCH_SIZE)
testing_dataset = remaining_dataset.skip(val_size).batch(BATCH_SIZE)

print(train_dataset.class_names)
for images, labels in train_dataset.take(1):
    print("image shape：", images.shape)  # 應該是 (4, 160, 160, 3)
    print("label：", labels.numpy())

def data_augmenter():
    data_augmentation = tf.keras.Sequential()
    data_augmentation.add(RandomFlip('horizontal'))
    data_augmentation.add(RandomRotation(0.2))
    return data_augmentation

data_augmentation = data_augmenter()

class_names = train_dataset.class_names
print("type：", class_names)

class_first_images = {}

for images, labels in train_dataset:
    for i in range(len(labels)):
        label = labels[i].numpy()
        if class_names[label] not in class_first_images:
            class_first_images[class_names[label]] = images[i]
    if len(class_first_images) == len(class_names):
        break

for class_name, image_tensor in class_first_images.items():
    plt.figure(figsize=(10, 10))
    plt.suptitle(f"{class_name} preview", fontsize=16)
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(image_tensor, 0))
        plt.imshow(augmented_image[0].numpy().astype("uint8"))
        plt.axis("off")
    plt.show()

def build_transfer_model(img_size=(200, 200), num_classes=1):
    input_shape = img_size + (3,)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmenter()(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_transfer_model(IMG_SIZE)
model.summary()

def visualize_model():
    layers_info = [
        ("Input Image", "200 x 200 x 3"),
        ("Data Augmentation", "Random Flip + Rotation"),
        ("Preprocessing", "Normalize to [-1, 1]"),
        ("MobileNetV2", "Pretrained CNN | Frozen"),
        ("Global Average Pooling", "1280 Features"),
        ("Dropout", "Rate = 0.2"),
        ("Dense + Sigmoid", "Dottie / Yaun")
    ]

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis("off")

    colors = [
        "#DBEAFE",
        "#D1FAE5",
        "#FEF3C7",
        "#C7D2FE",
        "#E0E7FF",
        "#FCE7F3",
        "#BBF7D0"
    ]

    for i, (title, subtitle) in enumerate(layers_info):
        y = 14 - i * 2.1

        box = FancyBboxPatch(
            (2, y),
            6, 1.3,
            boxstyle="round,pad=0.15",
            facecolor=colors[i],
            edgecolor="#334155",
            linewidth=1.5
        )
        ax.add_patch(box)

        ax.text(
            5, y + 0.85, title,
            ha="center", va="center",
            fontsize=14, fontweight="bold"
        )

        ax.text(
            5, y + 0.40, subtitle,
            ha="center", va="center",
            fontsize=10, color="#475569"
        )

        if i < len(layers_info) - 1:
            arrow = FancyArrowPatch(
                (5, y - 0.1),
                (5, y - 0.65),
                arrowstyle="-|>",
                mutation_scale=18,
                linewidth=1.5,
                color="#475569"
            )
            ax.add_patch(arrow)

    ax.set_title(
        "Cat Image Classification - Model Architecture",
        fontsize=18,
        fontweight="bold",
        pad=20
    )

    plt.savefig(
        "model_architecture.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

visualize_model()



history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)


from pathlib import Path
import matplotlib.pyplot as plt

# Save charts in the project folder
output_dir = Path(__file__).resolve().parent / "results"
output_dir.mkdir(exist_ok=True)

epochs = range(1, len(history.history["accuracy"]) + 1)

# 1. Accuracy Graph
plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    history.history["accuracy"],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs,
    history.history["val_accuracy"],
    marker="o",
    label="Validation Accuracy"
)

plt.title("Training vs. Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xticks(epochs)
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()

plt.savefig(
    output_dir / "accuracy.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


# 2. Loss Graph
plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    history.history["loss"],
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    history.history["val_loss"],
    marker="o",
    label="Validation Loss"
)

plt.title("Training vs. Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(epochs)
plt.grid(True, alpha=0.3)
plt.legend()

plt.savefig(
    output_dir / "loss.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

print(f"Charts saved to: {output_dir}")


def predict_image(img_path, model, class_names, threshold=0.6):
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)

    # Model already handles MobileNetV2 preprocessing
    img_array = np.expand_dims(img_array, axis=0)

    pred = float(model.predict(img_array, verbose=0)[0][0])

    if pred >= threshold:
        return f"{class_names[1]} (score={pred:.2f})"

    elif pred <= (1 - threshold):
        return f"{class_names[0]} (score={1 - pred:.2f})"

    else:
        return (
            f"Uncertain prediction: "
            f"{class_names[0]}={1-pred:.2f}, "
            f"{class_names[1]}={pred:.2f}"
        )

test_yuan = directory + "/test_yuan.jpeg"
test_dottie = directory + "/test_dottie.jpeg"

result_yuan = predict_image(
    img_path=test_yuan,
    model=model,
    class_names=class_names,
    threshold=0.6
)

result_dottie = predict_image(
    img_path=test_dottie,
    model=model,
    class_names=class_names,
    threshold=0.6
)

print("Prediction for Yuan:", result_yuan)
print("Prediction for Dottie:", result_dottie)

