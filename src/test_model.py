import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

model = tf.keras.models.load_model("mobilenet_best_model.h5")

dataset_path = r"E:\final project\dataset_final_split"
test_dir = os.path.join(dataset_path, "test")

test_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_generator = test_datagen.flow_from_directory(
    test_dir, 
    target_size=(224, 224), 
    batch_size=32, 
    class_mode="categorical", 
    shuffle=False
)

test_loss, test_acc = model.evaluate(test_generator, verbose=1)
print(f"\n Test Accuracy: {test_acc*100:.2f}%")

y_pred = model.predict(test_generator)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

cm = confusion_matrix(y_true, y_pred_labels)
print("\n Confusion Matrix:")
print(cm)

print("\n Classification Report:")
print(classification_report(y_true, y_pred_labels, target_names=class_labels))

plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks, class_labels, rotation=90, fontsize=8)
plt.yticks(tick_marks, class_labels, fontsize=8)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
