import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
import os

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()


# Normalize pixel values to be between 0 and 1
X_train, X_test = X_train / 255.0, X_test / 255.0

# Define a simple neural network for image classification
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
print("\nTest accuracy:", test_acc)

# Define a function to calculate accuracy
def calculate_accuracy(ground_truth, predicted):
    correct = sum(1 for gt, pred in zip(ground_truth, predicted) if gt == pred)
    return correct / len(ground_truth)

# Define a function to analyze misclassified digits
def analyze_misclassified(digits, ground_truth, predicted):
    misclassified = [i for i, (gt, pred) in enumerate(zip(ground_truth, predicted)) if gt != pred]
    misclassified_digits = [digits[i] for i in misclassified]
    misclassified_distribution = pd.Series(misclassified_digits).value_counts()

    plt.figure(figsize=(10, 5))
    plt.bar(misclassified_distribution.index, misclassified_distribution.values)
    plt.xlabel('Misclassified Digits')
    plt.ylabel('Frequency')
    plt.title('Distribution of Misclassified Digits')
    plt.xticks(rotation=0)
    plt.show()

# Main function
def main():
    # Set the path to your noisy images folder
    images_folder = 'noisy_digits/'

    # List of noisy images
    image_files = [os.path.join(images_folder, file) for file in os.listdir(images_folder)]

    # Ground truth labels (you may need to specify these)
    ground_truth_labels = ['3', '7', '2', '5', '8', '1', '0', '4', '9', '6']

    # Initialize lists to store ground truth and predicted labels
    ground_truth = []
    predicted_labels = []

    for image_file in image_files:
        extracted_text = extract_text_from_image(image_file)
        predicted_labels.append(extracted_text)
        image_filename = os.path.basename(image_file)
        ground_truth.append(image_filename.split('_')[0])

    # Calculate accuracy
    accuracy = calculate_accuracy(ground_truth, predicted_labels)
    print(f'OCR Accuracy: {accuracy * 100:.2f}%')

    # Analyze misclassified digits
    analyze_misclassified(ground_truth_labels, ground_truth, predicted_labels)

if __name__ == "__main__":
    main()

