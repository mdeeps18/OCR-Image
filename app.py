import os
import cv2
import pytesseract
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"https://github.com/arunkumarramanan/mit-deep-learning/raw/master/"

# Define a function to extract text from an image using OCR
def extract_text_from_image('https://github.com/arunkumarramanan/mit-deep-learning/raw/master/'):
    image = cv2.imread('https://github.com/arunkumarramanan/mit-deep-learning/raw/master/')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray_image, config='--psm 6')
    return text.strip()

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
