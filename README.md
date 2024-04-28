# Handwritten Letter Recognition Model

## Description:

This repository contains code for training and testing a Convolutional Neural Network (CNN) model to recognize handwritten letters. The model is trained on a combination of the EMNIST dataset and a custom dataset containing capital letters from A to Z. The goal is to accurately classify handwritten letters into their respective categories.

## Files:

- **sample_images:** This folder contains sample images of handwritten letters used to train and test the model.

- **image_to_pixel.ipynb:** This Jupyter Notebook contains the code to preprocess the handwritten data, convert it into numerical format, and store it in a CSV file named `capital_samples.csv`.

- **tensorflow_model.ipynb:** This Jupyter Notebook contains the final CNN model, including code for training and testing. The notebook is well-documented with comments for clarity.

## Training Process:

1. **EMNIST Dataset:** Initially, the CNN model is trained on the popular EMNIST dataset, which contains a large number of handwritten letters and digits. This serves as a baseline for the model's performance.

2. **Custom Dataset:** The model is further trained on a custom dataset, which includes capital letters from A to Z. This dataset is augmented with additional images to improve model generalization.

## Future Improvements:

- **Word Dataset:** The model can be further improved by training it on words of larger lengths. This would enable it to recognize complete words and phrases more accurately.

## Dependencies:

- Python 3.x
- TensorFlow/Keras
- NumPy
- Matplotlib
- scikit-learn

## Usage:

1. Clone the repository to your local machine.
2. Run the `image_to_pixel.ipynb` notebook to preprocess the handwritten data and generate the `capital_samples.csv` file.
3. Open and run the `tensorflow_model.ipynb` notebook to train and test the CNN model.

## Contributing:

Contributions to this project are welcome! Please feel free to submit pull requests or open issues if you encounter any bugs or have suggestions for improvements.
