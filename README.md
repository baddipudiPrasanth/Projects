# Sentiment Analysis using NLP

This repository contains a Jupyter Notebook that demonstrates a basic sentiment analysis model using Natural Language Processing (NLP) techniques. The project was developed in Google Colab and leverages various Python libraries to preprocess text data and build a machine learning model for sentiment classification.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model](#model)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Sentiment analysis is a method of identifying and categorizing opinions expressed in a piece of text to determine whether the sentiment conveyed is positive, negative, or neutral. This project applies various NLP techniques to preprocess text data and uses a logistic regression model to classify the sentiment.

## Features

- Text preprocessing using tokenization, stemming, and stopword removal.
- Feature extraction using TF-IDF vectorization.
- Logistic regression for binary sentiment classification.
- Model evaluation using accuracy score.

## Installation

To run the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/baddipudiPrasanth/Sentiment_Analysis.ipynb.git
   cd Sentiment_Analysis.ipynb
   ```

2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Since the project was developed in Google Colab, you can easily open and run the notebook online. Alternatively, you can download the notebook and run it locally using Jupyter:

1. Open the `Sentiment_Analysis.ipynb` notebook in Jupyter or Google Colab:
   ```bash
   jupyter notebook Sentiment_Analysis.ipynb
   ```

2. Execute the cells in the notebook to preprocess the data, train the model, and evaluate its performance.

## Dataset

The notebook expects a dataset in CSV format with a column containing the text data and a corresponding column with sentiment labels (e.g., positive, negative). Ensure that the dataset is located in the appropriate directory or update the path within the notebook.

## Preprocessing

Key preprocessing steps include:

- **Text Cleaning:** Removal of special characters, numbers, and irrelevant elements using regular expressions.
- **Stopword Removal:** Filtering out commonly used words that do not add significant meaning.
- **Stemming:** Reducing words to their base form using `PorterStemmer` from the NLTK library.

## Model

The model is built using:

- **TF-IDF Vectorization:** This converts the textual data into a numerical format suitable for model training.
- **Logistic Regression:** A robust algorithm used for binary classification tasks.

## Evaluation

The model's performance is evaluated using the accuracy score, a metric calculated on the test dataset.

## Contributing

Contributions are welcome! Feel free to fork the repository and submit a pull request with any improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---
