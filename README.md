# sentiment-review-movie

Course Project: Sentiment Analysis  
Author:  
University:

---

## Project Overview

This project aims to classify movie reviews from the IMDB dataset into two sentiment classes: positive and negative. It uses a combination of BERT for feature extraction and a Support Vector Machine (SVM) for classification.

An interactive web application is also developed using Streamlit to demonstrate the sentiment prediction in real time.

---

## Methodology

1. **Data Preprocessing**
   - Load and merge reviews labeled as positive and negative from the IMDB dataset
   - Clean the text data (lowercasing, removing punctuation, etc.)

2. **Feature Extraction**
   - Utilize BERT (bert-base-uncased) to convert text into numerical vectors
   - Use the [CLS] token representation for each review

3. **Model Training**
   - Train a Support Vector Machine (SVM) on the extracted features
   - Save the trained model in a .pkl file for deployment

4. **Deployment via Streamlit**
   - Build an interface using Streamlit
   - Load the trained model to perform sentiment classification from user input

---

## Folder Structure

```
sentiment-analysis-imdb/
├── notebook/
│   └── sentiment_analysis.ipynb    # Model training and evaluation
├── models/
│   └── svm_imdb_model.pkl          # Trained model file
├── app/
│   └── as_app.py                   # Streamlit interface script
├── requirements.txt                # Required Python packages
├── .gitignore                      # Git ignore file
└── README.md                       # Project documentation
```

---

## How to Run

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Run the Streamlit application
```bash
streamlit run app/as_app.py
```

---

## Outputs

- Real-time sentiment predictions from user input via Streamlit
- Output features such as visualizations or summary insights are still in progress and will be added in the next iteration

---

## Dataset

- Source: Large Movie Review Dataset v1.0 by Stanford  
  URL: https://ai.stanford.edu/~amaas/data/sentiment/
- Contains 50,000 reviews (25,000 for training, 25,000 for testing)

---

## Notes

- The model supports binary sentiment classification: positive or negative
- Extending the model to include neutral class would require additional data labeling

---

## Contributor

- Name:
