# 📰 Fake News Detection with Naive Bayes

This project uses Natural Language Processing (NLP) and a Naive Bayes classifier to distinguish between real and fake news articles. It applies text cleaning, TF-IDF feature extraction, and model evaluation to achieve strong performance on a publicly available dataset.

---

## 📂 Dataset

The dataset was sourced from Kaggle:  
[Fake and Real News Dataset by Clément Bisaillon](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

It includes:
- `True.csv` – real news articles  
- `Fake.csv` – fake news articles

---

## Project Workflow

1. **Data Loading & Labelling**
   - Combined real and fake news datasets with binary labels (`True` or `False`)
   - Shuffled and prepared the dataset for analysis

2. **Preprocessing**
   - Cleaned the text (punctuation removal, lowercasing)
   - Visualised label distribution
   - Extracted top keywords from each class using `CountVectorizer`

3. **Feature Extraction**
   - Used `TfidfVectorizer` with stop word removal and 1000 features

4. **Model Training**
   - Trained a `MultinomialNB` model on 80% of the data
   - Evaluated on the remaining 20%

5. **Model Evaluation**
   - Achieved **93% accuracy**
   - Precision, recall, and F1-scores all around **0.93** for both classes

---

## Future Improvements

- Experiment with more advanced models like Logistic Regression, SVM, or BiLSTM
- Incorporate bigram/trigram features for richer context
- Build a web app (e.g. Streamlit or Flask) for interactive news classification
- Perform hyperparameter tuning and cross-validation for further optimization
