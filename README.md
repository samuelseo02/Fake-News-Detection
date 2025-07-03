#  Fake News Detection with Naive Bayes

This project applies Natural Language Processing (NLP) and a Naive Bayes classifier to detect fake news articles using a labeled dataset. Completed as part of the subject *41040 Introduction to Artificial Intelligence* (November 2024).

---

## 🔧 My Contributions
- Independently built and trained the model
- Preprocessed over 45,000 articles using custom text cleaning
- Extracted features using TF-IDF vectorization
- Visualized label distribution and keyword frequency
- Tuned model and addressed overfitting with controlled feature selection

---

##  Dataset
- Source: [Fake and Real News Dataset – Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Includes two CSV files: `True.csv` (real news) and `Fake.csv` (fake news)
- Combined, labeled, and shuffled into a unified dataframe
- Cleaned using regex, lowercasing, and punctuation removal

---

##  Evaluation & Environment
- Model: `MultinomialNB` from scikit-learn
- Feature Extraction: `TfidfVectorizer` with 1000 max features
- Accuracy: **93%**
- F1-score: **0.93** for both classes
- Tools: Google Colab Pro with GPU (Tesla T4)
- Evaluation: `classification_report`, `accuracy_score`

---

##  Future Improvements
- Try more advanced models (Logistic Regression, SVM, LSTM)
- Add ROC/AUC metrics and cross-validation
- Expand dataset with more sources or languages
- Deploy with Flask or Streamlit for interactive classification
- Add model explanation with SHAP or LIME
