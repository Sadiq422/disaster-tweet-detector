# DisasterTweetDetector — NLP-Klassifikation von Katastrophen-Tweets

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=flat)
![NLTK](https://img.shields.io/badge/NLP-NLTK-154f5b?style=flat)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

Ein End-to-End-NLP-Projekt, das automatisch erkennt, ob ein Tweet von einer **echten Katastrophe** handelt oder nicht. Basierend auf dem bekannten Kaggle-Datensatz „Natural Language Processing with Disaster Tweets".

---

## 🎯 Problemstellung

In Krisen werden Twitter/X-Meldungen zur Echtzeit-Informationsquelle. Die Herausforderung: Sprache ist mehrdeutig („on fire" kann wörtlich oder bildlich gemeint sein). Ziel ist ein Modell, das echte Katastrophenmeldungen zuverlässig von harmlosen Tweets trennt, mit Fokus auf wenige Fehlalarme.

## 🗂️ Daten

Kaggle Disaster Tweets: rund **7.600 gelabelte Tweets** (`train.csv`) mit den Feldern `id`, `keyword`, `location`, `text`, `target` (1 = Katastrophe, 0 = keine) sowie `test.csv` und `sample_submission.csv`.

## 🧪 Vorgehen (Pipeline)

1. **Textbereinigung:** Entfernen von URLs, Mentions, Sonderzeichen; Kleinschreibung.
2. **Preprocessing:** Tokenisierung, Stopword-Entfernung, **Lemmatisierung**.
3. **Vektorisierung:** TF-IDF (und CountVectorizer zum Vergleich).
4. **Modellvergleich:** Naive Bayes (MultinomialNB), Logistic Regression, Support Vector Machine (SVC), Random Forest und **XGBoost**.
5. **Evaluation:** Accuracy, Precision, Recall und F1-Score; Optimierung mit Blick auf Precision (Reduktion von False Positives).

## 📈 Ergebnisse

- Bestes Modell: **ca. 81 % Accuracy** und ein **F1-Score von ~0,80** (gewichtet) auf den Validierungsdaten.
- Klassischer ML-Ansatz (TF-IDF + lineare/Boosting-Modelle) erwies sich als stark und ressourcenschonend gegenüber dem Aufwand.

*(Die genauen Werte je Modell stehen im Notebook im Klassifikationsbericht.)*

## 🛠️ Ausführung

```bash
pip install pandas numpy scikit-learn xgboost nltk matplotlib seaborn
jupyter notebook DisasterTweetDetector.ipynb
```

Lege `train.csv`, `test.csv` und `sample_submission.csv` in den Projektordner (Quelle: Kaggle-Wettbewerb „nlp-getting-started").

## 🗃️ Projektstruktur

```
DisasterTweetDetector.ipynb   # End-to-End-Notebook (EDA, Preprocessing, Modelle, Evaluation)
train.csv / test.csv          # Kaggle-Datensatz
sample_submission.csv         # Vorlage für die Vorhersage-Abgabe
```

## 🧰 Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, NLTK, TF-IDF, Matplotlib/Seaborn.

## 👤 Autor

**Sadiq Qais** — Junior Data Scientist & Data Analyst
[LinkedIn](https://www.linkedin.com/in/sadiq-qais) · qais.sadiq422@gmail.com

## 📄 Lizenz

MIT
