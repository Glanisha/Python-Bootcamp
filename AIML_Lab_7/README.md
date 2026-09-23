# AIML Lab 7 - Medical Reviews Analysis from Social Media Data

Sentiment classification of patient medical reviews using TF-IDF features with
Logistic Regression and Multinomial Naive Bayes.

## Dataset

Real patient reviews from **Drugs.com**. The 1 to 10 star rating that comes with
each review is mapped to a sentiment label:

| Rating | Sentiment |
|---|---|
| 8 to 10 | Positive |
| 5 to 7 | Neutral |
| 1 to 4 | Negative |

`prepare_dataset.py` downloads the reviews and writes a balanced sample of 1200
per class to `medical_reviews.csv` (3600 reviews). The 32 MB raw download is
git-ignored.

## Files

| File | Description |
|------|-------------|
| `medical_review_sentiment.ipynb` | The lab notebook, Parts A to H (with all outputs) |
| `medical_reviews.csv` | 3600 labelled reviews |
| `prepare_dataset.py` | Downloads the source data and builds the CSV |

## What each part does

| Part | Contents |
|------|----------|
| **A** | Load the reviews, counts, missing values, sentiment distribution, rating histogram |
| **B** | HTML unescape, lowercase, remove URLs, mentions, hashtags, punctuation, tokenize, remove stopwords |
| **C** | 80/20 stratified split, then TF-IDF with 5000 features and unigrams plus bigrams, fitted on the training data only |
| **D** | 5-fold stratified cross-validation of both models, selection by mean F1 |
| **E** | Sentiment prediction for a new review with class probabilities |
| **F** | Accuracy, precision, recall, F1, confusion matrix, per class recall and interpretation |
| **G** | Top 10 words per sentiment from the Logistic Regression coefficients |
| **H** | Batch feedback analysis with an escalation flag |

## Results

Cross-validation on the training set:

| Model | Mean CV Accuracy | Mean CV F1 |
|---|---|---|
| Logistic Regression | 0.5747 | 0.5736 |
| Multinomial Naive Bayes | 0.5642 | 0.5637 |

Selected model: Logistic Regression.

Test set (720 reviews):

| Metric | Value |
|--------|-------|
| Accuracy | 0.5556 |
| Precision (macro) | 0.5508 |
| Recall (macro) | 0.5556 |
| F1-Score (macro) | 0.5521 |

Per class recall:

| Class | Recall |
|---|---|
| Negative | 0.6292 |
| Positive | 0.6292 |
| Neutral | 0.4083 |

Confusion matrix:

```
                  Predicted
                  Negative  Neutral  Positive
Actual Negative        151       57        32
       Neutral          79       98        63
       Positive         33       56       151
```

55.6% against a 33% chance level for three balanced classes. Neutral is the
weakest class because a 5 to 7 star review usually mixes praise with complaints,
so its vocabulary overlaps both of the other classes. Confusing Negative
directly with Positive is rare (33 and 32 cases), so the two ends of the scale
are separated reasonably well.

Top words learned by the model:

| Sentiment | Top words |
|---|---|
| Positive | love, years, works, best, without, life, miracle, great |
| Negative | worse, not, never, horrible, worst, nothing, not recommend, stop |
| Neutral | however, also, mg, feel, trouble, not sure, problem |

## Note on domain

The model is trained on medication reviews. Part H includes two hospital service
reviews, and its confidence on those drops to around 0.4 because words such as
OPD and nursing staff never appear in training. A deployed system would need to
be retrained on the kind of feedback it actually receives.

## Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk
python prepare_dataset.py
jupyter notebook medical_review_sentiment.ipynb
```
