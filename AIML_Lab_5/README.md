# AIML Lab 5 - Clinical Text Processing with Chunking and NER

An NLP pipeline that performs tokenization, chunking and Named Entity
Recognition on unstructured clinical notes, and turns them into structured
clinical information.

## Dataset

12 short synthetic clinical notes in `clinical_notes.csv`. No real patient data
is used, and no names, phone numbers, addresses or hospital IDs appear in the
notes.

## Files

| File | Description |
|------|-------------|
| `clinical_nlp_ner.ipynb` | The lab notebook, Parts A to F (with all outputs) |
| `clinical_notes.csv` | 12 synthetic clinical notes |

## What each part does

| Part | Contents |
|------|----------|
| **A** | Load the notes, count records, select one note, tokenize and display tokens |
| **B** | Noun phrase chunking with root, POS, dependency and head, plus a filter for clinically meaningful phrases |
| **C** | NER with the general spaCy model, then a clinical entity layer, with entity text, label, start and end, and a displaCy visualization |
| **D** | Combined pipeline that returns tokens, chunks and entities as a structured table, run over the full dataset |
| **E** | Evaluation against 5 hand-annotated test notes, with precision, recall and F1 overall and per entity type |
| **F** | Discharge summary extraction mapped to structured EHR fields |

## Entity types

`DISEASE`, `SYMPTOM`, `MEDICATION`, `DOSAGE`, `INVESTIGATION`, `ANATOMY`,
`PROCEDURE`, `FREQUENCY`.

## Why a rule-based entity layer was added

The lab sheet warns that a general model may not produce healthcare categories.
That is exactly what happens. Running `en_core_web_sm` unchanged on
"Patient was diagnosed with pneumonia and prescribed Azithromycin 500 mg.":

```
Azithromycin  -> PERSON
```

It labels the drug as a person, misses `pneumonia` completely, and elsewhere
tags dosages as `QUANTITY` and durations as `DATE`. The notebook documents these
real labels rather than editing them by hand, then adds a spaCy `EntityRuler`
with a 74 term clinical vocabulary so that the healthcare categories can be
produced. In a real deployment this would be replaced by a clinical model such
as scispaCy or medspaCy.

## Results

Across the 12 notes the pipeline extracts 81 entities:

| Category | Count |
|---|---|
| Symptom | 24 |
| Investigation | 13 |
| Disease | 12 |
| Medication | 9 |
| Dosage | 9 |
| Frequency | 9 |
| Procedure | 3 |
| Anatomy | 2 |

Evaluation on 5 hand-annotated test notes (23 annotated entities):

| Metric | Value |
|--------|-------|
| Correctly extracted | 19 |
| Total extracted | 20 |
| Total actual | 23 |
| **Precision** | **0.950** |
| **Recall** | **0.826** |
| **F1-Score** | **0.884** |

The test notes deliberately include terms that are not in the vocabulary, so the
score reflects real behaviour. The four errors are:

| Entity | Type | Result |
|---|---|---|
| amoxicillin | MEDICATION | missed, not in the vocabulary |
| ct scan | INVESTIGATION | missed, not in the vocabulary |
| acute appendicitis | DISEASE | missed, not in the vocabulary |
| severe abdominal pain | SYMPTOM | matched only as "abdominal pain", wrong boundary |

This is the known limitation of a rule-based approach: it cannot generalise to
drug or disease names it has never seen.

## Run

```bash
pip install pandas spacy
python -m spacy download en_core_web_sm
jupyter notebook clinical_nlp_ner.ipynb
```
