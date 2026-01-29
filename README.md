@"
# ML-Based Address Matching with Geospatial Features

## Overview

During disaster recovery events and planned network outages, it becomes critical to quickly and accurately connect **vendor-provided data** with **internal site inventory data** in order to reduce downtime and accelerate restoration efforts.

In practice, these datasets are often maintained separately, use different identifiers, and vary in data quality—making direct joins unreliable.

This project provides a **machine learning–driven approach** to identifying and linking vendor premises to site inventory records when exact matches do not exist.

---

## What This Notebook Does

This notebook trains a **scikit-learn** model to match **vendor premises** to **site inventory** records using a combination of:

- Address text similarity features (street, city, state)
- House number differences
- Optional fuzzy string matching scores
- **Geographic distance features (miles)** derived from **latitude and longitude (LAT/LON)**
- A public geocoding service (**US Census Geocoder**) to generate **vendor LAT/LON**
  - Site inventory records are assumed to already include LAT/LON

The output is a scored set of candidate matches, allowing the best match per vendor location to be selected based on model probability.

---

## Approach

At a high level, the workflow is:

1. **Normalize and clean address fields**
2. **Geocode vendor addresses** using a public API (with local caching)
3. **Generate candidate pairs** using blocking and similarity-based pruning
4. **Engineer features**, including text similarity and geospatial distance
5. **Train a supervised ML model** (logistic regression via scikit-learn)
6. **Score and rank matches** to identify the most likely site per vendor record

---

## Machine Learning Model

- Framework: **scikit-learn**
- Model: **Logistic Regression**
- Features include both **text-based similarity metrics** and **geographic distance**
- Missing values handled via imputation
- Feature scaling applied prior to training

This design keeps the model **interpretable**, explainable, and easy to retrain as new labeled data becomes available.

---

## Configuration

Column names are intentionally abstracted to keep the notebook reusable.

You must map your dataset columns in the `COLUMN_MAP` section of the notebook before running:

```python
COLUMN_MAP = {
    "site": {...},
    "vendor": {...}
}
