# Student Performance Risk Predictor

A transparent, dependency-free machine-learning mini-project that estimates whether a student may be at risk of underperforming from attendance, study time, assignment completion, and prior academic score.

## Overview

Early identification of academic risk can help educators offer support before a student falls further behind. This project demonstrates an end-to-end binary classification workflow using a small, reproducible dataset and logistic regression implemented with Python's standard library.

## Problem statement

Given four student-support indicators, predict whether a student belongs to the `at_risk` class. The result is a probability rather than a definitive judgment and should be used for supportive intervention, not as a high-stakes decision.

## Features

- Standardizes numeric features before training.
- Trains a logistic regression classifier with gradient descent.
- Prints a risk probability, class prediction, and training accuracy.
- Includes deterministic sample data and unit tests.
- Uses no third-party runtime dependencies.

## Technology stack

- Python 3.10+
- Logistic regression and standardization implemented from scratch
- `unittest` for automated tests
- CSV for lightweight sample data

## Installation

```bash
git clone https://github.com/mohan843/student-performance-risk-predictor.git
cd student-performance-risk-predictor
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

`requirements.txt` is intentionally empty of third-party packages because the project uses only the Python standard library.

## Execution

Predict a lower-risk example:

```bash
python predictor.py --attendance 96 --study-hours 14 --assignments-completed 10 --prior-score 90
```

Predict a higher-risk example:

```bash
python predictor.py --attendance 55 --study-hours 2 --assignments-completed 2 --prior-score 40
```

## Testing

Run the test suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

## Sample output

```text
Training accuracy: 100.0%
Risk probability: 0.1%
Prediction: LOWER RISK
```

The exact probability can vary if the training data or model hyperparameters are changed.

## Ethical note

This is an educational demonstration using a tiny synthetic dataset. It is not validated for real educational decisions, and predictions should never replace human review or be used to label students unfairly.

## Future improvements

- Evaluate on a held-out test split and report precision, recall, and F1 score.
- Add calibration and explainability for individual predictions.
- Replace synthetic data with a carefully governed, representative dataset.
- Add a small Streamlit interface and model monitoring for drift.
