"""Student performance risk prediction with a dependency-free logistic model."""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

FEATURES = ("attendance", "study_hours", "assignments_completed", "prior_score")


@dataclass(frozen=True)
class StudentRecord:
    attendance: float
    study_hours: float
    assignments_completed: float
    prior_score: float
    at_risk: int | None = None

    def vector(self) -> list[float]:
        return [self.attendance, self.study_hours, self.assignments_completed, self.prior_score]


class StandardScaler:
    def fit(self, rows: Sequence[Sequence[float]]) -> "StandardScaler":
        if not rows:
            raise ValueError("At least one row is required")
        width = len(rows[0])
        self.means = [sum(row[i] for row in rows) / len(rows) for i in range(width)]
        self.stds = []
        for i in range(width):
            variance = sum((row[i] - self.means[i]) ** 2 for row in rows) / len(rows)
            self.stds.append(math.sqrt(variance) or 1.0)
        return self

    def transform(self, rows: Sequence[Sequence[float]]) -> list[list[float]]:
        return [[(value - mean) / std for value, mean, std in zip(row, self.means, self.stds)] for row in rows]


class LogisticRiskModel:
    def __init__(self, learning_rate: float = 0.08, epochs: int = 1200) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs

    @staticmethod
    def _sigmoid(value: float) -> float:
        value = max(-40.0, min(40.0, value))
        return 1.0 / (1.0 + math.exp(-value))

    def fit(self, rows: Sequence[Sequence[float]], labels: Sequence[int]) -> "LogisticRiskModel":
        if len(rows) != len(labels) or not rows:
            raise ValueError("Rows and labels must be non-empty and have equal length")
        self.weights = [0.0] * (len(rows[0]) + 1)
        for _ in range(self.epochs):
            gradients = [0.0] * len(self.weights)
            for row, label in zip(rows, labels):
                probability = self._sigmoid(self.weights[0] + sum(w * x for w, x in zip(self.weights[1:], row)))
                error = probability - label
                gradients[0] += error
                for i, value in enumerate(row, start=1):
                    gradients[i] += error * value
            for i, gradient in enumerate(gradients):
                self.weights[i] -= self.learning_rate * gradient / len(rows)
        return self

    def predict_proba(self, rows: Sequence[Sequence[float]]) -> list[float]:
        return [self._sigmoid(self.weights[0] + sum(w * x for w, x in zip(self.weights[1:], row))) for row in rows]

    def predict(self, rows: Sequence[Sequence[float]], threshold: float = 0.5) -> list[int]:
        return [int(probability >= threshold) for probability in self.predict_proba(rows)]


def load_csv(path: str | Path) -> tuple[list[list[float]], list[int]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows, labels = [], []
        for item in reader:
            rows.append([float(item[name]) for name in FEATURES])
            labels.append(int(item["at_risk"]))
    return rows, labels


def accuracy(actual: Iterable[int], predicted: Iterable[int]) -> float:
    actual, predicted = list(actual), list(predicted)
    return sum(a == p for a, p in zip(actual, predicted)) / len(actual)


def train_and_predict(data_path: str | Path, student: StudentRecord) -> tuple[float, int, float]:
    rows, labels = load_csv(data_path)
    scaler = StandardScaler().fit(rows)
    model = LogisticRiskModel().fit(scaler.transform(rows), labels)
    probability = model.predict_proba(scaler.transform([student.vector()]))[0]
    return probability, int(probability >= 0.5), accuracy(labels, model.predict(scaler.transform(rows)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict whether a student is at risk of underperforming.")
    parser.add_argument("--data", default="data/student_performance.csv")
    parser.add_argument("--attendance", type=float, required=True)
    parser.add_argument("--study-hours", type=float, required=True)
    parser.add_argument("--assignments-completed", type=float, required=True)
    parser.add_argument("--prior-score", type=float, required=True)
    args = parser.parse_args()
    probability, risk, train_accuracy = train_and_predict(args.data, StudentRecord(args.attendance, args.study_hours, args.assignments_completed, args.prior_score))
    print(f"Training accuracy: {train_accuracy:.1%}")
    print(f"Risk probability: {probability:.1%}")
    print("Prediction: AT RISK" if risk else "Prediction: LOWER RISK")
