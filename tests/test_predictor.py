import unittest
from pathlib import Path

from predictor import LogisticRiskModel, StandardScaler, accuracy, load_csv, train_and_predict, StudentRecord

DATA = Path(__file__).parents[1] / "data" / "student_performance.csv"


class PredictorTests(unittest.TestCase):
    def test_csv_loads_expected_rows(self):
        rows, labels = load_csv(DATA)
        self.assertEqual(len(rows), 24)
        self.assertEqual(len(labels), 24)

    def test_scaler_centers_training_data(self):
        scaler = StandardScaler().fit([[1.0, 10.0], [3.0, 20.0]])
        transformed = scaler.transform([[1.0, 10.0], [3.0, 20.0]])
        self.assertAlmostEqual(sum(row[0] for row in transformed), 0.0)
        self.assertAlmostEqual(sum(row[1] for row in transformed), 0.0)

    def test_model_separates_high_and_low_risk_examples(self):
        rows, labels = load_csv(DATA)
        scaler = StandardScaler().fit(rows)
        model = LogisticRiskModel().fit(scaler.transform(rows), labels)
        predictions = model.predict(scaler.transform(rows))
        self.assertGreaterEqual(accuracy(labels, predictions), 0.90)

    def test_high_support_student_is_lower_risk(self):
        probability, risk, train_accuracy = train_and_predict(DATA, StudentRecord(96, 14, 10, 90))
        self.assertEqual(risk, 0)
        self.assertLess(probability, 0.5)
        self.assertGreaterEqual(train_accuracy, 0.90)

    def test_low_support_student_is_at_risk(self):
        probability, risk, _ = train_and_predict(DATA, StudentRecord(55, 2, 2, 40))
        self.assertEqual(risk, 1)
        self.assertGreater(probability, 0.5)


if __name__ == "__main__":
    unittest.main()
