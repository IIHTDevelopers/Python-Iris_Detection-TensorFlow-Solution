import unittest
import numpy as np
import tensorflow as tf
from main import *
from main import get_prediction_sample  
from tests.TestUtils import TestUtils


class TestIrisClassifierYaksha(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_obj = TestUtils()
        try:
            cls.X_train, cls.X_test, cls.y_train, cls.y_test, cls.scaler = load_and_preprocess()
            cls.model = build_model()
            cls.model = train_model(cls.model, cls.X_train, cls.y_train)
        except Exception as e:
            print("Setup failed:", e)
            cls.X_train = cls.X_test = cls.y_train = cls.y_test = cls.scaler = cls.model = None

    def test_data_shape(self):
        try:
            result = (
                isinstance(self.X_train, np.ndarray) and
                self.X_train.shape[1] == 4 and
                len(self.X_train) > 0 and
                len(self.X_test) > 0 and
                len(self.y_train) > 0
            )
            self.test_obj.yakshaAssert("TestDataShape", result, "functional")
            print("TestDataShape =", "Passed" if result else "Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestDataShape", False, "functional")
            print("TestDataShape = Failed | Exception:", e)

    def test_model_structure(self):
        try:
            result = (
                self.model is not None and
                len(self.model.layers) == 2 and
                isinstance(self.model.layers[0], tf.keras.layers.Dense)
            )
            self.test_obj.yakshaAssert("TestModelStructure", result, "functional")
            print("TestModelStructure =", "Passed" if result else "Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestModelStructure", False, "functional")
            print("TestModelStructure = Failed | Exception:", e)

    def test_model_accuracy(self):
        try:
            _, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
            result = accuracy > 0.9
            self.test_obj.yakshaAssert("TestModelAccuracy", result, "functional")
            print("TestModelAccuracy =", "Passed" if result else "Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestModelAccuracy", False, "functional")
            print("TestModelAccuracy = Failed | Exception:", e)

    def test_prediction_sample_correctness_and_result(self):
        try:
            expected_sample = np.array([5.1, 3.5, 1.4, 0.2])
            actual_sample = get_prediction_sample()

            # ✅ Ensure correct shape: (4,) -> (1, 4)
            actual_sample = np.array(actual_sample).reshape(1, -1)

            # ✅ Check if sample used in main matches expected one
            sample_match = np.allclose(actual_sample[0], expected_sample)

            # ✅ Scale and predict
            sample_scaled = self.scaler.transform(actual_sample)
            prediction = self.model.predict(sample_scaled)
            is_setosa = prediction[0][0] > 0.5

            result = sample_match and is_setosa
            self.test_obj.yakshaAssert("TestPredictionSampleCorrectAndPrediction", result, "functional")
            print("TestPredictionSampleCorrectAndPrediction =", "Passed" if result else "Failed")

        except Exception as e:
            self.test_obj.yakshaAssert("TestPredictionSampleCorrectAndPrediction", False, "functional")
            print("TestPredictionSampleCorrectAndPrediction = Failed | Exception:", e)
