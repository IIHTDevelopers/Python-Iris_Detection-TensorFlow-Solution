import unittest
import numpy as np
from main import load_and_preprocess, build_model, train_model, predict_sample
from tests.TestUtils import TestUtils
import tensorflow as tf

class TestIrisClassifierYaksha(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_obj = TestUtils()
        try:
            cls.X_train, cls.X_test, cls.y_train, cls.y_test, cls.scaler = load_and_preprocess()
            cls.model = build_model()
            cls.model = train_model(cls.model, cls.X_train, cls.y_train)
        except Exception as e:
            print(f"Setup failed due to unimplemented functions: {e}")
            cls.X_train = np.random.rand(100, 4)
            cls.y_train = np.random.randint(0, 2, 100)
            cls.X_test = np.random.rand(20, 4)
            cls.y_test = np.random.randint(0, 2, 20)
            cls.scaler = None
            cls.model = build_model()

    def test_data_shape(self):
        try:
            result = (
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
            result = len(self.model.layers) == 2 and isinstance(self.model.layers[0], tf.keras.layers.Dense)
            self.test_obj.yakshaAssert("TestModelStructure", result, "functional")
            print("TestModelStructure =", "Passed" if result else "Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestModelStructure", False, "functional")
            print("TestModelStructure = Failed | Exception:", e)

    def test_model_prediction_for_setosa(self):
        try:
            sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Likely Setosa
            sample_scaled = self.scaler.transform(sample)
            prediction = self.model.predict(sample_scaled)
            result = prediction[0][0] > 0.5  # Classified as Setosa
            self.test_obj.yakshaAssert("TestSetosaPrediction", result, "functional")
            print("TestSetosaPrediction =", "Passed" if result else "Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestSetosaPrediction", False, "functional")
            print("TestSetosaPrediction = Failed | Exception:", e)



    def test_model_accuracy(self):
        try:
            _, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
            result = accuracy > 0.9
            self.test_obj.yakshaAssert("TestModelAccuracy", result, "functional")
            print("TestModelAccuracy =", "Passed" if result else "Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestModelAccuracy", False, "functional")
            print("TestModelAccuracy = Failed | Exception:", e)

