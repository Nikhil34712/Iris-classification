import unittest
import pickle
from sklearn.metrics import accuracy_score

class TestIrisClassification(unittest.TestCase):
    def setUp(self):
        # Load the trained model and test data from the .pkl file
        with open('iris.pkl', 'rb') as file:
            data = pickle.load(file)
            self.sv = data['sv']
            self.X_test = data['X_test']
            self.y_test = data['y_test']

    def test_accuracy(self):
        # Perform prediction on test data
        y_pred = self.sv.predict(self.X_test)
        # Calculate accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        # Assert that accuracy is greater than 0.9
        self.assertGreater(accuracy, 0.9, "Accuracy should be greater than 0.9")

if __name__ == '__main__':
    unittest.main()
