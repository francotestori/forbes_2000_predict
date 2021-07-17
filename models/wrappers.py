from sklearn import metrics
import numpy as np


class ResultWrapper:
    def __init__(self, y_test: np.ndarray, y_pred: np.ndarray) -> None:
        super().__init__()
        self.y_test = y_test
        self.y_pred = y_pred

    def mae(self):
        return metrics.mean_absolute_error(self.y_test, self.y_pred)

    def mse(self):
        return metrics.mean_squared_error(self.y_test, self.y_pred)

    def rmse(self):
        return np.sqrt(
            metrics.mean_squared_error(self.y_test, self.y_pred)
        )

    def r2(self):
        return metrics.r2_score(self.y_test, self.y_pred)

    def residuals(self):
        return self.y_test - self.y_pred

    def print(self):
        print('R2 Square:', self.r2())
        print('Mean Absolute Error:', self.mae())
        print('Mean Squared Error:', self.mse())
        print('Root Mean Squared Error:', self.rmse())

    def to_dict(self):
        return {
            'R2': self.r2(),
            'MAE': self.mae(),
            'MSE': self.mse(),
            'RMSE': self.rmse()
        }
