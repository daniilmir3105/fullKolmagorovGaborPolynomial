import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class KolmogorovGaborPolynomial:
    """
    Class for constructing the Kolmogorov-Gabor polynomial.

    Attributes:
    ----------
    models_dict : dict
        Dictionary for storing trained models.

    partial_polynomial_df : DataFrame
        DataFrame for storing intermediate results during training.

    stop : int
        Number of iterations for training the model.
    """

    def __init__(self):
        """
        Initialize the KolmogorovGaborPolynomial class.
        """
        self.models_dict = {}  # Dictionary for storing models

    def fit(self, X, Y, stop=None):
        """
        Train the model based on input data.

        Parameters:
        ----------
        X : DataFrame
            Input data (features).
        Y : DataFrame or Series
            Target values.
        stop : int, optional
            Number of iterations for training the model (default is None, which means using all features).

        Returns:
        ----------
        model : LinearRegression
            The trained model at the last iteration.
        """
        if stop is None:
            stop = len(X.columns)
        self.stop = stop

        # Create a copy of X for modification
        local_X = X.copy()

        # Initial model (first iteration)
        model = LinearRegression()
        model.fit(local_X, Y)
        predictions = model.predict(local_X)

        # Create a DataFrame for storing intermediate results
        self.partial_polynomial_df = pd.DataFrame(index=Y.index)
        self.partial_polynomial_df['Y'] = Y.values.flatten()
        self.partial_polynomial_df['Y_pred'] = predictions.flatten()

        # Add the first column from local_X, squared, to partial_polynomial_df and remove it from local_X
        self.partial_polynomial_df[local_X.columns[0] + '^2'] = local_X.iloc[:, 0] ** 2
        local_X.drop(local_X.columns[0], axis=1, inplace=True)

        self.models_dict['1'] = model

        for i in range(2, stop + 1):
            # Add new polynomial feature of Y_pred
            self.partial_polynomial_df[f'Y_pred^{i}'] = (predictions ** i).flatten()

            # Limit prediction values to avoid overflow
            self.partial_polynomial_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            self.partial_polynomial_df.fillna(0, inplace=True)

            # Add the next column from local_X, squared, to partial_polynomial_df, if available
            if not local_X.empty:
                self.partial_polynomial_df[local_X.columns[0] + '^2'] = local_X.iloc[:, 0] ** 2
                local_X.drop(local_X.columns[0], axis=1, inplace=True)

            # Train a new model with additional features
            model = LinearRegression()
            X_new = self.partial_polynomial_df.drop(columns='Y')
            model.fit(X_new, Y)
            predictions = model.predict(X_new)

            self.models_dict[str(i)] = model

        return self.models_dict[str(stop)]

    def predict(self, X, stop=None):
        """
        Make predictions based on the trained model.

        Parameters:
        ----------
        X : DataFrame
            Input data (features).
        stop : int, optional
            Number of iterations for prediction (default is None, which means using self.stop value).

        Returns:
        ----------
        predictions : ndarray
            Predicted values.
        """
        if stop is None:
            stop = self.stop

        # Create a copy of X for modification
        local_X = X.copy()

        # Initial predictions
        model = self.models_dict['1']
        predictions = model.predict(local_X)

        if stop == 1:
            return predictions

        # Create a DataFrame for storing intermediate prediction results
        predict_polynomial_df = pd.DataFrame(index=X.index)
        predict_polynomial_df['Y_pred'] = predictions.flatten()

        # Add the first column from local_X, squared, to predict_polynomial_df and remove it from local_X
        predict_polynomial_df[local_X.columns[0] + '^2'] = local_X.iloc[:, 0] ** 2
        local_X.drop(local_X.columns[0], axis=1, inplace=True)

        for i in range(2, stop + 1):
            # Add new polynomial feature of Y_pred
            predict_polynomial_df[f'Y_pred^{i}'] = (predictions ** i).flatten()

            # Limit prediction values to avoid overflow
            predict_polynomial_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            predict_polynomial_df.fillna(0, inplace=True)

            # Add the next column from local_X, squared, to predict_polynomial_df, if available
            if not local_X.empty:
                predict_polynomial_df[local_X.columns[0] + '^2'] = local_X.iloc[:, 0] ** 2
                local_X.drop(local_X.columns[0], axis=1, inplace=True)

            model = self.models_dict[str(i)]
            predictions = model.predict(predict_polynomial_df)

        return predictions
