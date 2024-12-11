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
    final_predictions_df : DataFrame
        DataFrame containing the final predictions.
    """
    if stop is None:
        stop = self.stop

    # Create a copy of X for modification
    local_X = X.copy()

    # Initial predictions
    model = self.models_dict['1']
    predictions = model.predict(local_X)

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

    # Create the final DataFrame with only the final predictions
    final_predictions_df = pd.DataFrame({'Predictions': predictions.flatten()}, index=range(len(predictions)))

    return final_predictions_df
