import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class FullKolmogorovGaborPolynomial:
    """
    Класс для построения полинома Колмогорова-Габора.

    Атрибуты:
    ----------
    models_dict : dict
        Словарь для хранения обученных моделей.

    partial_polynomial_df : DataFrame
        DataFrame для хранения промежуточных результатов во время обучения.

    stop : int
        Количество итераций для обучения модели.
    """

    def __init__(self):
        """
        Инициализирует класс KolmogorovGaborPolynomial.
        """
        self.models_dict = {}  # Словарь для хранения моделей

    def fit(self, X, Y, stop=None):
        """
        Обучает модель на основе входных данных.

        Параметры:
        ----------
        X : DataFrame
            Входные данные (признаки).
        Y : DataFrame или Series
            Целевые значения.
        stop : int, необязательно
            Количество итераций для обучения модели (по умолчанию None, что означает использование всех признаков).

        Возвращает:
        ----------
        model : LinearRegression
            Обученная модель на последней итерации.
        """
        if stop is None:
            stop = len(X.columns)
        self.stop = stop

        # Создаем копию X для модификации
        local_X = X.copy()

        # Начальная модель (первая итерация)
        model = LinearRegression()
        model.fit(local_X, Y)
        predictions = model.predict(local_X)

        # Создаем DataFrame для хранения промежуточных результатов
        self.partial_polynomial_df = pd.DataFrame(index=Y.index)
        self.partial_polynomial_df['Y'] = Y.values.flatten()
        self.partial_polynomial_df['Y_pred'] = predictions.flatten()

        # Добавляем первый столбец из local_X, возведенный в квадрат, в partial_polynomial_df и удаляем его из local_X
        self.partial_polynomial_df[local_X.columns[0] + '^2'] = local_X.iloc[:, 0] ** 2
        local_X.drop(local_X.columns[0], axis=1, inplace=True)

        self.models_dict['1'] = model

        for i in range(2, stop + 1):
            # Добавляем новую полиномиальную функцию Y_pred
            self.partial_polynomial_df[f'Y_pred^{i}'] = (predictions ** i).flatten()

            # Ограничиваем значения предсказаний, чтобы избежать переполнения
            self.partial_polynomial_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            self.partial_polynomial_df.fillna(0, inplace=True)

            # Добавляем следующий столбец из local_X, возведенный в квадрат, в partial_polynomial_df, если доступен
            if not local_X.empty:
                self.partial_polynomial_df[local_X.columns[0] + '^2'] = local_X.iloc[:, 0] ** 2
                local_X.drop(local_X.columns[0], axis=1, inplace=True)

            # Обучаем новую модель с дополнительными признаками
            model = LinearRegression()
            X_new = self.partial_polynomial_df.drop(columns='Y')
            model.fit(X_new, Y)
            predictions = model.predict(X_new)

            self.models_dict[str(i)] = model

        return self.models_dict[str(stop)]

    def predict(self, X, stop=None):
        """
        Делает предсказания на основе обученной модели.

        Параметры:
        ----------
        X : DataFrame
            Входные данные (признаки).
        stop : int, необязательно
            Количество итераций для предсказания (по умолчанию None, что означает использование значения self.stop).

        Возвращает:
        ----------
        predictions : ndarray
            Предсказанные значения.
        """
        if stop is None:
            stop = self.stop

        # Создаем копию X для модификации
        local_X = X.copy()

        # Начальные предсказания
        model = self.models_dict['1']
        predictions = model.predict(local_X)

        if stop == 1:
            return predictions

        # Создаем DataFrame для хранения промежуточных результатов предсказаний
        predict_polynomial_df = pd.DataFrame(index=X.index)
        predict_polynomial_df['Y_pred'] = predictions.flatten()

        # Добавляем первый столбец из local_X, возведенный в квадрат, в predict_polynomial_df и удаляем его из local_X
        predict_polynomial_df[local_X.columns[0] + '^2'] = local_X.iloc[:, 0] ** 2
        local_X.drop(local_X.columns[0], axis=1, inplace=True)

        for i in range(2, stop + 1):
            # Добавляем новую полиномиальную функцию Y_pred
            predict_polynomial_df[f'Y_pred^{i}'] = (predictions ** i).flatten()

            # Ограничиваем значения предсказаний, чтобы избежать переполнения
            predict_polynomial_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            predict_polynomial_df.fillna(0, inplace=True)

            # Добавляем следующий столбец из local_X, возведенный в квадрат, в predict_polynomial_df, если доступен
            if not local_X.empty:
                predict_polynomial_df[local_X.columns[0] + '^2'] = local_X.iloc[:, 0] ** 2
                local_X.drop(local_X.columns[0], axis=1, inplace=True)

            model = self.models_dict[str(i)]
            predictions = model.predict(predict_polynomial_df)

        return predictions
