# Полином Колмогорова-Габор  (Kolmogorov-Gabor Polynomial)

Библиотека для выполнения вычислений с использованием полного образа полинома Колмогорова-Габора.

## Описание

Эта библиотека реализует алгоритм для построения и использования полного полинома Колмогорова-Габора. Полином Колмогорова-Габора применяется в различных областях, таких как численные методы и анализ данных. Библиотека позволяет обучать модели, добавляя полиномиальные признаки на каждой итерации, а также делать предсказания, используя полученные полиномиальные признаки.

### Основные особенности:
- Обучение моделей с использованием полиномиальных признаков, полученных от исходных данных.
- Применение модели для предсказания с учетом полиномиальных признаков на разных уровнях.
- Возможность настройки количества итераций для обучения и предсказания.

## Установка

Для установки библиотеки можно использовать команду `pip`:

```bash
pip install kolmogorov-gabor-polynomial
```

## Использование

### Пример использования

```python
import pandas as pd
import numpy as np
from kolmogorov_gabor_polynomial import KolmogorovGaborPolynomial

# Пример данных
X = pd.DataFrame({
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100)
})

Y = pd.Series(np.random.rand(100))

# Создание экземпляра модели
model = KolmogorovGaborPolynomial()

# Обучение модели
model.fit(X, Y, stop=5)

# Предсказание на новых данных
predictions = model.predict(X, stop=5)

# Вывод предсказаний
print(predictions)
```

### Описание методов

#### `fit(self, X, Y, stop=None)`
Метод для обучения модели.

**Параметры**:
- `X` (DataFrame): Входные данные (признаки).
- `Y` (DataFrame или Series): Целевые значения.
- `stop` (int, необязательно): Количество итераций для обучения модели (по умолчанию None, что означает использование всех признаков).

**Возвращает**:
- `model` (LinearRegression): Обученная модель на последней итерации.

#### `predict(self, X, stop=None)`
Метод для выполнения предсказаний на основе обученной модели.

**Параметры**:
- `X` (DataFrame): Входные данные (признаки).
- `stop` (int, необязательно): Количество итераций для предсказания (по умолчанию None, что означает использование значения `self.stop`).

**Возвращает**:
- `predictions` (ndarray): Предсказанные значения.

## Зависимости

- `pandas`
- `numpy`
- `scikit-learn`