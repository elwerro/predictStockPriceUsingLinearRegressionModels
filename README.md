# predictStockPriceUsingLinearRegressionModels
Using sci-kit-learn linear regression models and xgboost model to predict stock price. XGBoost is an optimized distributed gradient boosting library. It uses Gradient Boosting framework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM).

This python script fetches the stock data from quandl.com then formats/splits the data in a way that uses every 6 consecutive days as training to predict the 7th day. It splits the data into two parts: the last 20 days as testing and the remaining for training. It will also graph results.

# Dependencies
```python3 -m pip install matplotlib```

```python3 -m pip install numpy```

```python3 -m pip install pandas```

```python3 -m pip install -U scikit-learn```

```python3 -m pip install xgboost```

```python3 -m pip install seaborn```

```python3 -m pip install quandl```

# Run script
```python3 predictStockPriceUsingLinearRegressionModels.py```
