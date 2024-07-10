# Predicting-the-Closing-Price-of-ACB-Stock
Develop a machine learning model for time series data to forecast the closing price of ACB stock

## 1. Introduction
### 1.1 Overview
The project explores utilizing Python and machine learning models like **ARIMA, SARIMA, and LSTM** to analyze and forecast the stock market code **ACB**. ACB is a significant stock with substantial trading activity in Vietnam. This analysis can assist investors in comprehending stock trends, recognizing patterns, and discovering potential investment prospects.

### 1.2 Objective
- Delve into unclear relationships and trends to provide a more detailed view of stock price fluctuations.
- Develop and optimize forecasting models to analyze the time series of stock prices for ACB.
  Optimize the performance of trained models and combine their results to enhance forecasting abilities.
- Help investors make smarter buying and selling decisions.
- Help investors maximize profits and minimize risks in trading.
  
### 1.3 Data Source 
- Use **vnstock library** to get live stock price data
```python
# Import libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
!pip install -U vnstock
from vnstock import *
!pip install pmdarima

# Get Data
data = stock_historical_data(symbol='ACB',
                            start_date="2006-11-01",
                            end_date='2024-03-31', resolution='1D', type='stock', beautify=True, decor=False, source='TCBS')

data.head()
```
![image](https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/fefd8589-83ab-422a-9ec1-a0a0c5d3cf63)


## 2. Exploratory Data Analysis (EDA)
### 2.1 Data decomposition
<p align="center">
  <img src="https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/8c86188e-21a2-4db3-be41-e12f9bf6eae4" width="500" /> 
  <img src="https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/dae09575-db8c-4658-93c5-ff8c97329732" width="480" />
  <br><em>Data before and after log transformation</em>
</p>

**Differences:**
- Proportion: Pre-log graph displays absolute values, while post-log graph shows logarithmic values (percentage change), resulting in different vertical axis scales.
- Volatility: Post-log graph exhibits more stable fluctuations with lower and less volatile standard deviation compared to pre-log graph.
- Trend: Both graphs demonstrate an increasing trend, but the post-log graph displays a smoother trend less influenced by value fluctuations.

**Reason for the difference**: Log transformation reduces the impact of large values, focusing on rate of change. This smoothing effect decreases volatility and enhances trend clarity.

**Conclusion:** Log transformation stabilizes variance, improves trend visibility, and reduces data fluctuations.

### 2.3 Check for seasonality and trend in data
```python
#Check for seasonality and trend in data:
from statsmodels.tsa.seasonal import seasonal_decompose
decompose_result = seasonal_decompose(df_closing, model='multiplicative', period=60)
decompose_result.plot()
plt.show()
```
![image](https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/e777cb41-d5f1-45cb-81f9-5172fb70a875)

The analysis of the df_closing data reveals 4 components:
- **Close (Original Value**): Represents the initial data with a trend of increase over time and volatility.
- **Trend**: Shows the general trend of the data, disregarding seasonal factors and noise. The trend is steadily increasing over time.
- **Seasonal**: Displays the repeating seasonal factor within a 60-unit time cycle (e.g., month or quarter). The seasonal factor fluctuates around 1, indicating a weak seasonal influence.
- **Resid (Residual)**: Represents the remaining noise or random fluctuations after removing the trend and seasonal components. The residual fluctuates around 1 with no clear pattern.

**Conclusion:**
- The data exhibits a clear increasing trend over time.
- The seasonal impact is negligible.
- The random residuals show no particular pattern.

### 2.4 Check the stationarity of the data series
```python
# Check the stationarity of the dataset function
# ADF
def adf_test(data):
    print('Augmented Dickey-Fuller Test:')
    indices = ['Test Statistic', 'p-value', 'No. Lags Used', 'Number of Observations Used']
    test = adfuller(data, autolag='AIC')
    result = pd.Series(data=test[0:4], index=indices)
    for key, value in test[4].items():
        result[f'Critical Value ({key})'] = value
    return result

# KPSS
def kpss_test(data):
    print('KPSS Test:')
    indices = ['Test Statistic', 'p-value', 'No. Lags Used']
    test = kpss(data)
    result = pd.Series(data=test[0:3], index=indices)
    for key, value in test[3].items():
        result[f'Critical Value ({key})'] = value
    return result
```
``` python
print(adf_test(df_closing))
```
![image](https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/ea024872-9e79-482b-b638-63fc5f996a40)

**Comment:**
The absolute value of the statistical test value is smaller than the absolute value at the 1%, 5%, and 10% levels, so the hypothesis H0 cannot be rejected 

**=> The time series is not stationary.**

```python
print(kpss_test(df_closing))
```
![image](https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/94adbc79-00fe-409b-ae8d-86cbeaba4011)

**Comment:** The test statistic value is greater than the value at the 1%, 5%, and 10% levels, so we can reject the hypothesis H0 

**=> The time series is not stationary.**

## 3. Data Preprocessing
###

## 4. Model building process

## 5. Results 
