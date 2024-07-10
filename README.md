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
  <img src="https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/8c86188e-21a2-4db3-be41-e12f9bf6eae4" width="300" /> 
  <img src="https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/dae09575-db8c-4658-93c5-ff8c97329732" width="280" />
  <br><em>Data before and after log transformation</em>
</p>

**Differences:**
- Proportion: Pre-log graph displays absolute values, while post-log graph shows logarithmic values (percentage change), resulting in different vertical axis scales.
- Volatility: Post-log graph exhibits more stable fluctuations with lower and less volatile standard deviation compared to pre-log graph.
- Trend: Both graphs demonstrate an increasing trend, but the post-log graph displays a smoother trend less influenced by value fluctuations.

**Reason for the difference**: Log transformation reduces the impact of large values, focusing on rate of change. This smoothing effect decreases volatility and enhances trend clarity.

**Conclusion:** Log transformation stabilizes variance, improves trend visibility, and reduces data fluctuations.

### 2.2 Check for seasonality and trend in data
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

### 2.3 Check the stationarity of the data series
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

### 2.4 Autocorrelation test
```python
#correlation test
def plot_correlation(data):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    lag_plot(data, ax=axes[0])
    plot_pacf(data, ax=axes[1])
    plot_acf(data, ax=axes[2])
    plt.show()
# Check Auto Correlation
plot_correlation(df_closing)
```
![image](https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/dd9597c3-a3bb-4f4e-80b4-2d43c971cd97)

**Scatter Plot:** This plot shows a strong positive linear relationship between two variables, perhaps the closing prices of stocks at two different points in time. The x-axis is labeled as ( y(t) ) and the y-axis is ( y(t+1) ), both ranging from 5000 to 25000, indicating that today's closing price has a large impact on tomorrow's price.

**Partial Autocorrelation Graph (PACF):** This plot has the x-axis representing lags from 0 to about 35 and the y-axis from -1.00 to +1.00. The blue points are distributed near the zero line, showing low partial correlations for different lags, indicating that there is no significant relationship between the stock closing values ​​and previous values ​​after removing the influence of more recent closing values.

**Autocorrelation Graph (ACF):** This graph shows significant autocorrelation at different lags, indicating that the stock's closing price has a strong correlation with its past values, which may indicate non-randomness in the data and the potential for seasonal or trend patterns.

### 2.5 Overall ACB's closing price
Let's take a look at how ACB's closing price has changed over time:
```python
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(data['time'], data['close'])
plt.xlabel("Thời gian")
plt.ylabel("Giá Đóng cửa") 
plt.title("Giá Đóng cửa ACB Theo Thời gian")
plt.show()
```
![image](https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/01b5efdb-e79b-499c-b0b0-62161dcea1b3)

The chart shows a strong growth trend of ACB's closing price from 2012 to the end of 2023.

The main observable periods are:
- 2012 - 2015: Volatile period with prices fluctuating between 3000 - 5000.
-2016 - 2017: Stable period and slight growth.
- Late 2017 - 2020: Dramatic growth period, especially from late 2017 to early 2018 and mid-2020.
- 2021 - 2024: Strong volatility period with many peaks and troughs, but overall still maintaining an upward trend.

### 2.6 Closing Price Distribution Analysis

![image](https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/b9f795dd-7ae1-4033-926a-afb568dafb6a)

Analyzing the above chart will help us determine whether the data is normally distributed or skewed.

The histogram shows that the closing price distribution of ACB shares is right-skewed, with most of the values ​​concentrated in the range of 4000 to 10000.
Main features:
- Most common value: Around 4000 - 5000 (this is the early stage of the data, when ACB shares are still low).
- Values ​​that appear frequently:
- Around 9000 - 10000
- Around 25000 - 30000 (this is the late stage of the data, when ACB shares have grown significantly).
- Few values: In the middle of the range of 15000 - 20000.

**Comments:**
The right-skewed distribution shows the growth of ACB shares over time.

### 2.7 Summary of closing price statistics
We can also obtain descriptive statistics to summarize the central tendency and dispersion of closing prices:
![image](https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/e09e506a-5dc1-4f10-aadb-e8f8a868ecf7)

The box plot shows the distribution of ACB stock closing prices and provides information about outliers.
**Observations:**
- Most of the values ​​are concentrated: Within the box, from about 8000 to 17000.
- Median: The horizontal line inside the box shows that the median is close to 10000.
- Outliers: There are some outliers above the box, shown by individual dots. This shows that there are trading days when ACB's closing price increases dramatically compared to most other days.

**Comments:**
- The box plot confirms the presence of outliers in the closing price data, consistent with the analysis from the previous histogram.
- The gap between the quartiles is quite large, indicating significant price fluctuations of ACB stock.

### 2.8 Correlation Between Variables

![image](https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/adf11607-43d9-4824-bb26-19433fda3b34)

The correlation matrix shows the linear relationship between variables in ACB stock data.

**Observations:**

- Very strong correlation (nearly 1): Between price variables (open, high, low, close). This shows that these values ​​tend to change together.
- Medium correlation (about 0.67 - 0.68): Between trading volume and price variables. This relationship shows that when price changes, trading volume also tends to change.

**Comments:**
- The strong correlation between price variables is understandable, because they all reflect the value of ACB stock at different times during the trading day.
- Medium correlation with trading volume shows that trading volume can be a factor affecting price, but not the only factor.

### 2.9 ACB's Transaction Volume Over Time
![image](https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/1b96a598-9cd7-49af-90cb-ed6a8e4c6da5)

The chart shows the fluctuations in ACB stock trading volume from 2012 to early 2024.

**Observations:**
- Period 2012 - 2017: Trading volume remained low and relatively stable.
- From 2018 onwards: Trading volume fluctuated more strongly, with many peaks appearing.
- The highest peak: Located in early 2021, showing the special interest of investors at that time.

**Comments:**
- Trading volume tends to increase over time, especially from 2018 onwards, possibly due to the development of the stock market and the increasing interest in ACB stocks.

### 2.10 Relationship between closing price and trading volume
![image](https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/6dce3f96-b4d2-44e6-85ed-0f872238fbf1)

The scatter plot shows the relationship between the closing price and trading volume of ACB stock.

**Observations:**
- General trend: There is a positive correlation between the two variables, meaning that when the closing price increases, the trading volume also tends to increase.
- Dispersion: The relationship is not completely linear and has a fairly large dispersion. At the same price, the trading volume can fluctuate within a wide range.
- Outliers: There are some data points that are separate from the general trend, showing trading days with unusual volume compared to the price.

**Comments:**
- The positive correlation between price and trading volume shows that investor interest increases as the stock price increases.
However, the dispersion and the presence of outliers show that trading volume is not the only factor affecting price, and there may be other factors at play.

## 3. Data Preprocessing
### 3.1 Handling Missing Values
```python
print(data.isnull().sum())
```
![image](https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/a5d95c21-45ad-4118-aae2-b942ae261fc3)

The results show that the dataset does not contain missing values.

### 3.2 Handing Time Index
```python
# Handing Time Index
data['time'] = pd.to_datetime(data['time'])
data = data.set_index('time')
data = data[~data.index.duplicated(keep='first')]  # Delete duplicates
new_data = data.copy()
```
## 4. Model building process
![image](https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/400572f0-ed21-406a-9b28-00ef1dce3414)

## 5. Results 

