# Predicting-the-Closing-Price-of-ACB-Stock
Develop a machine learning model for time series data to forecast the closing price of ACB stock

## 1. Introduction
### 1.1 Overview
The project explores utilizing Python and machine learning models like **ARIMA, SARIMA, and LSTM** to analyze and forecast the stock market code **ACB**. ACB is a significant stock with substantial trading activity in Vietnam. This analysis can assist investors in comprehending stock trends, recognizing patterns, and discovering potential investment prospects.

### 1.2 Objective
- Delve into unclear relationships and trends to provide a more detailed view of stock price fluctuations.
- Develop and optimize forecasting models to analyze the time series of stock prices for ACB.
  Optimize the performance of trained models and combine their results to enhance forecasting abilities.
  
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

In this project, the technical chosen to split data into train and test sets includes: Traditional way, Multi-partition, Sliding window.
```python
# Split data into train and test
# Case 1: Split data in traditional way
train_size = int(len(df_train_test) * 0.9)
train_simple, test_simple = df_train_test[:train_size], df_train_test[train_size:]
# Print the sizes of each split for verification
print(f"Simple Time Split: Train={len(train_simple)}, Test={len(test_simple)}")

# Plot train and test data
plot_train_test(df_train_test, train_simple, test_simple, df_val)
```
![image](https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/7087ae62-b2c4-419a-a71c-dfeedbc0b067)

``` python
# Multi-partition cross-validation data split function
def mulTsCrossValidation(num, n_splits):
    split_position_lst = []
    for i in range(1, n_splits + 1):
        train_size = i * num // (n_splits + 1) + num % (n_splits + 1)
        test_size = num // (n_splits + 1)
        start = 0
        split = train_size
        end = train_size + test_size
        if end > num:
            end = num
        split_position_lst.append((start, split, end))
    split_position_df = pd.DataFrame(split_position_lst, columns=['start', 'split', 'end'])
    return split_position_df

split_position_df = mulTsCrossValidation(len(df_train_test), 3)
table = split_position_df
table.insert(0, 'No.', range(1, 1 + len(table)))
print(table)
```
<p align="center">
  <img src="https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/ae282adf-7314-4db8-a908-b557a933f742" width="280" /> 
  <img src="https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/9d5e025a-a05d-48f6-94c8-0ff2b05a0e7c" width="280" />
  <img src="https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/28634834-80b9-4750-8d10-308f517b3e74" width="280" />
  <br><em>Visualize train, test sets in Multi-partition cross-validation </em>
</p>

``` python
# Sliding window cross validation data split function
def slideWindowTsCrossValidation(num, n_splits):
    kfold_size = num // n_splits

    split_position_lst = []
    for i in range(n_splits):
        start = i * kfold_size
        end = start + kfold_size
        split = int(0.8 * (end - start)) + start
        split_position_lst.append((start, split, end))

    split_position_df = pd.DataFrame(split_position_lst, columns=['start', 'split', 'end'])
    return split_position_df
split_position_df = slideWindowTsCrossValidation(len(df_train_test), 3)
table = split_position_df
table.insert(0, 'No.', range(1, 1 + len(table)))
print(table)
```
<p align="center">
  <img src="https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/17d8b53f-f318-4692-a354-7e9d46a4ecb1" width="280" /> 
  <img src="https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/171426cd-f68f-469d-a1c6-c0873ffccd1a" width="280" />
  <img src="https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/cb3eb9cd-3b96-4898-b9e3-80412b38321a" width="280" />
  <br><em>Visualize train, test sets in Sliding window cross-validation </em>
</p>

#### AR
**Select the optimal lag order for the autoregressive (AR) model using the ar_select_order function.**
```python
# AR model
models = ar_select_order(train_simple, maxlag=30)
print('The lag order selected is: ', models.ar_lags)
model = AutoReg(train_simple, lags=1, old_names=False)
model_fit = model.fit()
model_fit.summary()
```
![image](https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/fad065b8-62ed-41f5-ad87-3213e1a70342)

The optimizal lag level is 1

**Train and visualize out come**
```python
# Train model
fc_ar = model_fit.forecast(len(test_simple)+len(df_val))
fc_ar_test = model_fit.forecast(len(test_simple))
fc_ar_val = model_fit.forecast(len(df_val))
fc_ar_values = fc_ar
fc_ar_values_test = fc_ar_test
fc_ar_values_val = fc_ar_val
fc_ar_values.index = (test_simple+df_val).index
fc_ar_values_test.index = test_simple.index
fc_ar_values_val.index = df_val.index

# Visualize result 
plt.figure(figsize=(12,5))
plt.plot(train_simple, label='Train')
plt.plot(test_simple, label='Test')
plt.plot(fc_ar_values_test, label='Predicted')
plt.fill_between(fc_ar_values_test.index, fc_ar_values_test, test_simple, color='b', alpha=.10)
plt.legend(loc='best')
plt.title('AR model')
plt.show()
```
<p align="center">
  <img src ="https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/624cb2bb-4d05-4539-a9cd-b4b0a322347c"/>
  <br><em>Visualize results in tranditiona way </em>
</p>
Do the same for the remaining train set split cases and save the results in a separate table. And here is results

<p align="center">
  <img src="https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/ff1058a0-6fed-4983-a2cc-e09e68cc868f" width="280" /> 
  <img src="https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/5437cfd6-35ea-4993-91f2-841eee4b9289" width="280" />
  <img src="https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/a3554a40-4008-4416-9b6b-afc02a518f55" width="280" />
  <br><em>Visualize AR model results in Multi-partition cross-validation </em>
</p>

<p align="center">
  <img src="https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/5dcd98c0-7aab-42a8-be6b-5d8cb5fa3c10" width="280" /> 
  <img src="https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/6e42568d-1add-4438-bf8c-dc93a0910d5d" width="280" />
  <img src="https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/838f951c-368d-4abc-a613-2fa66b718cba" width="280" />
  <br><em>Visualize AR model results in Sliding window cross-validation </em>
</p>


#### ARIMA
**Convert data to a stopped sequence**
```python
# Convert data --> stop sequence
def convert_stationary(data):
    data_diff = data.diff(1).dropna()
    return data_diff
train_simple_diff = convert_stationary(train_simple)

fix, ax =plt.subplots(2, sharex="all")
train_simple.plot(ax=ax[0], title ="Closing price")
train_simple_diff.plot(ax=ax[1], title = "1st derivative")
plt.show()
```
![image](https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/48c9ec1c-5ea3-4be5-9344-a05c08e4e363)

```python
# Retest based on ADF and KPSS
print(adf_test(train_simple_diff))
print('----------------------------------------')
print(kpss_test(train_simple_diff))
```
![image](https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/3819e23b-0772-413b-b131-e95572336293)

=> Data has stopped

**Determine ARIMA model parameters**
```python
plot_correlation(train_simple_diff)
```
![image](https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/c676faf2-0898-48df-acac-3e1580c2bfcd)

Based on the two PACF and ACF charts, we can determine that **the parameter "p" can be 1 or 2 and the parameter "q" can be 1 or 2.**

**Determine optimal ARIMA model parameters**
```python
stepwise_fit = auto_arima(train_simple, trace=True, suppress_warnings=True)
print(stepwise_fit.summary())
stepwise_fit.plot_diagnostics(figsize=(15, 8))
plt.show()
```
![image](https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/5f657bf8-84c5-4342-9c89-b2ca17f6da5d)

The best ARIMA model selected in this trandition case is **ARIMA(1,1,0)**

**Train and visualize out come**
```python
fc_arima = fitted.forecast(len(test_simple)+len(df_val))
fc_arima_test = fitted.forecast(len(test_simple))
fc_arima_val = fitted.forecast(len(df_val))
fc_values_arima = fc_arima
fc_values_arima_test = fc_arima_test
fc_values_arima_val = fc_arima_val
fc_values_arima.index = (test_simple+df_val).index
fc_values_arima_test.index = test_simple.index
fc_values_arima_val.index = df_val.index

# Plot test data
plt.figure(figsize=(12,5))
plt.plot(train_simple, label='Train')
plt.plot(test_simple, label='Test')
plt.plot(fc_values_arima_test, label='Predicted')
plt.fill_between(fc_values_arima_test.index, fc_values_arima_test, test_simple, color='b', alpha=.10)
plt.legend(loc='best')
plt.title('ARIMA model')
plt.show()
```
![image](https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/aa027950-6821-4365-b3a4-1a8a0d2ff880)

Do the same for the remaining train set split cases and save the results in a separate table. And here is results

<p align="center">
  <img src="https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/ddc98777-fe3d-40ef-bee6-c311408334fc" width="280" /> 
  <img src="https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/03e46394-4c02-40de-af97-05c9cd28f2e6" width="280" />
  <img src="https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/0cad9731-e6c2-412e-a87f-29b968de326b" width="280" />
  <br><em>Visualize AR model results in Multi-partition cross-validation </em>
</p>

<p align="center">
  <img src="https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/ab892a57-c8e3-48dc-a968-da5974a6343c" width="280" /> 
  <img src="https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/acac96ca-5f8d-4132-b271-d424cf38cef3" width="280" />
  <img src="https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/e78e17be-2285-473e-aa43-8b980aa612d6" width="280" />
  <br><em>Visualize AR model results in Sliding window cross-validation </em>
</p>

#### SARIMA
Similar to the ARIMA model, but with the inclusion of a seasonal parameter.
```python
#Check for seasonality and trend in data:
from statsmodels.tsa.seasonal import seasonal_decompose
decompose_result = seasonal_decompose(df_closing, model='multiplicative', period=60)
decompose_result.plot()
plt.show()
```
![image](https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/03cc3c85-5eb7-4ef1-85fd-2da24f4efa67)

- It is observed that although it is not much affected by seasonality because the seasonal value is around the value, it will be difficult to determine the specific seasonality of this data set, so we should use the estimation method by looking at the trend chart to evaluate, we choose the seasonal parameter as 12 months.
- Use auto_arima to help us automatically determine the most optimal p, d, q parameters to put into the SARIMA model, add the argument seasonal=True so that the model understands that the data is seasonal and m =12 is the number of months we define as the data cycle.

```python
stepwise_fit = auto_arima(df_train_sar, trace=True, suppress_warnings=True, seasonal=True, stepwise=True, test='adf',
                         m=12, d=1, D=1, start_q = 1, start_p = 1,
                         max_p=3,  max_q=3)
print(stepwise_fit.summary())
stepwise_fit.plot_diagnostics(figsize=(15, 8))
plt.show()
```
The result returned after the auto arima function runs to find optimal parameters is as follows:

![image](https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/e18278f4-5c87-4049-bdff-19ad0c4c3669)

- SARIMAX(0,1,1)x(2,1,0,12). Based on this, we will put those parameters into the SARIMA model for training.
- Use the SARIMAX function from the statsmodels library to apply the SARIMA model to the data file for training

**Train and visualize out come**
```python
import statsmodels.api as sm
model = sm.tsa.statespace.SARIMAX(df_train_sar, order=(0,1,0), seasonal_order=(2,1,0,12), enforce_stationarity = False, enforce_invertibility=False)
fitted = model.fit()
print(fitted.summary())

predict = fitted.predict(start= len(df_train_sar), end = len(df_train_sar)+len(df_test_sar)-1)
fc_values_arima_test = fitted.predict(start = len(df_test_sar))
predict.index = df_test_sar.index

# Print the predictions with their dates
df_test_sar = np.exp(df_test_sar)
df_train_sar = np.exp(df_train_sar)
predict = np.exp(predict)
print(predict)

plt.figure(figsize=(12,5))
plt.plot(df_train_sar, color='g', label='Train')
plt.plot(df_test_sar, color = 'y', label='Test')
plt.plot(predict, color='b', label='Test')
plt.title('ARIMA model')
plt.show()
```
  ![image](https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/6fd0db6e-fc8c-4291-9446-e47f42acdb44)

#### LSTM

Built and develop LSTM model

```python
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

numerical_columns = data.select_dtypes(include=["number"]).columns

scaler = MinMaxScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Get data
data_updated = data.drop('ticker', axis=1)


print(data_updated)
print(data_updated.columns)
print(data_updated['close'])

# Preparing data for LSTM model
def prepare_data(data_updated, target_col, window_size):
    X, y = [], []
    for i in range(len(data_updated) - window_size):
        window = data_updated.iloc[i:(i + window_size)]
        X.append(window.values)
        y.append(data_updated.iloc[i + window_size][target_col])

    # Convert list of numpy arrays to numpy arrays
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y

# LSTM model definition
def create_lstm_model(units=50, activation='relu', learning_rate=0.001):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units=units, activation=activation, input_shape=(X_train[0].shape[0], X_train[0].shape[1])),
        tf.keras.layers.Dense(units=1)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Prepare the data and split it into training and testing sets
window_size = 30
target_col = 'close'
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_updated[[target_col]])  # Take only 'close' column to normalize
data_updated[target_col] = scaled_data  # Assign normalized data back to 'close' column
X, y = prepare_data(data_updated, target_col, window_size)

# Split into train test set and validation set
X_train_test, X_val = X[:int(len(X) * 0.9)], X[int(len(X) * 0.9):]
y_train_test, y_val = y[:int(len(y) * 0.9)], y[int(len(y) * 0.9):]

# Continue to divide the train test set into train set and test set
X_train, X_test, y_train, y_test = train_test_split(X_train_test, y_train_test, test_size=0.1, shuffle=False)


# Define hyperparameters to search
param_grid = {
    'units': [50, 100, 150],
    'activation': ['relu', 'tanh'],
    'learning_rate': [0.001, 0.01, 0.1]
}

# Perform optimal hyperparameter search
best_score = float('inf')
best_params = {}
for units in param_grid['units']:
    for activation in param_grid['activation']:
        for learning_rate in param_grid['learning_rate']:
            model = create_lstm_model(units=units, activation=activation, learning_rate=learning_rate)
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)  # Train model
            y_pred = model.predict(X_train)  # Prediction on training set

            # Check and remove NaN values
            y_pred = np.nan_to_num(y_pred)

            score = mean_squared_error(y_train, y_pred)  # Evaluate model
            if score < best_score:
                best_score = score
                best_params = {'units': units, 'activation': activation, 'learning_rate': learning_rate}

# Print out the optimal results
print("Best: %f using %s" % (best_score, best_params))

# Train the best model on the entire training set
best_model = create_lstm_model(**best_params)
best_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

# Create a list to store all the predicted values
predicted_prices_test = []

# Predicting closing price from test set using trained model
y_pred_test = best_model.predict(X_test)

# Convert the predicted and actual values ​​of the test set back to the original space
y_pred_original = scaler.inverse_transform(y_pred_test)* (28500.000000 - 4341.000000) + 4341.000000
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)) * (28500.000000 - 4341.000000) + 4341.000000

# Loop through all predictions and actual values ​​to print out
for i in range(len(y_pred_original)):
    predicted_price = y_pred_original[i][0]
    actual_price = y_test_original[i][0]
    # print("Predicted close price:", predicted_price, "Actual close price:", actual_price)
    predicted_prices_test.append(predicted_price)  # Add predicted price to list
```

```python
# Number of data points to use
num_data_points = len(y_pred_original)

# Use only the part of data.index corresponding to the number of data points
indices = data.index[-num_data_points:]

# Visualize comparison between predicted and actual values
plt.plot(indices, y_pred_original, label='Predicted close price', color='blue')
plt.plot(indices, y_test_original, label='Actual close price', color='red')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.title('Comparison between Predicted and Actual Close Prices')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# Number of values ​​to display
num_values = 50

# Get final data
last_data = data.iloc[-num_values:]

# Create an array containing the indices of the data points
indices = range(num_values)

# Get the time corresponding to the data points
timestamps = last_data.index

# Visualize comparison between predicted and actual values
plt.plot(timestamps, y_pred_original[-num_values:], label='Predicted close price', color='blue')
plt.plot(timestamps, y_test_original[-num_values:], label='Actual close price', color='red')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.title('Comparison between Predicted and Actual Close Prices (50 closest data)')
plt.xticks(rotation=45)
plt.legend()
plt.show()
```
![image](https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/479e29e3-26d0-43bd-b71e-8fd23f251ffd)

## 5. Results 

![image](https://github.com/thanhloc81/Predicting-the-Closing-Price-of-ACB/assets/151768013/6ae0e41e-57eb-4940-81a3-7f301d0a2792)

The results show that the RMSE score of the LSTM model is the smallest and ranks first, meaning that the LSTM model provides the best prediction efficiency on the same training dataset. In-depth analysis reveals:

**LSTM Model:**

- RMSE: 0.016391, the LSTM model has the lowest RMSE value, implying a very small average error between prediction and actual value.
- MAPE: 0.164, the MAPE value of the LSTM model is the highest, indicating a higher average percentage of absolute error compared to the actual value. This suggests that the model may have larger errors relative to the actual value at a fixed rate.
- MAE: 0.106880, the MAE value of the LSTM model is the lowest, showing a very small average absolute error between prediction and actual value.

**ARIMA and Auto Regressive Models:**
The ARIMA and Auto Regressive models have lower RMSE, MAPE, and MAE values compared to the LSTM model, implying they also have good prediction capabilities on the test dataset.
However, some ARIMA and Auto Regressive models have the lowest MAPE values, suggesting they have the lowest average percentage of absolute error compared to the actual value.

In summary, although the LSTM model has the lowest RMSE and MAE, indicating the highest accuracy in predicting values, it is important to note its high MAPE value, indicating that some of its predictions may not be accurately correct at a fixed rate relative to the actual value. On the other hand, some ARIMA and Auto Regressive models have the lowest MAPE values, indicating they have the lowest average percentage of absolute error compared to the actual value.
