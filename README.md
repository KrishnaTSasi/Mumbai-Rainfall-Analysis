# Mumbai-Rainfall-Analysis
 🌧️ **Mumbai Rainfall Time Series Analysis Project**

 📅 Duration: 1922 – 2021 (100 years)

 🔢 **1. Data Collection**

* **Source**: Open government portals or IMD (India Meteorological Department)
* **Data**: Monthly total rainfall in Mumbai from 1922 to 2021

 ✅ Interpretation:

Collected data was reliable and continuous for 100 years, allowing long-term time series analysis.

 🧹 **2. Data Cleaning & Preprocessing**

* Handled missing values (if any) using interpolation or forward-fill
* Converted date column to datetime format and set it as index
* Resampled (if needed) to monthly/annual aggregates

 ✅ Interpretation:

The data was consistent and required minimal cleaning. It was ready for time series analysis.

 📊 **3. Exploratory Data Analysis (EDA)**

* **Line Plot**: Monthly rainfall over 100 years
* **Boxplots**: Month-wise rainfall distribution
* **Histograms**: Distribution of rainfall amounts
* **Seasonal Plot**: Identify patterns across months/years

 ✅ Interpretation:

* Peak rainfall observed in **July**
* Heavy rains consistently during **June to September (Monsoon)**
* High inter-annual variability in monsoon seasons

 🧭 **4. Decomposition of Time Series**

Used `seasonal_decompose()` or STL to split into:

* **Trend**
* **Seasonality**
* **Residual (Irregular)**

 ✅ Interpretation:

* No **seasonal component**  
* Slight **upward trend** in long-term rainfall
* Residuals showed random fluctuations

 📉 **5. Stationarity Testing**

* **ADF Test (Augmented Dickey-Fuller)**

  * ADF statistic: -8.29
  * p-value: \~0.000
    → **Reject null hypothesis** (Data is stationary)

* **KPSS Test (Kwiatkowski-Phillips-Schmidt-Shin)**

  * KPSS statistic: 0.768
  * p-value: 0.01
    → **Reject null** (Data is trend-stationary)

 ✅ Interpretation:

* ADF confirms **stationarity**
* KPSS shows **trend-stationary**

 🔁 **6. ACF & PACF Analysis**

* **ACF (Autocorrelation)**: Suggests White Noise Model
* **PACF (Partial ACF)**: Suggests White Noise Model

 ✅ Interpretation:

* No Seasonal spikes indicating **White Noise Model**
* Used plots to highest rainfall in month and year.

 📈 **7. Forecasting**

* Forecasted future rainfall values (e.g., next 12 or 24 months)
* Visualized forecast with confidence intervals

 ✅ Interpretation:

* Seasonal peaks continued in forecasted values
* Useful for monsoon planning and water management

 📊 **9. Residual Diagnostics**

* **Ljung-Box Test**:

  * p-value > 0.05 → Residuals are white noise ✅
* **Breusch–Pagan Test**:

  * p-value > 0.05 → Homoscedasticity ✅
* Residual plots showed no autocorrelation or trends

 ✅ Interpretation:

Model assumptions were satisfied. Residuals are well-behaved → **Good model fit**

**9. Code**

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import drive
drive.mount('/content/drive')

data=pd.read_csv("/content/drive/MyDrive/mumbai-monthly-rains.csv")
data

import matplotlib.pyplot as plt
plt.figure(figsize=(10,8))
plt.plot(data['Year'],data['Total'])
plt.title('Annual Rainfall in Mumbai')
plt.xlabel('Year')
plt.ylabel('Rainfall(mm)')
plt.legend()
plt.show()

# Hovering Annual Rainfall in Mumbai
import plotly.express as px
fig=px.line(data,x='Year',y='Total')
fig.update_layout(title='Annual Rainfall in Mumbai',title_font_size=20,title_x=0.5)
fig.update_xaxes(title_text='Year')
fig.update_yaxes(title_text='Rainfall(mm)')
fig.show()

# Avearage monthly rainfall using rainfall
import matplotlib.pyplot as plt
plt.figure(figsize=(10,8))
monthly_avg=data[['Jan','Feb','Mar','April','May','June','July','Aug','Sept','Oct','Nov','Dec']].mean()
monthly_avg.plot(kind='bar')
plt.title('Average Monthly Rainfall in Mumbai')
plt.xlabel('Month')
plt.ylabel('Rainfall(mm)')
plt.show()

** Highest Rainfall: 944 mm
Month: July **

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data=data.melt(id_vars=['Year'], value_vars=['Jan','Feb','Mar','April','May','June','July','Aug','Sept','Oct','Nov','Dec'], var_name='Month', value_name='Rainfall')

plt.figure(figsize=(14, 6))
sns.boxplot(x='Month', y='Rainfall', data=datak, order=['Jan','Feb','Mar','Apr','May','June','July','Aug','Sept','Oct','Nov','Dec'])
plt.title('Monthly Rainfall Distribution in Mumbai (1922–2021)')
plt.ylabel('Rainfall (mm)')
plt.grid(True)
plt.show()

# Seasonal rainfall
import matplotlib.pyplot as plt
data['Pre_Monsoon']=data[['Mar','April','May']].sum(axis=1)
data['Monsoon']=data[['June','July','Aug','Sept']].sum(axis=1)
data['Post_Monsoon']=data[['Oct','Nov','Dec']].sum(axis=1)
data['Winter']=data[['Jan','Feb']].sum(axis=1)
seasons=['Pre_Monsoon','Monsoon','Post_Monsoon','Winter']
for season in seasons:
  plt.plot(data['Year'],data[season],marker='o',label=season)
plt.title('Seasonal Rainfall in Mumbai')
plt.xlabel('Year')
plt.ylabel('Rainfall(mm)')
plt.legend()
plt.show()


**Monsoon season (June–September) receives the highest rainfall.**

**Post Monsoon season (March, April, May) is receiving the second highest rainfall.**

# Seasonal rainfall
data['Pre_Monsoon']=data[['Mar','April','May']].sum(axis=1)
data['Monsoon']=data[['June','July','Aug','Sept']].sum(axis=1)
data['Post_Monsoon']=data[['Oct','Nov','Dec']].sum(axis=1)
data['Winter']=data[['Jan','Feb']].sum(axis=1)
seasons=['Pre_Monsoon']#,'Monsoon','Post_Monsoon','Winter']
for season in seasons:
  plt.plot(data['Year'],data[season],marker='o',label=season)
plt.title('Seasonal Rainfall in Mumbai')
plt.xlabel('Year')
plt.ylabel('Rainfall(mm)')
plt.legend()
plt.show()

**Highest Rainfall: 944 mm
Year: 2005**

import matplotlib.pyplot a plt
from statsmodels.tsa.seasonal import seasonal_decompose
result=seasonal_decompose(data['Total'],model='additive',period=12)
result.plot()
plt.show()

*8Additive: seasonal fluctuations and trend are roughly constant over time.
Trend: Long-term increase or decrease.
Seasonal: Regular repeating patterns.
Residual: Remaining noise.**

*Check the stationary.**

**ADF Test**

from statsmodels.tsa.stattools import adfuller
adf_result=adfuller(data['Total'])
print("ADF statistics:",adf_result[0])
print("p_value:",adf_result[1])


**p_value is less than hypothesis. so we reject the hypothesis.
i.e a Stationary**

**KPSS Test**

from statsmodels.tsa.stattools import kpss
kpss_result=kpss(data['Total'])
print("KPSS statistics:",kpss_result[0])
print("p_value:",kpss_result[1])

*p_value is  less than hypothesis. so we are not rejecting the hypothesis.*


**Plot ACF and PACF**

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
plot_acf(data['Total'],lags=36)
plt.show()
plot_pacf(data['Total'],lags=36)
plt.show()

**The ACF and PACF plots show that most spikes are within the confidence intervals, indicating no significant autocorrelation in the residuals.
i.e a white noise.**

**White noise, it suggests there are no significant trends (e.g., increasing or decreasing rainfall over time) or seasonal patterns (e.g., predictable monsoon behavior).**

from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test=acorr_ljungbox(data['Total'], lags=[10], return_df=True)
print(lb_test)

**p-value > 0.05: Fail to reject
𝐻
0

 . i.e  white noise.**


from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test=acorr_ljungbox(data['Total'],lags=[10],boxpierce=True)
print(lb_test)



import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
qqplot(data['Total'],line='s')
plt.title("Q-Q plot")
plt.show()

**points lie close to red line, that data is approximately normally distributed .
Mean:close to 0, as the middle points align well with line.
Variance: Mostly constant in center, but deviations  non-constant variance.**

import matplotlib.pyplot as plt
import seaborn as sns
residuals=result.resid
plt.figure(figsize=(8,6))
sns.histplot(residuals, kde=True, bins=20, color='blue', edgecolor='black')
plt.title("Histogram(rainfall data)")
plt.xlabel("Rainfall(mm)")
plt.ylabel("Frequency")
plt.show()

**result.resid :- difference between the predicted and actual values.
KDE- Kernel Density Estimation, a smooth curve showing distribution of data, like a histogram but continuous.
Normal Distribution: A bell-shaped histogram.
Skewness/Kurtosis: *Skewed or heavy-tailed histograms.
                   *Right Skew (positively skewed): right tail of histogram is longer.
                   *Left Skew (negatively skewed): left tail is longer.
^residuals  to be skewed right (positively skewed), indicating the errors are not symmetrically distributed around zero.**                 

**Most of the data points cluster around the mean.

The distribution is symmetric.

Extreme values (very high or very low rainfall) are rare.**

**Shapiro Wilk test**

from scipy.stats import shapiro
stat,p_value=shapiro(data['Total'])
print(f"Test statistic:{stat}")
print(f"p_value:{p_value}")

**Test Statistic: 0.9825 ( 1, data is normal).
P-value: 0.1191 (> 0.05).
 i.e, data is normally distributed & cannot be rejected.

The Mumbai rainfall data is likely normally distributed.

data is Gaussian:
The mean is zero (stable) and variance is constant.
Q-Q plots, histograms, and statistical tests (e.g., Shapiro-Wilk test) confirmed.

Data is white noise and normally distributed, it suggests that Mumbai’s rainfall over 121 years is random and stable, with no significant trends or patterns.**


 🧠 **10. Insights & Conclusions**

* July has highest rainfall; consistent peak across years
* Mumbai rainfall shows seasonal and mild upward trend
* Data is stationary.
* Forecasting supports urban water management and disaster planning


