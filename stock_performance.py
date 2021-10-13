# Importing the required libraries

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from pandas_datareader import data
import matplotlib.pyplot as plt
plt.style.use('bmh')

# Input the securities

n = int(input('Enter the number of securities: '))
tickers = []
for i in range(n):
    tickers.append(input())

# Combining adjusted closing prices since 2011 into one dataframe 'sec_data' (current date = 30/10/2020)

sec_data = pd.DataFrame()

for t in tickers:
    x = data.DataReader(t, data_source='yahoo', start='2011-1-1')
    print('\n'+t+':')
    print(x.head())
    sec_data[t] = x['Adj Close']

print(sec_data)


# STOCK PERFORMANCE SEGMENT

# Computing daily log returns 'sec_returns'

sec_returns = np.log(sec_data / sec_data.shift(1))
print(sec_returns.tail())

# Plotting the daily prices of all securities

(sec_data/sec_data.iloc[0]*100).plot(figsize=(15,10))
plt.title('Past Stock Prices Normalized to 100')
plt.xlabel('Years')
plt.ylabel('Normalized price')
plt.show()

# Computing average annual returns and risks, and their ratios

avg_ret=[]
avg_risk=[]
ret_risk_ratio=[]
for t in tickers:
    mn=(sec_returns[t].mean()*250)
    sd=(sec_returns[t].std()*250**0.5)
    avg_ret.append(mn)
    avg_risk.append(sd)
    ret_risk_ratio.append(mn/sd)

dict = {'Average Annual Return':avg_ret,'Average Annual Risk':avg_risk,'Return-to-Risk Ratio':ret_risk_ratio}
mean_returns = pd.DataFrame(dict,index=tickers)
print(mean_returns)

# The security with highest ratio

max_ret_risk_ratio = mean_returns['Return-to-Risk Ratio'].max()

print('Details of the security with the highest return-to-risk ratio:')
print(mean_returns.loc[mean_returns['Return-to-Risk Ratio']==max_ret_risk_ratio])

# All possible portfolios

from itertools import combinations 

portfolios = []

for r in range(2, n+1):
    combinations_object = combinations(tickers, r)
    portfolios += combinations_object

print(portfolios)

# 1000 iterations of randomized weight distribution and plotting return vs risk for each portfolio

port_returns=[[]]
port_risks=[[]]
port_weights=[[]]
mean_ret_t = mean_returns.transpose()
i=0
max_rrr=0
max_rrr_index=-1

for p in portfolios:
    port=list(p)
    port_dat=mean_ret_t[port]
    l=len(port)
    ann_ret=port_dat.loc['Average Annual Return',:]
    ann_risk=port_dat.loc['Average Annual Risk',:]
    i+=1
    reti=[]
    riski=[]
    wts=[]
    for x in range (1000):
        weights = np.random.random(l)
        weights /= np.sum(weights)
        ret=np.dot(ann_ret,weights)
        ris=np.sqrt(np.dot(weights.T,np.dot(sec_returns[port].cov() * 250, weights)))
        reti.append(ret)
        riski.append(ris)
        wts.append(weights)
        if ret/ris > max_rrr:
            max_rrr = ret/ris
            max_rrr_index = i
            max_rrr_weights = weights
            max_rrr_ret=ret
    port_returns.append(reti)
    port_risks.append(riski)
    port_weights.append(wts)
    
    
    port_df = pd.DataFrame({'Return': reti, 'Risk': riski})
    port_df.plot(x='Risk', y='Return', kind='scatter', figsize=(10, 6));
    plt.xlabel('Expected Risk')
    plt.ylabel('Expected Return')
    plt.title('CODE = '+str(i)+'   :   '+str(port))

# Portfolio with highest RRR 

print('The best choice for a portfolio:')
print(portfolios[max_rrr_index-1])
print('\nThe best ratios of investment in each security (in order):')
print(max_rrr_weights)
print('\nThe estimated maximum annual return-to-risk ratio for this portfolio with the suggested weights:')
print(round(max_rrr, 5))
print('\nReturn:\t',str(round(np.exp(max_rrr_ret)*100-100,3)),'%\t','Risk:\t',str(round(max_rrr_ret/max_rrr,5)))

# Highest return for a given risk

code=int(input('Enter the code for your preferred portfolio (refer the plots above): '))
max_risk = float(input('Enter the maximum risk you are willing to take (refer the plots above): '))

max_return = np.array(port_returns[code])[tuple(np.where(np.array(port_risks[code])<=max_risk))].max()
max_return_index = port_returns[code].index(max_return)
print('\nEstimated maximum annual profit with the risk:')
print(str(round(np.exp(max_return)*100-100,3)),'%')
print('\nRespective weights for the securities:')
print(port_weights[code][max_return_index])


# STOCK PRICE PREDICTION SEGMENT

# Dataframe with past and shifted past prices

pred_sec=input('Enter the ticker for the security that you want to predict the prices of: ')
days = int(input('Enter the number of future days for prediction: '))
pred_prices = sec_data[[pred_sec]]
pred_prices.columns = ['Past Prices']
pred_prices['Shifted Prices'] = pred_prices[['Past Prices']].shift(-days)
print(pred_prices)

# Preparing feature array

X = np.array(pred_prices.drop(['Shifted Prices'],1))[:-days]
print(X)

# Preparing target array

y = np.array(pred_prices['Shifted Prices'])[:-days]
print(y)

# Splitting into training and testing datasets, computing confidence intervals for different methods of regression

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
lin_reg_r2 = lin_reg.score(x_test, y_test)
print("Linear regression confidence interval:\t\t", lin_reg_r2)

dec_tree_reg = DecisionTreeRegressor()
dec_tree_reg.fit(x_train, y_train)
dec_tree_r2 = dec_tree_reg.score(x_test, y_test)
print("Decision tree regressor confidence interval:\t", dec_tree_r2)

svr = SVR() 
svr.fit(x_train, y_train)
svr_r2 = svr.score(x_test, y_test)
print("Support vector regression confidence interval:\t", svr_r2)

# Past prices in the last 'days' days

last_days_prices = np.array(pred_prices.drop(['Shifted Prices'],1))[-days:]
print(last_days_prices)

# Predicting future prices for 'days' days

predicted_prices = lin_reg.predict(last_days_prices)
print(predicted_prices)

# Plotting the predicted prices

plt.figure(figsize=(15,10))
plt.plot(predicted_prices)
plt.xlabel('Future days')
plt.ylabel('Predicted price (in USD)')
plt.title('Predicted Stock Prices')
plt.show()
