from cmath import sqrt
from distutils.spawn import spawn
from fileinput import close
from lib2to3.pygram import Symbols
from operator import index
from xml.sax.handler import property_interning_dict
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
from scipy.optimize import minimize


end_date = date.today()
start_date = end_date - relativedelta(years=15)

tickers = [ 'IWV',
            'AGG',
            'DBC',
            'VNQ']

price_df = pdr.get_data_yahoo(symbols=tickers,start=start_date, end=end_date, ret_index=True)
ret_df = price_df['Ret_Index'].pct_change(1) #df of rolling daily returns for each asset class
start_weight = [1/len(tickers)]*len(tickers)
ew_port_ret = ret_df.dot(start_weight) #df of daily returns of equal weight portfolio 

stdev = []
for ticks in tickers:
    stdev.append(ret_df[ticks].std()*np.sqrt(252))
ann_ret = []
for ticks in tickers:
    ann_ret.append((ret_df[ticks].mean()+1)**252-1)

rf = []
today=date.today()
while True:
    try:
        rf_df = pdr.get_data_yahoo(symbols='^IRX',start=today,end=today)
        break
    except KeyError:
        today = today - relativedelta(days=1)
for n in tickers:
    rf.append(rf_df['Close'].values[0]/100)
rf_rate = rf[0]


risk_ret_df = {'Symbols':tickers, 'Annual_Return':ann_ret,'Annual_Risk':stdev,'Risk_Free_Rate':rf}
risk_ret_df = pd.DataFrame(risk_ret_df)
risk_ret_df['Sharpe_Ratio'] = (risk_ret_df.Annual_Return - risk_ret_df.Risk_Free_Rate)/risk_ret_df.Annual_Risk
risk_ret_df.loc[risk_ret_df.shape[0]] = 'EW_PORT', (ew_port_ret.mean()+1)**252-1, ew_port_ret.std()*np.sqrt(252) ,rf_rate, (((ew_port_ret.mean()+1)**252-1) - rf_rate)/(ew_port_ret.std()*np.sqrt(252))

ann_cov_matrix = ret_df.cov()*252
ew_port_var = np.transpose(start_weight)@ann_cov_matrix@start_weight
w = np.transpose(start_weight)@ann_cov_matrix
risk_attribution = []
for n in range(len(tickers)):
    risk_attribution.append(list(w)[n]*start_weight[n]/ew_port_var)

def get_risk_attribution(weights, rets):
    cov_matrix = rets.cov()*252
    port_var = np.transpose(weights)@cov_matrix@weights
    ws = np.transpose(weights)@cov_matrix
    risk_att = []
    for n in range(len(rets.columns)):
        risk_att.append(list(ws)[n]*weights[n]/port_var)
    return risk_att

def objective_error(weights, rets):
    risk_att = get_risk_attribution(weights, rets)
    target = [1/len(rets.columns)]*len(rets.columns)
    error = 0
    for i in range(len(rets.columns)):
        error += (risk_att[i]-target[i])**2
    return error

def get_risk_parity_weights(rets):
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                   {'type': 'ineq', 'fun': lambda x: x})
    weights = minimize(fun=objective_error, x0=start_weight, args=rets,constraints=constraints, tol=1e-10,)
    return weights.x

rp_weights = get_risk_parity_weights(ret_df)

result = pd.DataFrame(data=[rp_weights, get_risk_attribution(rp_weights, ret_df)], columns=tickers, index=['Weights','Risk Attribution'])
print(result)

rp_port_ret = ret_df.dot(rp_weights) #df of daily returns of rp weight portfolio
risk_ret_df.loc[risk_ret_df.shape[0]] = 'RP_PORT', (rp_port_ret.mean()+1)**252-1, rp_port_ret.std()*np.sqrt(252) ,rf_rate, (((rp_port_ret.mean()+1)**252-1) - rf_rate)/(rp_port_ret.std()*np.sqrt(252))

rp_port_ret_index = np.cumprod(1 + rp_port_ret/(np.std(rp_port_ret)*sqrt(252)*100))
ew_port_ret_index = np.cumprod(1 + ew_port_ret/(np.std(ew_port_ret)*sqrt(252)*100))


plt.plot(rp_port_ret_index.fillna(1), color='b', label= 'rp')
plt.plot(ew_port_ret_index.fillna(1), color='r', label='ew')
plt.legend()
plt.show()