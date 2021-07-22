from google.cloud import bigquery
from google.oauth2 import service_account

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf



from scipy import stats

from pmdarima import auto_arima

import warnings
warnings.filterwarnings('ignore')

class HumanitarianCrisisModel:
    
    
    def __init__(self, country_code, frequency='d', forecast_column='HumanitarianCrisisProportions', event_count=10000):
        self.country_code = country_code
        self.frequency = frequency
        self.event_count = event_count
        self.forecast_column = forecast_column
        self.query_job = self.run_query()
        self.country_df = self.convert_query_to_df()
        self.labeled_df = self.label_df()
        self.clean_df = self.clean_df()
        self.humanitarian_crisis_ts = self.group_df()
        self.country_name = self.get_country_name()
        self.residual = self.get_residual()
        self.best_order = self.run_auto_arima()[1]
        self.arima_model = self.run_best_ARIMA_model()[0]
        
        
        
    def run_query(self):
                
        credentials = service_account.Credentials.from_service_account_file(
            '../keys/graphic-charter-320020-b542ced38875.json', scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

        client = bigquery.Client(credentials=credentials, project=credentials.project_id)

        query = f"""
        SELECT SQLDATE, EventCode
        FROM `gdelt-bq.gdeltv2.events` 
        WHERE ActionGeo_CountryCode = @country
        AND SQLDATE > 20160101
        ORDER BY SQLDATE DESC
        LIMIT @event_count
        """

        # parameterized query that extract event rows from the GDELT dataset in BigQuery
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("country", "STRING", self.country_code),
                bigquery.ScalarQueryParameter("event_count", "INT64", self.event_count),

            ]
        )
        query_job = client.query(query, job_config=job_config) 
        
        return query_job
    
    def convert_query_to_df(self):
        
        # converts the query into a dataframe
        country_df = self.query_job.to_dataframe()
        
        return country_df
    
    def label_df(self):
        # determines which events should be labeled as humanitarian crisis events
        humanitarian_crisis_events = ['0233', '0243', '0333', '0343', '0243', '073', '074', '0833', '0861', '0863',
                                  '092', '094', '103', '1033', '1043', '1122', '1124', '1233', '137', '138', 
                                  '1382', '1383', '1384', '1385', '1413', '1423', '1451', '1452', '1453', '1454', 
                                  '175', '18', '180', '181', '182', '1821', '1822', '1823', '183', '1831', '1832', 
                                  '1833', '184']

        self.country_df['IsHumanitarianCrisis'] = self.country_df.EventCode.isin(humanitarian_crisis_events)
        labeled_df  = self.country_df
        return labeled_df
    
    
    
    def clean_df(self):

        # Structure the dataframe; set the event date as the index
        self.labeled_df.SQLDATE = pd.to_datetime(self.labeled_df.SQLDATE, format='%Y%m%d', errors='ignore')
        self.labeled_df = self.labeled_df.set_index('SQLDATE')
        clean_df = self.labeled_df
        
        return clean_df
    
    def group_df(self):

        # Group the events by the given frequency
        country_df_grouped = self.clean_df['IsHumanitarianCrisis'].groupby(pd.Grouper(freq=self.frequency)).agg(['sum','count','mean'])
        country_df_grouped.columns = ['HumanitarianCrisisEvents', 'TotalEvents', 'HumanitarianCrisisProportions']
        humanitarian_crisis_ts = country_df_grouped[self.forecast_column].dropna()
#         humanitarian_crisis_ts = humanitarian_crisis_ts[humanitarian_crisis_ts.index > '2016-01-01']
        
        return humanitarian_crisis_ts
    
    def get_country_name(self):

        # Lookup the name of the given country code
        lookup_df = pd.read_csv('../data/country_code_lookup.csv')
        country_name = lookup_df[lookup_df['Country Code'] == self.country_code].Country.values[0]
        
        return country_name

    def plot_original_data(self):
        # plot the original data
        plt.figure(figsize=(14, 4))
        plt.figtext(.5,.9,f'Humanitarian Crisis in {self.country_name}\n', fontsize=20, ha='center')
        plt.plot(self.humanitarian_crisis_ts, color='#D2042D') 
        plt.ylabel(f'{self.forecast_column}');
        
        
    def stationarity_test(self, ts, window=30, title='Stationarity Test'):
        # performn a Dickey-Full Test on the Time Series 
        print('Results of Dickey-Fuller Test:')
        dftest = adfuller(ts, autolag = 'AIC')

        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '# Lags Used', '# of Obs Used'])
        for k, v in dftest[4].items():
            dfoutput[f'Critiical Values: {k}'] = v

        print(dfoutput)

        rollmean = ts.rolling(window=window).mean()
        rollstd = ts.rolling(window=window).std()
        
        # plot the Time Series data against the rolling mean and the std to visualize stationarity
        plt.figure(figsize=(14, 4))
        original = plt.plot(ts, color = '#D2042D', label = 'Original')
        mean = plt.plot(rollmean, color = '#0092F4', label = 'Rolling Mean')
        std = plt.plot(rollstd, color = '#99d3fb', label = 'Rolling Std')
        plt.ylabel(f'{self.forecast_column}')
        plt.legend(loc='best')
        plt.figtext(.5,.9,f'{title}\n', fontsize=20, ha='center')
        
        
    def decompose_ts(self):
        
        decomposition = seasonal_decompose(self.humanitarian_crisis_ts, period=7)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        
        # plot the trend and seasonality of data using seasonal decompose method from statsmodels
        plt.figure(figsize=(14,12))
        plt.subplot(411)
        plt.plot(self.humanitarian_crisis_ts, label = 'Original', color = '#D2042D')
        plt.legend()
        plt.subplot(412)
        plt.plot(trend, label = 'Trend', color = '#99d3fb')
        plt.legend()
        plt.subplot(413)
        plt.plot(seasonal, label = 'Seasonal', color = '#0092F4')
        plt.legend()
        plt.subplot(414)
        plt.plot(self.residual, label = 'Residual', color = '#003a62')
        plt.legend(loc='upper right')
        
    def get_residual(self):
        decomposition = seasonal_decompose(self.humanitarian_crisis_ts, period=7)
        residual = decomposition.resid
        residual.dropna(inplace=True)

        return residual

            
        
    def plot_residual(self):
        
        (mu, sigma) = stats.norm.fit(self.residual)
        
        plt.figure(figsize=(10,6))
        plt.hist(self.residual, color='#003a62')
        plt.legend([f'Normal dist. ($\mu=$ {mu:.2f} and $\sigma=$ {sigma:.2f} )'], loc='best')
        plt.ylabel('Frequency')
        plt.figtext(.5,.9,f'Residual distribution\n', fontsize=20, ha='center')
        
    def run_auto_arima(self):
        
        stepwise_fit = auto_arima(self.humanitarian_crisis_ts, trace=False, suppress_warnings=True)
        stepwise_fit.summary()
        
        return stepwise_fit.summary(), stepwise_fit.order
    
    def plot_pacf_acf(self):
    
        lag_pacf = pacf(self.humanitarian_crisis_ts, method='ols', nlags = 10 )
        lag_acf = acf(self.humanitarian_crisis_ts, nlags = 10)


        plt.figure(figsize=(8,8))
        plt.subplot(211)
        plt.plot(lag_pacf, color='#0075c3')
        plt.axhline(y=0, linestyle='--', color='grey')
        plt.axhline(y=-1.96/np.sqrt(len(self.humanitarian_crisis_ts)), linestyle='--', color='grey')
        plt.axhline(y=1.96/np.sqrt(len(self.humanitarian_crisis_ts)), linestyle='--', color='grey')
        plt.title('Partial Autocorrelation Function')


        plt.subplot(212)
        plt.plot(lag_acf, color='#0066ab')
        plt.axhline(y=0, linestyle='--', color='grey')
        plt.axhline(y=-1.96/np.sqrt(len(self.humanitarian_crisis_ts)), linestyle='--', color='grey')
        plt.axhline(y=1.96/np.sqrt(len(self.humanitarian_crisis_ts)), linestyle='--', color='grey')
        plt.title('Autocorrelation Function')
        
    

    def run_best_ARIMA_model(self, order=None):
        
        if order is None:
            model = ARIMA(self.humanitarian_crisis_ts, order=self.best_order)   
        else:
            model = ARIMA(self.humanitarian_crisis_ts, order=order)
        model = model.fit()
        model.summary()
        return model, model.summary()
    
    def predict(self, end_frequency=30):
        end=len(self.humanitarian_crisis_ts)
        
        pred = self.arima_model.predict(start=1, end=end+end_frequency)
        if pred.index[0] in [0,1]:
            pred = self.arima_model.predict()
                
        plt.figure(figsize=(14,4))
        plt.plot(self.humanitarian_crisis_ts, color='#D2042D', label='original')
        plt.plot(np.abs(pred), color='#0092F4', label='forecasted')
        plt.figtext(.5,.9,f'Humanitarian Crisis Forecast on {self.country_name}\n', fontsize=20, ha='center')
        plt.ylabel(f'{self.forecast_column}')
        plt.legend()
        
    def predict_with_ci(self, end_frequency=30):
        
        fig, ax = plt.subplots(figsize=(14,4))
        
        end=len(self.humanitarian_crisis_ts)
        pred = self.arima_model.predict(start=1, end=end+end_frequency)
        
        if pred.index[0] in [0,1]:
            self.arima_model.plot_predict(ax=ax)
        else:
            self.arima_model.plot_predict(end=end+end_frequency, ax=ax)
           
        plt.gca().get_lines()[1].set_color('#f6cdd5')        
        plt.figtext(.5,.9,f'Humanitarian Crisis Forecast on {self.country_name}\n', fontsize=20, ha='center')
        plt.ylabel(f'{self.forecast_column}')
        plt.axhline(y=0.2, linestyle='--', color='black')

        
        
        

