# HumanitarianCrisisPrediction



## Introduction 
As a global citizen, it is quite atrocious and emotional to observe or witness the deprivation of basic human rights such as clean water, food, sanitation, education, healthcare access, and freedom from any people in the world in the 21st century. According to the United Nations Office for the Coordination of Humanitarian Affairs (UN OCHA), it is expected that an estimated 235 million people will need humanitarian aid and protection in 2021. This roughly translates to 1 in 33 people, which is an increase from a 2020 reported ratio of 1:45. The U.N. alone will require approximately $35 billion to provide humanitarian assistance to half the people in crisis across 56 countries. The aim of this project is to use a machine learning approach for predicting humanitarian crisis occurrence for days, weeks, or months into the future. 
 
The vision of this project is to use machine learning solutions to address the challenging problems that underprivileged people face in terms of receiving or restoring their basic human rights and dignity. 
 
## Research Question
Can one have an insight into the future state of the level of humanitarian crisis presence in a particular country based on historical information?
 
## Research Purpose
Early visibility or an indication of a possible humanitarian crisis in the future will allow:
1. IGOs & NGOs to closely monitor and advocate for high-risk countries 
2. Non-profit organizations to allocate their budget according to the measure of the severity in each region
3. Activists to gauge the consequences of political or economic decisions on the well-being of the general population of the particular country
4. Policymakers to develop robust and effective regulations for crisis prevention and protection of human rights.
 
 
## Data Description
In this paper, I analyze worldwide events that were recorded in the Global Database of Events, Language, and Tone (GDELT) project. The GDLET project, powered by Google Ideas, is the largest public dataset in the world that provides real-time information and metadata from the world’s news media entities dating as far back as 1979. It uses Conflict and Mediation Event Observations (CAMEO) coding for storing the data and gets updated every 15 mins with the latest print, broadcast, and web sources in over 100 languages and 300 categories. It is a massive network built to captures a societal scale behavior through connecting key actors including people, organizations, locations, classes, ethnicities, religious groups, themes, and so on. Essentially, it looks at any event that occurred around the world, the content of the event, the parties that were involved in the event, and the feelings the event evoked every single minute. The two main databases the platform provides are event records and themes and emotions measurement from every news article. For humanitarian crisis study, the focus will be performing analysis on the GDELT 2.0 Event Database.
 
Considering the database stores trillions of data points, Google’s Big Query supplies the infrastructure to interact with the Big Data. As of July 16, 2021, the events database contains ~584 million rows and 60 columns. Much of the research, therefore, involves understanding the dataset from the official documentation such as user manuals, codebooks, lookup files, CAMEO code taxonomy. High level, the research shows the database holds critical information such as the date of when the event took place, the date it was added, the key attributes and characteristics of the two actors involved in the event, the breakdown of the various attributes of the event, and finally the georeferencing of the event. 
 
  
## Approach
The first step in assessing this time series data is labeling the class output as a humanitarian crisis event (IsHumanitarianCrisis = 1) or not (IsHumanitarianCrisis = 0). To determine the proper labeling requires understanding the definition of a humanitarian crisis. The Humanitarian Coalition, a coalition of Canada’s leading non-governmental organizations, defines a humanitarian crisis as “a singular event or a series of events that are threatening in terms of health, safety or well-being of a community or large group of people.” Based on this definition, I took the conservative approach of classifying the following event codes as an indication of a humanitarian crisis. 

Another notable manipulation of the data was to group the original dataset per a given frequency (i.e., day(s) or month(s)) to create a table that shows the total number of events that were reported in the given timeframe, and of those the total number of humanitarian crisis events in that same timeframe. The primary reason behind taking the proportion is to evaluate if the frequency of occurrence across different countries and different time periods. In other words, comparing absolute values may not give the best result with a lower volume of news data obtained in the earlier years relative to recent times. The other important underlying strategy with this decision is to tackle the problem through a time series regression framework. 
 
The country I selected to build the initial model and assess the error metrics was Yemen. International Rescue Committee (IRC) ranks Yemen as the number 1 country on the watchlist for a devastating humanitarian crisis issue in 2021. Yemen remains the world's worst humanitarian crisis with over 80% of the population in need of aid (24.3 million out of 29.8 million) and  2.3 million children under the age of five projected to suffer severe malnutrition. However, later on, the test was extended to other countries representing various rankings on the crisis spectrum. 
 
 
## Analysis
The main attributes to keep in mind when attempting to solve the research question posed above are: 
Rare-event prediction (with imbalance class samples): crises like financial, economic, technological, natural, etc.. are regarded as rare occurrences, hence one has to be aware that the positive class represents the minority class
Performing data exploration on Big Data can be challenging. Essentially, one has to be cognizant of the curse of dimensionality. The models I used are expensive to train and require the use of tools that accommodate a large set of data
The use of domain knowledge in the EDA step. The problem definition and data analysis aspects of the project were guided by prior knowledge around foreign relations and newly acquired knowledge with GDLET documentations.
 
The two models I considered for predicting future values are ARIMA (autoregressive integrated moving average) and LSTM (long short term memory - a class of RNN, recurrent neural networks). Without violating the K-Folds principle, the time series cross-validation and prediction were done on real-time data with the entire available dataset at any given point in time. Additionally, I selected the root mean squared error (RMSE) error metrics to evaluate the performance of the two models. 
 
## ARIMA Model Results
### Stationarity Test
When training the ARIMA model, the first validation check is conducting a hypothesis test to determine whether or not the data is stationary, meaning showing that the probability does not depend on when one begins observing the data. Stationarity also means that it is a random process that generated the data. I did so by visualizing the daily proportion of humanitarian crisis events in Yemen since July 2017. In the visualization, the consistency in the rolling 30 days means and standard deviations provides evidence for supporting the stationarity of the data. For a more accurate assessment, I also implemented the Dickey-Fuller test to verify stationarity via hypothesis testing. With such a small p-value of 3.620840e-22, the test confirms rejecting the null hypothesis (using 5% Critical Value-alpha). 
 
### Seasonal Decomposition
Furthermore, I decomposed the time series data, which separated the trend, seasonality, and residual components. When performing a decomposition, I used a weekly cycle in the data to get the best split of seasonality, and residuals. The data series shows no trend. However, as specified there is a weekly pattern in the seasonality, which I will keep in mind when running auto arima grid-search to tune the hyperparameters. Residuals, values leftover from trend, and seasonality also demonstrate stationarity. Observation on the residuals is as such 1) it is normally distributed and 2) a Dickey-Fuller test p-value of 2.923403e-27 ascertains stationarity once again. 
 
### PACF and ACF
The ACF plots coefficients of correlation between a time series and its lagged values while PACF draws the partial correlation between a time series and its lagged values. With the stationary data, we can take the plots into consideration to identify the AR and MA orders. 
 
### Hypertunning (p, d, q)
There are three parameters in the ARIMA model that require estimation through an iterative trial. I used the pmdarima library to find the best ARIMA model optimizes for a given information_criterion on AIC, Akaike Information Criterion. My result for the Yemen trained model was (2, 1, 1), which I used for predicting 30 days into the future. 
 
## LSTM Model Results  
### Cross-Validation
I used MinMaxScaler to normalize the data before I proceeded to split the train/test (0.75/0.25). Considering neural networks can take a long time to run, often require more data to train relative to other models, and have various parameters to tune, I undertook the simplest approach to train on the Yemen time series. 
 
### Hidden Layers
For a head-to-head comparison against the ARIMA model, I used a 30 days window as my input shape. I trained the model on 4 input units, and relu activation with 1 output compiled on adam optimizer. The final results failed to capture the heavy daily fluctuation observed in the actual data. 
 
 
## Conclusion

 
 
## Future Work





Resources: 
 1) "What Is a Humanitarian Crisis", Humanitarian Coalition, Retrieved on 6 May 2013.
 2) http://data.gdeltproject.org/documentation/CAMEO.Manual.1.1b3.pdf
 3) https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax
 4) https://www.dataindependent.com/pandas/pandas-to-datetime/
 5) https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71
 6) https://htmlcolorcodes.com/colors/shades-of-red/ 