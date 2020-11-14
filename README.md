I built a machine learning model predicting Joe Biden's results across the country, based on Hillary's results in 2016 and census data. It was on per-county basis (about 3100 counties altogether). For training the model I excluded the counties where fraud was suspected. I then tested the model on these counties. If Biden's result was larger than the model anticipated, it would've been suspicious.
Here are the results
state | county | prediction | result
------|--------|------------|--------
az | maricopa | 0.51 | 0.5
az |  pima |  0.59 |  0.59
ga |  cobb |  0.56  | 0.56
ga |  dekalb |  0.85 |  0.83
ga |  fulton |  0.74 |  0.73
ga |  gwinnett |  0.56 |  0.58
mi |  oakland |  0.55 |  0.56
mi |  wayne |  0.69 |  0.68
nv |  clark |  0.56 |  0.54
pa |  allegheny |  0.59 |  0.6
pa |  bucks |  0.52 |  0.52
pa |  montgomery |  0.65 |  0.63
pa |  philadelphia |  0.85 |  0.81
wi |  dane |  0.75 |  0.76
wi |  milwaukee |  0.69 |  0.69
pa |  delaware |  0.63 |  0.63

Technicals:  
Model mean absolute error, cross-validated: 0.015  
Data sources: page scraped decisiondeskhq.com, page scraped https://www.census.gov/quickfacts, https://github.com/tonmcg/US_County_Level_Election_Results_08-20   
SQLite database with census and results is in data/  
Regressor: xgb.XGBRegressor(objective='reg:logistic', n_estimators=300, learning_rate=0.1, max_depth=6)  
Hyper-parameters tuned on predicting 2016 Hillary's results on census data only.  
Samples weighted by population during training  
Has 3090 training instances, uses 40 features. Some counties are missing from the training data because they're missing census data.   
Most useful feature without Hillary's result: county's population density. Most useful feature with Hillary's result: bachelor's degree holders per capita.  
