# fast-prophet
faster implementation of Facebook's Prophet algorithm for time series forecasting


For more details on the logic behind this code check out my series of posts in towards data science:

https://towardsdatascience.com/how-to-run-facebook-prophet-predict-x100-faster-cce0282ca77d


The code for faster predict can be used to replace Prophet's confidence interval, which takes ~1 second, to reduce the runtime to ~10ml.
The code for faster fit is mostly a POC and needs more work for productization to allow non yearly seasonalities and a few other features from Prophet. The code includes a scipt that compares Prophet results with the new method.
