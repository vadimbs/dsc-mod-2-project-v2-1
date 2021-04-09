The main ojective of this project is to build the most parsimonious model which will be able to predict the price of a property given only few predictors.



# Contents
* #### EDA
  * Import and explore data
  * Initial data visualization
  * Data cleaning
  * Remove outliers
  * Clean data visualization
  * Normalize data
  * Plot data on map
* #### Questions
  * What is dependence between `sqft_living` vs `price` and `sqft_lot` vs `price`?
  * Is there a difference in price based on the renovation status within the last 10 years or 10+ years with respect to square footage of living area?
  * Does the price depend on the number of bedrooms?
  * How did the price change over the time for the subset of data for which this data is available?
* #### Build Model
  * Linear regression using Statsmodels
  * Cross validation
  * Linear regression using scikit-learn
  * Check for normality
  * Check for heteroscedascity
  * Check for multicollinearity
  * Visualizing prediction accuracy
  * Model with log transformed and normalized data
* #### Recommendations
  * Zipcode is the most significant predictor of the price in this dataset. Zipcode is responsible for 52% of the R^2. As such, zipcode should be treated as the most essential piece of information about the property.
  * Surprizingly, renovation status was not a significant predictor of the price, however, renovation status was available only for a small subset of data. It is reasonable to expect revonation status to be a significant and positive predictor of the property price.
  * Properties with large square footage did not fit well into the model. Thus, this model should only be used to predict price of medium size. (Limitation)
  * In addition, presence of "view" was not found to be significant. Interpretation for this variable was missing. Data for variable 'view' was available for only a small subset of data.<br><br><br><br>
  
### Model R^2 is 0.792 with `zipcode` and `sqft_living` as predictors

![alt text](model.png?raw=true)