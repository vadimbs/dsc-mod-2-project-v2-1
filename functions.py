import numpy as np
import matplotlib.pyplot as plt



import warnings
warnings.filterwarnings('ignore')


def print_reduce_perc(old_df,new_df):
    """
    Print out difference between dataframes
    """
    old_df_len = len(old_df)
    new_df_len = len(new_df)
    reduce = round(100 - (new_df_len * 100 / old_df_len), 2)
    display(f'Dataframe length -  before: {old_df_len}, after: {new_df_len}. Size reduction: {reduce}%')
    
    
    

def print_scatter(df,x_cols,y):
    """
    df -- dataframe
    x_cols -- list of column names
    y -- string with dependent variable name 
    """
    
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15,6))

    col = np.array_split(x_cols, 2)

    for i in range(len(axes)):
        for xcol, ax in zip(col[i],axes[i]):
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            df.plot(kind='scatter', x=xcol, y=y, ax=ax)

            
            


def print_hist(df, bins='auto'):
    """
    df -- Dataframe
    """
    
    fig = plt.figure(figsize = (15,15))
    ax = fig.gca()
    df.hist(ax = ax, bins=bins);

    

# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold

# def print_cross_val_score(df,continious,categorical,outcome):
#     """
    
#     """
    
#     df_ohe = pd.get_dummies(df[categorical], columns=categorical, drop_first=True)
#     preprocessed = pd.concat([df[continious], df_ohe], axis=1)

#     X = preprocessed.drop(outcome, axis=1)
#     y = preprocessed[outcome]

#     cross_validation = KFold(n_splits=10, shuffle=True, random_state=1)

#     regression = LinearRegression()
#     baseline = np.mean(cross_val_score(regression, X, y, cv=cross_validation, n_jobs=-1))
#     print(baseline)

    
        
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
import scipy.stats as stats
plt.style.use('ggplot')

def print_linear_model_summary(df,x_cols,y):
    """
    df -- dataframe
    y -- string with dependent variable name 
    x_cols -- list (or list of lists) with predictor column names
    """

    x_cols = sum(x_cols, []) #flatten
    print(x_cols)
    predictors = '+'.join(x_cols)
    formula = y + '~' + predictors
    model = ols(formula=formula, data=df).fit()
    display(model.summary())
    
    fig = plt.figure(figsize=(15,8))
    fig = sm.graphics.plot_regress_exog(model, y, fig=fig)
    plt.show()
    
    
    
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

def print_RFE(X, y, n=3):
    """
    X -- Dataframe with features (predictors)
    y -- Dataframe with dependent variable 
    n -- number features to select
    """
    
    linreg = LinearRegression()
    selector = RFE(linreg, n_features_to_select=n)
    selector = selector.fit(X, y)
    display(X.loc[:,selector.support_].columns)
    
    
    
import statsmodels.api as sm

def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    """ 
    Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax() ## I change for python 3.7
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included



def log_norm(df,cols):
    """
    Replace cols with log transformed and normilized ones
    
    df --- Dataframe
    cols --- columns to process (continious data)
    """
    
    for col in cols: 
        col_log = np.log(df[col])    
        df[col] = (col_log - col_log.mean()) / col_log.std()
        
    return df



def normalize(df,cols):
    """
    Replace cols with normilized ones
    
    df --- Dataframe
    cols --- columns to process (continious data)
    """

    for col in cols:    
        df[col] = (df[col] - df[col].mean()) / df[col].std()
        
    return df



def print_comparison(*dfs, title=False, x_label=False, alpha=0.4):
    """
    Prints dependency between 2 predictors and outcome
    """
    
    plt.figure(figsize=(16,8))

    for df in dfs:
        
        regression = LinearRegression()
        vals = df.iloc[:,:1].values.reshape(-1, 1)
        regression.fit(vals, df.iloc[:,1:2])
        pred = regression.predict(vals)

        print(f'{df.columns[0]} coefficient: {regression.coef_[0][0]}')

        plt.scatter(df.iloc[:,:1], df.iloc[:,1:2], alpha=alpha, label = df.columns[0], s=2)
        plt.plot(df.iloc[:,:1], pred, linewidth=4, c='white') # outline
        plt.plot(df.iloc[:,:1], pred, linewidth=2)
            

    if title:
        plt.title(title)
    else:
        plt.title(f'Dependence between {dfs[0].columns[1]} and '+ ', '.join(f'{df.columns[0]}' for df in dfs))
    plt.ylabel(dfs[0].columns[1])
    
    if x_label:
        plt.xlabel(x_label)
    else:
        plt.xlabel(dfs[0].columns[0])

    plt.legend();




    