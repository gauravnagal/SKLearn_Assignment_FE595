'''
Using the Linear Regression class and the Boston housing data set, determine which element has 
the most influence on the price of a house in Boston. Be aware that the element with the "most" 
influence might also be the element with the "most negative" influence.
'''

from sklearn.datasets import load_boston
from sklearn import linear_model
import numpy as np
import pandas as pd

def most_influence():
    
    # load data to 'boston' variable
    boston = load_boston()
    
    # feature names in boston dataset
    print(boston.feature_names)
    # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']

    # perform linear regression
    lr_model = linear_model.LinearRegression()
    lr_fit = lr_model.fit(boston.data, boston.target)
    lr_coeffs = list(lr_fit.coef_)
    
    print(lr_coeffs)
    # [-0.10801135783679539, 0.04642045836687953, 0.020558626367068917, 2.6867338193448442, -17.766611228299986, 
    # 3.8098652068092282, 0.0006922246403431768, -1.47556684560025, 0.30604947898516427, -0.012334593916574021, 
    # -0.9527472317072921, 0.00931168327379375, -0.5247583778554881]
    
    coeffs_df = pd.concat([pd.DataFrame(list(boston.feature_names)), pd.DataFrame(np.transpose(lr_coeffs))], axis = 1)
    coeffs_df.columns = ['Attribute', 'Coefficient']
    print(coeffs_df)
    #       Attribute  Coefficient
    # 0       CRIM    -0.108011
    # 1         ZN     0.046420
    # 2      INDUS     0.020559
    # 3       CHAS     2.686734
    # 4        NOX   -17.766611
    # 5         RM     3.809865
    # 6        AGE     0.000692
    # 7        DIS    -1.475567
    # 8        RAD     0.306049
    # 9        TAX    -0.012335
    # 10   PTRATIO    -0.952747
    # 11         B     0.009312
    # 12     LSTAT    -0.524758

    print('The element with the most influence on the price of a house in Boston is {}.'
          .format(coeffs_df.loc[np.abs(coeffs_df['Coefficient']) == max(np.abs(coeffs_df['Coefficient'])),'Attribute'].iloc[0]))

if __name__ == '__main__':
    most_influence()
