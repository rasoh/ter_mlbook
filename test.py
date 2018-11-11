
import pandas as pd # Import the library and give a short alias: pd

rent = pd.read_csv("rent-ideal.csv")
#print(rent.head(5))

""" prices = rent['price']
avg_rent = prices.mean()
print(f"Average rent is ${avg_rent:.0f}")

bybaths = rent.groupby(['bathrooms']).mean()
bybaths = bybaths.reset_index() # overcome quirk in Pandas
print(bybaths[['bathrooms','price']]) # print just num baths, avg price """

#import matplotlib.pyplot as plt
#bybaths.plot.line('bathrooms','price', style='-o')
#plt.show()

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


#X, y = rent[['bathrooms']], rent['price']
#print(type(X), type(y))
# rf = RandomForestRegressor()
# rf.fit(X, y)
#print( rf.predict([[0], [1]]) )
# predictions = rf.predict(X)
# e = mean_absolute_error(y, predictions)
# print(f"${e:.0f} average error; {e*100.0/y.mean():.2f}% error")

""" X, y = rent[['bedrooms','bathrooms']], rent['price']
rf = RandomForestRegressor()
rf.fit(X, y)
predictions = rf.predict(X)
e = mean_absolute_error(y, predictions)
print(f"${e:.0f} average error; {e*100.0/y.mean():.2f}% error") """

""" X, y = rent[['latitude','longitude']], rent['price']
rf = RandomForestRegressor()
rf.fit(X, y)
e = mean_absolute_error(y, rf.predict(X))
print(f"${e:.0f} average error; {e*100.0/y.mean():.2f}% error") """

""" X, y = rent[['bedrooms','bathrooms','latitude','longitude']], rent['price']
rf = RandomForestRegressor()
rf.fit(X, y)
e = mean_absolute_error(y, rf.predict(X))
print(f"${e:.0f} average error; {e*100.0/y.mean():.2f}% error") """

from sklearn.model_selection import train_test_split

""" X, y = rent[['bedrooms','bathrooms','latitude','longitude']], rent['price']
# 20% of data goes into test set, 80% into training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 

rf = RandomForestRegressor()
rf.fit(X_train, y_train)

e = mean_absolute_error(y_test, rf.predict(X_test))
print(f"${e:.0f} average error; {e*100.0/y.mean():.2f}% error") """

from sklearn.model_selection import cross_val_score

X, y = rent[['bedrooms','bathrooms','latitude','longitude']], rent['price']

""" k = 5
cv_err = cross_val_score(RandomForestRegressor(),
                         X, y, cv=k,
                         scoring='neg_mean_absolute_error')
m_err = -cv_err.mean()
std_err = cv_err.std()
print(f"${m_err:.0f} average error +/-${2*std_err:.2f}; {m_err*100.0/y.mean():.2f}% error")
 """

def validate(model):
    cv_err = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    m_err = -cv_err.mean()
    std_err = cv_err.std()
    print(f"${m_err:.0f} average error +/-${2*std_err:.2f}; {m_err*100.0/y.mean():.2f}% error")

""" rf = RandomForestRegressor(n_estimators=100)
validate(rf) """

## What the model says about the data

from rfpimp import *
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X, y)
I = importances(rf, X, y)
print(I)