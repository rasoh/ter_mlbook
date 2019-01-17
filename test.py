import pandas as pd # Import the library and give a short alias: pd
rent = pd.read_csv("rent-ideal.csv")
'''
print(rent.head(5))
prices = rent['price']
avg_rent = prices.mean()
print(f"Average rent is ${avg_rent:.0f}")
bybaths = rent.groupby(['bathrooms']).mean()
bybaths = bybaths.reset_index() # overcome quirk in Pandas
print(bybaths[['bathrooms','price']]) # print just num baths, avg price

import matplotlib.pyplot as plt

bybaths.plot.line('bathrooms','price', style='-o')
plt.show()
'''
#X, y = rent[['bedrooms','bathrooms','latitude','longitude']], rent['price']
# print(type(X), type(y))

from sklearn.ensemble import RandomForestRegressor
'''
rf = RandomForestRegressor(n_estimators=10) # use 10 trees
rf.fit(X, y)

unknown_x = [2, 1, 40.7957, -73.97] # 2 bedrooms, 1 bathroom, ...
predicted_y = rf.predict([unknown_x])
print(predicted_y)
'''
from sklearn.metrics import mean_absolute_error

'''
predictions = rf.predict(X)
e = mean_absolute_error(y, predictions)
ep = e*100.0/y.mean()
print(f"${e:.0f} average error; {ep:.2f}% error")

print('just location features')
X, y = rent[['latitude','longitude']], rent['price']
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X, y)
location_e = mean_absolute_error(y, rf.predict(X))
location_ep = location_e*100.0/y.mean()
print(f"${location_e:.0f} average error; {location_ep:.2f}% error")
'''
from sklearn.model_selection import train_test_split


X, y = rent[['bedrooms','bathrooms','latitude','longitude']], rent['price']
# 20% of data goes into test set, 80% into training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
'''
rf = RandomForestRegressor(n_estimators=10)
rf.fit(X_train, y_train)
validation_e = mean_absolute_error(y_test, rf.predict(X_test))
print(f"${validation_e:.0f} average error; {validation_e*100.0/y.mean():.2f}% error")
'''

from rfpimp import *
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)
I = importances(rf, X_test, y_test)
print(I)
plot_importances(I)

# end 3.2.6