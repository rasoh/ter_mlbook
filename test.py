'''
import pandas as pd # Import the library and give a short alias: pd
rent = pd.read_csv("rent-ideal.csv")

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

#X, y = rent[['bedrooms','bathrooms','latitude','longitude']], rent['price']
# print(type(X), type(y))

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=10) # use 10 trees
rf.fit(X, y)

unknown_x = [2, 1, 40.7957, -73.97] # 2 bedrooms, 1 bathroom, ...
predicted_y = rf.predict([unknown_x])
print(predicted_y)

from sklearn.metrics import mean_absolute_error


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

from sklearn.model_selection import train_test_split


X, y = rent[['bedrooms','bathrooms','latitude','longitude']], rent['price']
# 20% of data goes into test set, 80% into training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 

rf = RandomForestRegressor(n_estimators=10)
rf.fit(X_train, y_train)
validation_e = mean_absolute_error(y_test, rf.predict(X_test))
print(f"${validation_e:.0f} average error; {validation_e*100.0/y.mean():.2f}% error")


from rfpimp import *
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)
I = importances(rf, X_test, y_test)
print(I)
plot_importances(I)

'''

# 3.3
'''
from sklearn.datasets import load_breast_cancer
import pandas as pd

cancer = load_breast_cancer()

X = cancer.data
y = cancer.target
df = pd.DataFrame(X, columns=cancer.feature_names)

features = ['radius error', 'texture error', 'concave points error',
            'symmetry error', 'worst texture', 'worst smoothness',
            'worst symmetry']
df = df[features] # select just these features
# print("target[0:30] =", y[0:30]) # show 30 values of malignant/benign target
# df.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.15)

from sklearn.ensemble import RandomForestClassifier

cl = RandomForestClassifier(n_estimators=300)
cl.fit(X_train, y_train)
validation_e = cl.score(X_test, y_test)
print(f"{validation_e*100:.2f}% correct")

from rfpimp import *
I = importances(cl, X_test, y_test)
plot_importances(I)
'''
# 3.4

import pandas as pd
import matplotlib.pyplot as plt

addr640 = pd.read_csv("640.csv")

# print(addr640.digit.values)
addr640 = addr640.drop('digit', axis=1) # drop digit column

""" six_img_as_row = addr640.iloc[0].values  # digit '6' is first row
img28x28 = six_img_as_row.reshape(28,28) # unflatten as 2D array
plt.imshow(img28x28, cmap='binary')
plt.show()

six_img_as_row[six_img_as_row>0] = 1  # convert 0..1 to 0 or 1
six_img_as_row = six_img_as_row.astype(int)
img28x28 = six_img_as_row.reshape(28,28)
s = str(img28x28).replace(' ','')     # remove spaces
print(s) """

digits = pd.read_csv("mnist-10k-sample.csv")
images = digits.drop('digit', axis=1) # get just pixels
targets = digits['digit']             # get just digit value
""" 
fig, axes = plt.subplots(10, 5, figsize=(4, 6.5)) # make 10x5 grid of plots

for i, ax in enumerate(axes.flat):
    img_as_row = images.iloc[i].values
    img28x28 = img_as_row.reshape(28,28)
    ax.axis('off') # don't show x, y axes
    ax.imshow(img28x28, cmap='binary')
    ax.text(0, 8, targets[i], color='#313695', fontsize=18)
plt.show() 

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

cl = RandomForestClassifier(n_estimators=900, n_jobs=-1)
cl.fit(images, targets)
pred = cl.predict(addr640)
print(pred)

import numpy as np;
np.set_printoptions(precision=3)

digit_values = range(10)
prob = cl.predict_proba(addr640)
prob_for_2nd_digit = prob[1]
print(prob_for_2nd_digit)

pred_digit = np.argmax(prob_for_2nd_digit)
print("predicted digit is", pred_digit)

pred_digit = np.argmax(prob_for_2nd_digit)
bars = plt.bar(digit_values, prob_for_2nd_digit, color='#4575b4')
bars[pred_digit].set_color('#fdae61')
plt.xlabel("predicted digit")
plt.xticks(digit_values)
plt.ylabel("likelihood 2nd image\nis a specific digit")
plt.show() 

fours = images[targets==4] # find all "4" images

fig, axes = plt.subplots(15, 8, figsize=(4,6.5))
for i, ax in enumerate(axes.flat):
    img = fours.iloc[i,:].values.reshape(28,28)
    ax.axis('off')
    ax.imshow(img, cmap='binary') """
""" 
X_train, X_test, y_train, y_test = \
    train_test_split(images, targets, test_size=.2)

cl = RandomForestClassifier(n_estimators=900, n_jobs=-1)
cl.fit(X_train, y_train)
rfaccur = cl.score(X_test, y_test)
print(rfaccur)

from sklearn.linear_model import LogisticRegression

# create linear model
lm = LogisticRegression(solver='newton-cg', multi_class='multinomial')
lm.fit(X_train, y_train)

lmaccur = lm.score(X_test, y_test)
print(lmaccur) """

import numpy as np
X_train = np.array([1,2,3,4,5]).reshape(5,1) # 5 rows of 1 column
y_train = np.array([1,2,3,4,5])              # 1 column
X_test = np.array([6]).reshape(1,1)          # 1 row of 1 column

from sklearn.linear_model import LinearRegression   
lm = LinearRegression()
lm.fit(X_train,y_train)
print("y =", lm.predict(X_test))

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train,y_train)
print("y =", rf.predict(X_test) )
