# Importing the libraries
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import pickle

data = pd.read_csv("Extension Data.csv")

# Removing outliers
z = np.abs(stats.zscore(data))
threshold = 3
dt = data[(z < 3).all(axis=1)]

# Splitting Training and Test Set
# Since we have a very small dataset, we will train our model with all availabe data.

# Creating var's
X = dt.iloc[:, :3].values
y = dt.iloc[:, -1].values

# Scaling the Data
scaler = StandardScaler()
# transform data
scaler.fit_transform(X)

# Decision Tree regressor
d_tree = DecisionTreeRegressor()
d_tree.fit(X, y)

# Saving model to disk
pickle.dump(d_tree, open('model.pkl', 'wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[2, 9, 6]]))

