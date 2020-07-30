import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('Social_Network_Ads.csv')

from sklearn.model_selection import train_test_split

X=df.drop('Purchased',axis=1)
y=df['Purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=1000,criterion='entropy',random_state=0)

rfc.fit(X_train,y_train)

pickle.dump(rfc,open('model_titanic.pkl','wb'))

model = pickle.load(open('model_titanic.pkl','rb'))
print(model.predict([[20,30000]]))
