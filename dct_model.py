'''
### read csv file

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
path = "/content/drive/MyDrive/Titanic-Dataset.csv"
data = pd.read_csv(path)
'''

import seaborn as sns
sns.heatmap(data.isna(), cbar = False)

# Drop PassengerId, Name, Ticket (spacific) and Cabin (alot na)
data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)

# fill Age with avg
data['Age'].fillna(data['Age'].mean(), inplace = True)

# drop na embarked
data.dropna(inplace = True)

sns.heatmap(data.isna(), cbar = False)

# convert sex column to is male >> 1 = male, 0 = female

data.rename(columns = {'Sex' : 'is male'}, inplace = True)
data['is male'] = np.where(data['is male'] == 'male', 1, 0)

# embarked

embarked = data['Embarked']
data.drop('Embarked', axis = 1, inplace = True)
data['Embarked = S'] = np.where(embarked == 'S', 1, 0)
data['Embarked = C'] = np.where(embarked == 'C', 1, 0)
data['Embarked = Q'] = np.where(embarked == 'Q', 1, 0)

# split test and train data

from sklearn.model_selection import  train_test_split
train_x, test_x, train_y, test_y = train_test_split(data.drop('Survived', axis = 1), data['Survived'], test_size=0.3, random_state = 555)

# model

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

clf = DecisionTreeClassifier()
clf.fit(train_x, train_y)

y_pred = clf.predict(test_x)

print(f"Accuracy: {accuracy_score(test_y, y_pred)}")
print("Classification Report:")
print(classification_report(test_y, y_pred))



