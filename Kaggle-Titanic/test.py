import pandas as pd


train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
train = train.loc[0:10, :]
test = test.loc[0:10, :]
train.drop(['PassengerId', 'Survived', 'Pclass', 'Name'], axis=1, inplace=True)
test.drop(['PassengerId', 'Pclass', 'Name'], axis=1, inplace=True)
train.loc[train['Age'] <= 100, 'Age'] = 0

train_test_data = [train, test]
for dataset in train_test_data:
    print(dataset)
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[dataset['Age'] > 62, 'Age'] = 4