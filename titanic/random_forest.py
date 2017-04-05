"""
train a classifier with random forest.
"""
import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier


def arrange_data(data):
    """
    replace categories with one hot forms.
    drop useless columns.
    """
    # age to 3 categories: less than 16, greater than 15, unknown
    data['Age'].fillna(-2, inplace=True)
    data.loc[data['Age'] > 15, 'Age'] = -1
    data.loc[data['Age'] > 0, 'Age'] = 0
    data.loc[data['Age'] == -1, 'Age'] = 1
    data.loc[data['Age'] == -2, 'Age'] = 2

    # one hot Age
    dummies = pd.get_dummies(data['Age'])
    dummies.columns = ['Age_0', 'Age_1', 'Age_2']

    data = data.join(dummies)

    # one hot Embarked
    dummies = pd.get_dummies(data['Embarked'])
    dummies.columns = ['Embarked_S', 'Embarked_C', 'Embarked_Q']

    data = data.join(dummies)

    # one hot Pclass
    dummies = pd.get_dummies(data['Pclass'])
    dummies.columns = ['Pclass_1', 'Pclass_2', 'Pclass_3']

    data = data.join(dummies)

    # has family or not, one hot
    data['Family'] = data['Parch'] + data['SibSp']
    data.loc[data['Family'] > 0, 'Family'] = 1
    data.loc[data['Family'] == 0, 'Family'] = 0

    # one hot sex
    data.loc[data['Sex'] == 'female', 'Sex'] = 0
    data.loc[data['Sex'] == 'male', 'Sex'] = 1

    # drop all usless columns
    data = data.drop(['Cabin', 'Name', 'Ticket'], axis=1)
    data = data.drop(['Age'], axis=1)
    data = data.drop(['Embarked'], axis=1)
    data = data.drop(['Parch', 'SibSp'], axis=1)
    data = data.drop(['Pclass'], axis=1)

    return data


def load_data_train(path):
    """
    load training data from path.
    """
    data = pd.read_csv(path)

    # there are empty cells in the 'Embarked' column in the training set.
    # most of the passengers came from 'S'
    data['Embarked'].fillna('S', inplace=True)

    data = arrange_data(data)

    # split training set into ids, features and labels.
    ids = data['PassengerId']
    xs = data.drop(['PassengerId', 'Survived'], axis=1)
    ys = data['Survived']

    return ids, xs, ys


def load_data_test(path):
    """
    load test data from path.
    """
    data = pd.read_csv(path)

    # there are empty cells in the 'Fare' column in the test set.
    # mean of 'Fare' is ~35.
    data['Fare'].fillna(35.0, inplace=True)

    data = arrange_data(data)

    # split test set into ids and features.
    ids = data['PassengerId']
    xs = data.drop('PassengerId', axis=1)

    return ids, xs


def main():
    """
    do the training and predicting.
    """
    path_home = os.path.expanduser('~')
    path_train = os.path.join(path_home, 'datasets/kaggle/titanic/train.csv')
    path_test = os.path.join(path_home, 'datasets/kaggle/titanic/test.csv')

    ids_train, xs_train, ys_train = load_data_train(path_train)
    ids_test, xs_test = load_data_test(path_test)

    random_forest = RandomForestClassifier(n_estimators=100)

    random_forest.fit(xs_train, ys_train)

    ys_test = random_forest.predict(xs_test)

    # output for kaggle.
    results = pd.DataFrame({
        'PassengerId': ids_test,
        'Survived': ys_test})

    results.to_csv('results_titanic_random_forest.csv', index=False)


if __name__ == '__main__':
    main()
