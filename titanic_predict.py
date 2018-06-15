# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

#------------------------------------------------------------#

def getDataFromCSV():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    return train_df, test_df


def dataProcess(train_df, test_df):
    #去除Ticket和Cabin两个变量
    train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
    test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
    combine = [train_df, test_df]
    print("------去掉无用的Ticket与Cabin------\n")

    #统计combine中的姓名前缀：Mr/Miss/Mis...
    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    print("------统计Title------\n+{0}".format(pd.crosstab(train_df['Title'], train_df['Sex'])))

    '''
    替换上一步中的一些title中的某些title
    Master->Master
    Mr->Mr
    Mile->Miss
    Ms->Miss
    Mme->Mrs
    其余->Rare
    '''
    for dataset in combine:
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',\
                                                     'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    print("------将Title进行转换------\n {0}".format(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()))

    '''
    把Title中的的内容转换为数值
    Mr->1
    Miss->2
    Mrs->3
    Master->4
    Rare->5
    nan->0
    '''
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    for dataset in combine:
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)
    print("------将Title数值化------\n {0}".format(train_df.head(5)))

    #现在去掉无用的Name和PassengerId
    train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
    test_df = test_df.drop(['Name'], axis=1)
    combine = [train_df, test_df]
    print("------去掉无用的Name与PassengerId------\n")

    #把性别转换为数值：female = 1， male = 0
    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)
    print("------将性别数值化------\n {0}".format(train_df.head(5)))

    #给Age为空的数据添加数据
    guessAge(combine)
    print("------给Age为空的数据添加数据------\n")
    print(train_df.info(), test_df.info())

    #添加一个变量AgeBand，按年龄将人群分为5类，最后再将这5类转化为数值0-4，最后再去掉AgeBand
    train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
    for dataset in combine:
        dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[dataset['Age'] > 64, 'Age'] = 4
    train_df = train_df.drop(['AgeBand'], axis=1)
    combine = [train_df, test_df]
    print("------将年龄数值化------\n {0}".format(train_df.head(5)))

    #添加一个新变量FamilySize，用于统计此人是一人还是家人同行，用变量IsAlone = 0/表示
    #最后在数据中去掉Parch/SibSp/FimalySize
    for dataset in combine:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    for dataset in combine:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1   #家庭成员只有1个，就标记IsAlong = 1
    train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    combine = [train_df, test_df]
    print("------处理Parch/SibSp------\n {0}".format(train_df.head(5)))

    #将Pclass与Age合并，创造新的变量Age*Pclass
    for dataset in combine:
        dataset['Age*Class'] = dataset.Age * dataset.Pclass
    print("------合并Age*Pclass------\n {0}".format(train_df.head(5)))

    #处理Embark，先填充nan为S，最后将其数值化S->0  C->1 Q->2
    freq_port = train_df.Embarked.dropna().mode()[0]
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    print("------处理Embark并数值化------\n {0}".format(train_df.head(5)))

    #处理Fare
    #先给test的Fare的nan用中位数来填充，再分别对test和train进行处理，划分为4个等级并数值化，最后去掉中间变量
    test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
    train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
    for dataset in combine:
        dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
        dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)
    train_df = train_df.drop(['FareBand'], axis=1)
    combine = [train_df, test_df]
    print("------处理Fare并数值化------\n {0}".format(train_df.head(5)))

    return train_df, test_df


def guessAge(combine):
    '''
    给数据中Age为空的项预测年龄
    :param combine:
    :return:
    '''
    guess_ages = np.zeros((2, 3))
    for dataset in combine:
        for i in range(0, 2):
            for j in range(0, 3):
                guess_df = dataset[(dataset['Sex'] == i) & \
                                   (dataset['Pclass'] == j + 1)]['Age'].dropna()

                # age_mean = guess_df.mean()
                # age_std = guess_df.std()
                # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

                age_guess = guess_df.median()

                # Convert random age float to nearest .5 age
                guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

        for i in range(0, 2):
            for j in range(0, 3):
                dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), \
                            'Age'] = guess_ages[i, j]

        dataset['Age'] = dataset['Age'].astype(int)


if __name__ == "__main__":
    #get data
    train, test = getDataFromCSV()
    train_pro, test_pro = dataProcess(train, test)

    #train data
    X_train = train_pro.drop('Survived', axis=1)
    Y_train = train_pro['Survived']
    X_test = test_pro.drop('PassengerId', axis=1).copy()

    print(X_train.shape, Y_train.shape, X_test.shape)
    print("--------------Train Start...--------------\n")
    print("--------------Logistic Regression--------------")
    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    Y_pred = logreg.predict(X_test)
    acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
    print("准确度为：{0}".format(acc_log))

    print("--------------Support Vector Machine--------------")
    svc = SVC()
    svc.fit(X_train, Y_train)
    Y_pred = svc.predict(X_test)
    acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
    print("准确度为：{0}".format(acc_svc))

    print("--------------K-Nearest Neighbors--------------")
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
    print("准确度为：{0}".format(acc_knn))

    print("--------------Gaussian Naive Bayes--------------")
    gaussian = GaussianNB()
    gaussian.fit(X_train, Y_train)
    Y_pred = gaussian.predict(X_test)
    acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
    print("准确度为：{0}".format(acc_gaussian))

    print("--------------Perceptron--------------")
    perceptron = Perceptron()
    perceptron.fit(X_train, Y_train)
    Y_pred = perceptron.predict(X_test)
    acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
    print("准确度为：{0}".format(acc_perceptron))

    print("--------------Linear SVC--------------")
    linear_svc = LinearSVC()
    linear_svc.fit(X_train, Y_train)
    Y_pred = linear_svc.predict(X_test)
    acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
    print("准确度为：{0}".format(acc_linear_svc))

    print("--------------Stochastic Gradient Descent--------------")
    sgd = SGDClassifier()
    sgd.fit(X_train, Y_train)
    Y_pred = sgd.predict(X_test)
    acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
    print("准确度为：{0}".format(acc_sgd))

    print("--------------Decision Tree--------------")
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, Y_train)
    Y_pred = decision_tree.predict(X_test)
    acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
    print("准确度为：{0}".format(acc_decision_tree))

    print("--------------Random Forest--------------")
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, Y_train)
    Y_pred = random_forest.predict(X_test)
    random_forest.score(X_train, Y_train)
    acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
    print("准确度为：{0}".format(acc_random_forest))