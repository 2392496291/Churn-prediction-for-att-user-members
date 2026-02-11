import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
def dm01_数据预处理():
    data = pd.read_csv('./data/churn.csv')
    data.info()
    # 因为上述的Churn, gender是字符串类型, 我们对其做热编码(one-hot)处理.
    data = pd.get_dummies(data)
    data.info()
    print(data.head(10))
    data.drop(['gender_Male', 'Churn_No'], axis=1, inplace=True)
    print(data.head(10))
    data.rename(columns={'Churn_Yes':'flag'}, inplace=True)
    print(data.head(10))
    print(data.flag.value_counts()) 
def dm02_会员流失可视化情况():
    data = pd.read_csv('./data/churn.csv')
    data = pd.get_dummies(data)
    data.drop(['gender_Male', 'Churn_No'], axis=1, inplace=True)
    data.rename(columns={'Churn_Yes':'flag'}, inplace=True)
    print(data.flag.value_counts())
    print(data.columns) 
    # 参数x意思是: x轴的列名(是否是月度会员, 0 -> 不是会员, 1 -> 是会员)
    sns.countplot(data, x='Contract_Month', hue='flag')
    plt.show()
def dm03_逻辑回归模型训练评估():
    data = pd.read_csv('./data/churn.csv')
    data = pd.get_dummies(data)
    data.drop(['gender_Male', 'Churn_No'], axis=1, inplace=True)
    data.rename(columns={'Churn_Yes':'flag'}, inplace=True)
    x = data[['Contract_Month', 'PaymentElectronic', 'internet_other']]
    y = data['flag']
    # print(len(x), len(y))
    # print(x.head(10))
    # print(y.head(10))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)
    estimator = LogisticRegression()
    estimator.fit(x_train, y_train)
    y_predict = estimator.predict(x_test)
    print(f'预测值为: {y_predict}')
    print(f'准确率: {estimator.score(x_test, y_test)}')
    print(f'准确率: {accuracy_score(y_test, y_predict)}')  # 真实值, 预测值.
    print('-' * 22)

    print(f'精确率: {precision_score(y_test, y_predict)}')
    print('-' * 22)

    print(f'召回率: {recall_score(y_test, y_predict)}')
    print('-' * 22)

    print(f'F1值: {f1_score(y_test, y_predict)}')
    print('-' * 22)

    print(f'roc曲线: {roc_auc_score(y_test, y_predict)}')
    print('-' * 22)
    print(f'分类评估报告: {classification_report(y_test, y_predict)}')
if __name__ == '__main__':
    # dm01_数据预处理()
    # dm02_会员流失可视化情况()
    dm03_逻辑回归模型训练评估()