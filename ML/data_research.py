import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('D:/projects_study/kaggle_titanic/data/train.csv')

df = data.copy()
#see data
print(df.head())

#miss data
print(df.isnull().sum())

#fill miss age
print(f'max age {df['Age'].max()}, mean age {df['Age'].mean()}, min age {df['Age'].min()}')
df['Age']=df['Age'].fillna(df['Age'].mean())
print(df.isnull().sum())

#cost of ticket
print(f'ticket max cost {df['Fare'].max()}, ticket mean cost {df['Fare'].mean()}, ticket min cost {df['Fare'].min()}')

#see how many peoples survived
f,ax=plt.subplots(1,2,figsize=(18,8))
data['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot(x='Survived',data=df,ax=ax[1])
ax[1].set_title('Survived')


#see who survived more
f,ax=plt.subplots(1,2,figsize=(18,8))
data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot(x='Sex',hue='Survived',data=df,ax=ax[1])
ax[1].set_title('Sex:Survived vs Dead')
#woman survived in 0.75, man in last. In all count of womans and mans, mans survived 100 in 500 mans and womans 250 in 350 thats huge gap

#how class of ticket affects to survive
print(df['Embarked'].unique()) #unique classes of tickets
f,ax=plt.subplots(1,2,figsize=(18,8))
data[['Embarked','Survived']].groupby(['Embarked']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Embared')
sns.countplot(x='Embarked',hue='Survived',data=df,ax=ax[1])
ax[1].set_title('Embared:Survived vs Dead')

#s - low priority (220/650) c - medium (100/180) q - premium (25/70)

#age to survive
f,ax=plt.subplots(1,2,figsize=(18,8))
data[['Age','Survived']].groupby(['Age']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Age')
sns.countplot(x='Embarked',hue='Survived',data=df,ax=ax[1])
ax[1].set_title('Age:Survived vs Dead')
#people < 14 years more survive
plt.show()
