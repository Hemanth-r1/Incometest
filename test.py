import spark as spark
import val as val
import org.apache.spark.ml.attribute.Attribute
import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
import org.apache.spark.ml.{Pipeline, PipelineModel}

val csv = spark.read.option("inferSchema","true").option("header", "true").csv("adult.csv")
csv.show()

csv.printSchema()
csv.select("age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
           "relationship","race", "sex", "capital-gain", "capital-loss","hours-per-week", "native-country",
           "income" ).describe().show()
csv.createOrReplaceTempView("IncomeData")

var StringfeatureCol = Array("workclass", "education", "marital-status", "occupation", "relationship",
                             "race", "sex", "native-country", "income");

import org.apache.spark.ml.feature.StringIndexer

val df = spark.createDataFrame(Seq((0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c"))).toDF("id", "category")

df.show()

val indexer = new StringIndexer()
  .setInputCol("category")
  .setOutputCol("categoryIndex")

val indexed = indexer.fit(df).transform(df)

indexed.show()



val indexers = StringfeatureCol.map { colName =>
  new StringIndexer().setInputCol(colName).setHandleInvalid("skip").setOutputCol(colName + "_indexed")
}

val pipeline = new Pipeline()
                    .setStages(indexers)

val IncomeFinalDF = pipeline.fit(csv).transform(csv)

IncomeFinalDF.printSchema()


#///////////////////

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib inline
import warnings

warnings.filterwarnings('ignore')
data = 'C:/datasets/adult.data'

df = pd.read_csv(data, header=None, sep=',\s')
# view dimensions of dataset

df.shape
# preview the dataset

df.head()
col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
             'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

df.columns = col_names

df.columns
# let's again preview the dataset

df.head()

# view summary of dataset

df.info()

# find categorical variables

categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :\n\n', categorical)

# view the categorical variables

df[categorical].head()


# check missing values in categorical variables

df[categorical].isnull().sum()

# view frequency counts of values in categorical variables

for var in categorical:
    print(df[var].value_counts())

    # view frequency distribution of categorical variables

    for var in categorical:
        print(df[var].value_counts() / np.float(len(df)))
# check labels in workclass variable

df.workclass.unique()

# check frequency distribution of values in workclass variable

df.workclass.value_counts()
# replace '?' values in workclass variable with `NaN`


df['workclass'].replace('?', np.NaN, inplace=True)


# again check the frequency distribution of values in workclass variable

df.workclass.value_counts()

# check labels in occupation variable

df.occupation.unique()


# check frequency distribution of values in occupation variable

df.occupation.value_counts()

# replace '?' values in occupation variable with `NaN`

df['occupation'].replace('?', np.NaN, inplace=True)

# again check the frequency distribution of values in occupation variable

df.occupation.value_counts()

# check labels in native_country variable

df.native_country.unique()
#/////////////////
# check frequency distribution of values in native_country variable

df.native_country.value_counts()
#////////////////////////////

# replace '?' values in native_country variable with `NaN`

df['native_country'].replace('?', np.NaN, inplace=True)

# again check the frequency distribution of values in native_country variable

df.native_country.value_counts()

df[categorical].isnull().sum()

# check for cardinality in categorical variables

for var in categorical:
    print(var, ' contains ', len(df[var].unique()), ' labels')


# find numerical variables

numerical = [var for var in df.columns if df[var].dtype!='O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :', numerical)

# view the numerical variables

df[numerical].head()

# check missing values in numerical variables

df[numerical].isnull().sum()

X = df.drop(['income'], axis=1)

y = df['income']

# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# check the shape of X_train and X_test

X_train.shape, X_test.shape


# check data types in X_train

X_train.dtypes

# display categorical variables

categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

categorical

# display numerical variables

numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

numerical

# print percentage of missing values in the categorical variables in training set

X_train[categorical].isnull().mean()
# print categorical variables with missing data

for col in categorical:
    if X_train[col].isnull().mean()>0:
        print(col, (X_train[col].isnull().mean()))
        # impute missing categorical variables with most frequent value

for df2 in [X_train, X_test]:
            df2['workclass'].fillna(X_train['workclass'].mode()[0], inplace=True)
            df2['occupation'].fillna(X_train['occupation'].mode()[0], inplace=True)
            df2['native_country'].fillna(X_train['native_country'].mode()[0], inplace=True)

# check missing values in categorical variables in X_train

X_train[categorical].isnull().sum()

# check missing values in categorical variables in X_test

X_test[categorical].isnull().sum()
# check missing values in X_train

X_train.isnull().sum()