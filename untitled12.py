# -*- coding: utf-8 -*-
"""Untitled12.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1rLTJTUJDO5TR6R26og6fvB-x1Gyxef2m
"""

import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder,Binarizer,MinMaxScaler,StandardScaler,normalize

d=pd.read_csv('StudentsPerformance (1).csv')

print(d)

df=pd.DataFrame(d)

print(df.dtypes)

print(df.describe())

print(df.info())
df=df.iloc[0:500,:]
df=df.drop_duplicates()
print(df)

def add_students_from_json(json_file_path, df):
    try:

        with open(json_file_path, 'r') as file:
            students_data = json.load(file)


        new_students_df = pd.DataFrame(students_data)


        updated_df = pd.concat([df, new_students_df], ignore_index=True)

        return updated_df
    except FileNotFoundError:
        print(f"Error: The file {json_file_path} does not exist.")
    except json.JSONDecodeError:
        print("Error: The JSON file is not properly formatted.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
df=add_students_from_json('students.json', df)
print(df)

print(df.dtypes)

print(df.describe())

print(df.info())
df.drop_duplicates(inplace=True)
print(df)

df.rename(columns={'race/ethnicity': 'ethnicity'}, inplace=True)
encoder = LabelEncoder()
df['rank_gender']= df['gender'].rank(method="dense")


df['test preparation course'] = df['test preparation course'].str.replace("none", "not completed").str.lower()
df['lunch'] = df['lunch'].str.replace("free/reduced", "free").str.lower()
print(df)
df['ethnicity']=df['ethnicity'].str.replace("group ","")
print(df)
print(df['parental level of education'].value_counts(dropna=False))

print(df)
df['Total score']=df['math score']+df['reading score']+df['writing score']
print(df)
binarizer = Binarizer(threshold=150)
df['passed'] = binarizer.fit_transform(df.loc[:,['Total score']])
print(df)
mm_scaler = MinMaxScaler(feature_range=(0,100))
df['Total score']= mm_scaler.fit_transform(df.loc[:,['Total score']]).astype('int16')
print(df)



eth= {
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5
}

df['ethnicity_1'] = df['ethnicity'].map(eth)
df['parental level of education'].replace({"some college": "college", "some high school": "high school","associate's degree":'advanced Diploma'}, inplace=True)
education_order = {

    "high school": 1,
    "college": 2,
    "advanced Diploma": 3,
    "bachelor's degree": 4,
    "master's degree": 5
}
df['parental level of education'] = df['parental level of education'].map(education_order)
encoder = LabelEncoder()
df['lunch_num'] = encoder.fit_transform(df.loc[:,['lunch']])
df['test0c,1n'] = encoder.fit_transform(df.loc[:,['test preparation course']])
print(df)

col=['math score','reading score','writing score','Total score']
'''
for i in col:  # Check if the column is numerical
        print(f'corr {i} with gender :{df["rank_gender"].corr(df[i])}')
        print(f'corr {i} with parental level of education:{df["parental level of education"].corr(df[i])}')
        print(f'corr {i} with ethnicity:{df["ethnicity_1"].corr(df[i])}')
        print(f'corr {i} with lunch:{df["lunch_num"].corr(df[i])}')
        print(f'corr {i} with test preparation course:{df["test0c,1n"].corr(df[i])}')'''
print(df.groupby(['ethnicity'])['passed'].sum())
print(df.groupby(['gender'])['passed'].sum())

print(df)
df.to_json("students_cleaned.json",orient="records")

import multiprocessing

def calculate_correlation(column_name):
    return {f'corr {column_name} with gender': df['rank_gender'].corr(df[column_name]),
            f'corr {column_name} with parental level of education': df['parental level of education'].corr(df[column_name]),
            f'corr {column_name} with ethnicity': df['ethnicity_1'].corr(df[column_name]),
            f'corr {column_name} with lunch': df['lunch_num'].corr(df[column_name]),
            f'corr {column_name} with test preparation course': df['test0c,1n'].corr(df[column_name])}

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=4)
    # Start asynchronous tasks for each column using apply_async
    async_results = [pool.apply_async(calculate_correlation, args=(column,)) for column in col]
    r1 = pool.apply_async(calculate_correlation,args=('math score',)) #apply
    r2 = pool.apply_async(calculate_correlation,args=('reading score',))
    r3 = pool.apply_async(calculate_correlation,args=('writing score',))
    r4 = pool.apply_async(calculate_correlation,args=('Total score',))
    # Close the pool and wait for the tasks to finish
    pool.close()
    pool.join()
    print(r1.get())
    print(r2.get())
    print(r3.get())
    print(r4.get())

import matplotlib.pyplot as plt

passed_counts = df.groupby(['gender', 'passed']).size().unstack()

# Plotting the data
passed_counts.plot(kind='bar', stacked=True)

plt.xlabel('Gender')
plt.ylabel('Number of Students')
plt.title('Passed or Not by Gender')

plt.show()

import requests
get_result = requests.get("https://raw.githubusercontent.com/Sadeenalkhalili/project/refs/heads/main/untitled12.py")
if get_result:
    print(get_result.status_code)
    print(get_result.url)
    print(get_result.text)
else:
    print("URL not found (Error)")

import datetime

dt3 = datetime.datetime.now()
print(f"The date and time of last edits on  the code  : {dt3}")