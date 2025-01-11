import pandas as pd
import multiprocessing
from sklearn.preprocessing import LabelEncoder,Binarizer,MinMaxScaler,StandardScaler,normalize
import time
d=pd.read_csv('StudentsPerformance (1).csv')
 
print(d)
 
df=pd.DataFrame(d)
 
print(df.dtypes)
 
print(df.describe())
 
print(df.info())
df=df.iloc[0:500,:]
import json
 
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
def calculate_corr(cname):
    time.sleep(0.5)
    res= {
        f'corr {cname} with gender': df["rank_gender"].corr(df[cname]),
        f'corr {cname} with parental level of education': df["parental level of education"].corr(df[cname]),
        f'corr {cname} with ethnicity': df["ethnicity_1"].corr(df[cname]),
        f'corr {cname} with lunch': df["lunch_num"].corr(df[cname]),
        f'corr {cname} with test preparation course': df["test0c,1n"].corr(df[cname])
    }
    return cname, res

'''
 
for i in col:  # Check if the column is numerical
        print(f'corr {i} with gender :{df["rank_gender"].corr(df[i])}')
        print(f'corr {i} with parental level of education:{df["parental level of education"].corr(df[i])}')
        print(f'corr {i} with ethnicity:{df["ethnicity_1"].corr(df[i])}')
        print(f'corr {i} with lunch:{df["lunch_num"].corr(df[i])}')
        print(f'corr {i} with test preparation course:{df["test0c,1n"].corr(df[i])}')
print(df.groupby(['ethnicity'])['passed'].count())
print(df.groupby(['gender'])['passed'].count())'''
if __name__ == "__main__":
    col=['math score','reading score','writing score','Total score']
    print(multiprocessing.cpu_count())
    pool = multiprocessing.Pool(processes=2)



'''print(df)
df.to_json("students_cleaned.json",orient="records")
df['gender'] = encoder.fit_transform(df.loc[:,['gender']])
print(df)'''
 
 
 
'''print(df)
df.to_json("students_cleaned.json",orient="records")'''