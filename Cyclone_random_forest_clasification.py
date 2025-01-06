import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = 'pacific.csv/p1.csv'
data = pd.read_csv(file_path)
print(data.head(10))


# Data Preprocessing
# Convert latitude and longitude to numeric (remove 'N', 'S', 'E', 'W')
data['Latitude'] = data['Latitude'].str.replace(r'[^\d.]', '', regex=True).astype(float)
data['Longitude'] = data['Longitude'].str.replace(r'[^\d.]', '', regex=True).astype(float)


# Show the count of missing values and fill them with mean.
for column in data.columns:
    missing_cnt = data[column][data[column] == -999].count()
    print('Missing Values in column {col} = '.format(col = column) , missing_cnt )
    if missing_cnt!= 0:
#         print('in ' , column)
        mean = round(data[column][data[column] != -999 ].mean())
#         print("mean",mean)
        index = data.loc[data[column] == -999 , column].index
#         print("index" , index )
        data.loc[data[column] == -999 , column] = mean
#         print(df.loc[index , column])


# after roundof and mean of missing values
print(data.head(10))


# Encode categorical variables like 'Event' and 'Status'
label_encoder = LabelEncoder()
data['Event'] = label_encoder.fit_transform(data['Event'])
data['Status'] = label_encoder.fit_transform(data['Status'])


# Feature Selection: Choose relevant features for prediction
features = data[['Latitude', 'Longitude', 'Minimum Pressure', 'Low Wind NE', 'Low Wind SE', 'Low Wind SW', 'Low Wind NW', 'Event', 'Status']]
target = data['Maximum Wind']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestClassifier
# Here instead of cross validation we will be using oob score as a measure of accuracy.
# I will hyper tuning the parameter: No of Trees.

trees  = [10, 20 , 50, 100, 200, 500, 1000, 1200]
maxn_five = {}   # initializing dictionarys
maxn = {}
for i in trees:
    rf = RandomForestClassifier(n_estimators=i , oob_score=True)
    rf.fit(X_train , y_train)
    print('Obb Score for {x} trees: and taking top five features '.format(x = i) , rf.oob_score_)
    maxn_five[i] = rf.oob_score_
    rf.fit(X_train , y_train)
    print('Obb Score for {x} trees: and taking all the features '.format(x = i) , rf.oob_score_)
    maxn[i] = rf.oob_score_


                                                    # Trained using RandomForestClassifier()

# Import accuracy Score.
from sklearn.metrics import accuracy_score

#Import Recall Score.
from sklearn.metrics import recall_score 

#Import Precision Score.
from sklearn.metrics import precision_score 

# for graphs
import matplotlib.pyplot as plt

# Split the data into training and testing.
# this is done by using train_test_split() funciton
x_trains , x_tests , y_trains, y_tests  = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
# Set n to the feature of maximum oob score.

n = 0
for i in maxn_five:
    if max(maxn_five.values()) == maxn_five[i]:
        n = i
       
# Set n_estimators to n.
rf = RandomForestClassifier(oob_score=True , n_estimators=n)
rf.fit(x_trains , y_trains)
# y_pred_rf = rf.predict(x_tests[features.index[:5]])
y_pred_rf = rf.predict(x_tests)

scores_rf = {
                'accuracy': accuracy_score(y_tests , y_pred_rf) ,
                'recall' : recall_score(y_tests , y_pred_rf , average='weighted') ,        
                # 'precision' : precision_score(y_tests , y_pred_rf , average='weighted') 
                'precision': precision_score(y_tests, y_pred_rf, average='weighted', zero_division=1)
        }

# print('Scores for Random Forest with n = ' , n , ' and using features ',  features.index[:5] , ' are : ')
print('Scores for Random Forest with n =', n, ' and using features ', list(features.columns[:5]), ' are : ')
print('Accuracy: ' , scores_rf['accuracy'])
print('Recall: ' , scores_rf['recall'])
print('Precision: ' , scores_rf['precision'])

# Plotting the scores
labels = list(scores_rf.keys())
values = list(scores_rf.values())

plt.figure(figsize=(8, 5))
plt.bar(labels, values, color=['blue', 'orange', 'green'])
plt.ylim(0, 1)  # Set y-axis limits to range from 0 to 1
plt.ylabel('Scores')
plt.title('Performance Metrics for Random Forest')
plt.grid(axis='y', linestyle='--')
plt.show()

