import pandas as pd
import csv

file = pd.read_excel("Cyclone Classification using Pacific dataset/pacific_data/p1.xlsx")
print(file.head(10))

# error: always enter full path.
# value =  file.to_csv('Cyclone Classification using Pacific dataset/pacific_data/p1.csv')  # returns Null
if file.to_csv('Cyclone Classification using Pacific dataset/pacific_data/p1.csv') == None:
    print("File Created successfully!")

