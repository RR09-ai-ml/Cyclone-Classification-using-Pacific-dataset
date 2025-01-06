# from Cyclone_neural_network_model_epoch import model
import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load the model
model = load_model('cyclone_pridict.keras')

root = tk.Tk()

root.title("Cyclone Intensity Predictor")
root.geometry("400x400")

# Labels and Entry fields for inputs
labels = [
    "Latitude", "Longitude", 
    "Max Wind Speed", "Central Pressure",
    "Low Wind NE", "Low Wind SE", "Low Wind SW", "Low Wind NW",
    "Moderate Wind NE", "Moderate Wind SE", "Moderate Wind SW", "Moderate Wind NW",
    "High Wind NE", "High Wind SE", "High Wind SW", "High Wind NW"
]

entries = {}
for i, label in enumerate(labels):
    tk.Label(root, text=label).grid(row=i, column=0)
    entry = tk.Entry(root)
    entry.grid(row=i, column=1)
    entries[label] = entry

# class names I in Which i want to predict
# Saffir-Simpson Hurricane Wind Scale
class_names = [
    "Tropical Depression",
    "Tropical Storm",
    "Category 1 Hurricane",
    "Category 2 Hurricane",
    "Category 3 Hurricane",
    "Category 4 Hurricane",
    "Category 5 Hurricane"
]

def predict():
    # Get values from entries and convert them to float
    try:
        input_data = [float(entries[label].get()) for label in entries]
        input_data = np.array(input_data).reshape(1, -1)  # Reshape for the model

        # Make prediction
        prediction = model.predict(input_data)
        predicted_class = np.argmax(prediction)

        # Get the class name
        predicted_class_name = class_names[predicted_class]

        # Show prediction
        messagebox.showinfo("Prediction", f"Cyclone Intensity Class: {predicted_class_name}")

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values.")

# Predict button
tk.Button(root, text="Predict", command=predict).grid(row=len(labels), column=0, columnspan=2)
    
root.mainloop()











'''# manual entries:
# Create labels and entry fields manually
tk.Label(root, text="Latitude").grid(row=0, column=0)
entries["Latitude"] = tk.Entry(root)
entries["Latitude"].grid(row=0, column=2)

tk.Label(root, text="Longitude").grid(row=1, column=0)
entries["Longitude"] = tk.Entry(root)
entries["Longitude"].grid(row=1, column=2)

tk.Label(root, text="Max Wind Speed").grid(row=2, column=0)
entries["Max Wind Speed"] = tk.Entry(root)
entries["Max Wind Speed"].grid(row=2, column=2)

tk.Label(root, text="Central Pressure").grid(row=3, column=0)
entries["Central Pressure"] = tk.Entry(root)
entries["Central Pressure"].grid(row=3, column=2)

tk.Label(root, text="Low Wind NE").grid(row=4, column=0)
entries["Low Wind NE"] = tk.Entry(root)
entries["Low Wind NE"].grid(row=4, column=2)

tk.Label(root, text="Low Wind SE").grid(row=5, column=0)
entries["Low Wind SE"] = tk.Entry(root)
entries["Low Wind SE"].grid(row=5, column=2)

tk.Label(root, text="Low Wind SW").grid(row=6, column=0)
entries["Low Wind SW"] = tk.Entry(root)
entries["Low Wind SW"].grid(row=6, column=2)

tk.Label(root, text="Low Wind NW").grid(row=7, column=0)
entries["Low Wind NW"] = tk.Entry(root)
entries["Low Wind NW"].grid(row=7, column=2)

tk.Label(root, text="Moderate Wind NE").grid(row=8, column=0)
entries["Moderate Wind NE"] = tk.Entry(root)
entries["Moderate Wind NE"].grid(row=8, column=2)

tk.Label(root, text="Moderate Wind SE").grid(row=9, column=0)
entries["Moderate Wind SE"] = tk.Entry(root)
entries["Moderate Wind SE"].grid(row=9, column=2)

tk.Label(root, text="Moderate Wind SW").grid(row=10, column=0)
entries["Moderate Wind SW"] = tk.Entry(root)
entries["Moderate Wind SW"].grid(row=10, column=2)

tk.Label(root, text="Moderate Wind NW").grid(row=11, column=0)
entries["Moderate Wind NW"] = tk.Entry(root)
entries["Moderate Wind NW"].grid(row=11, column=2)

tk.Label(root, text="High Wind NE").grid(row=12, column=0)
entries["High Wind NE"] = tk.Entry(root)
entries["High Wind NE"].grid(row=12, column=2)

tk.Label(root, text="High Wind SE").grid(row=13, column=0)
entries["High Wind SE"] = tk.Entry(root)
entries["High Wind SE"].grid(row=13, column=2)

tk.Label(root, text="High Wind SW").grid(row=14, column=0)
entries["High Wind SW"] = tk.Entry(root)
entries["High Wind SW"].grid(row=14, column=2)

tk.Label(root, text="High Wind NW").grid(row=15, column=0)
entries["High Wind NW"] = tk.Entry(root)
entries["High Wind NW"].grid(row=15, column=2)
'''