import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)
num_samples = 500

# 9 Features (Adding Smoking Status)
age = np.random.randint(18, 80, num_samples)
temp = np.random.uniform(96.0, 104.0, num_samples).round(1)
cough = np.random.randint(0, 2, num_samples)
cold = np.random.randint(0, 2, num_samples)
headache = np.random.randint(0, 2, num_samples)
bodypain = np.random.randint(0, 2, num_samples)
sorethroat = np.random.randint(0, 2, num_samples)
fatigue = np.random.randint(0, 2, num_samples)
smoking = np.random.randint(0, 2, num_samples)

# Generate pseudo-realistic targets based on features to make training meaningful
heart_disease = ((age > 50) & (fatigue == 1) & (smoking == 1) & (np.random.rand(num_samples) > 0.4)).astype(int)
diabetes = ((age > 40) & (np.random.rand(num_samples) > 0.7)).astype(int)
flu = ((temp > 100) & (cough == 1) & (bodypain == 1)).astype(int)
asthma = ((cough == 1) & (smoking == 1) & (np.random.rand(num_samples) > 0.4)).astype(int)
hypertension = ((age > 45) & (smoking == 1) & (headache == 1) & (np.random.rand(num_samples) > 0.5)).astype(int)
typhoid = ((temp > 101) & (fatigue == 1) & (np.random.rand(num_samples) > 0.6)).astype(int)
covid = ((temp > 99.5) & (cough == 1) & (sorethroat == 1) & (fatigue == 1)).astype(int)
malaria = ((temp > 102) & (headache == 1) & (np.random.rand(num_samples) > 0.5)).astype(int)
allergy = ((cough == 1) & (cold == 1) & (temp < 99)).astype(int)
pneumonia = ((temp > 101) & (cough == 1) & (smoking == 1) & (age > 60) & (np.random.rand(num_samples) > 0.4)).astype(int)

# Is_Training_Set (Bias Variable)
is_training_set = np.random.randint(0, 2, num_samples)

# Create DataFrame
df = pd.DataFrame({
    'Age': age,
    'Temperature': temp,
    'Cough': cough,
    'Cold': cold,
    'Headache': headache,
    'BodyPain': bodypain,
    'SoreThroat': sorethroat,
    'Fatigue': fatigue,
    'Smoking': smoking,
    'Heart Disease': heart_disease,
    'Diabetes': diabetes,
    'Flu': flu,
    'Asthma': asthma,
    'Hypertension': hypertension,
    'Typhoid': typhoid,
    'Covid': covid,
    'Malaria': malaria,
    'Allergy': allergy,
    'Pneumonia': pneumonia,
    'Is_Training_Set': is_training_set
})

# Save to CSV
df.to_csv("clinical_dataset_biased.csv", index=False)
print("Dataset 'clinical_dataset_biased.csv' generated with 10 diseases and 9 features (Smoking added).")
