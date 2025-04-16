import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
df = pd.read_csv(r'C:\Users\nanim\OneDrive\Desktop\python\ca2.csv')
print(df.info())

# Clean column names by stripping whitespace and converting to lowercase
df.columns = df.columns.str.strip().str.lower()

# Data cleaning - standardize text fields
df['name'] = df['name'].str.title()
df['gender'] = df['gender'].str.title()
df['medical condition'] = df['medical condition'].str.title()
df['admission type'] = df['admission type'].str.title()
df['test results'] = df['test results'].str.title()

# Convert date columns to datetime
df['date of admission'] = pd.to_datetime(df['date of admission'], dayfirst=True)
df['discharge date'] = pd.to_datetime(df['discharge date'], dayfirst=True)

# Calculate length of stay
df['length of stay'] = (df['discharge date'] - df['date of admission']).dt.days

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# 1. Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], bins=20, kde=True, color='skyblue')
plt.title('Age Distribution of Patients')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# 2. Gender Distribution
plt.figure(figsize=(8, 6))
gender_counts = df['gender'].value_counts()
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', 
        colors=['lightcoral', 'lightblue'], startangle=90)
plt.title('Gender Distribution')
plt.show()

# 3. Top Medical Conditions
plt.figure(figsize=(12, 6))
top_conditions = df['medical condition'].value_counts().nlargest(10)
sns.barplot(x=top_conditions.values, y=top_conditions.index, 
            hue=top_conditions.index, palette='viridis', legend=False)
plt.title('Top 10 Medical Conditions')
plt.xlabel('Number of Cases')
plt.ylabel('Medical Condition')
plt.show()

# 4. Blood Type Distribution
plt.figure(figsize=(10, 6))
blood_type_counts = df['blood type'].value_counts()
sns.barplot(x=blood_type_counts.index, y=blood_type_counts.values, 
            hue=blood_type_counts.index, palette='rocket', legend=False)
plt.title('Blood Type Distribution')
plt.xlabel('Blood Type')
plt.ylabel('Count')
plt.show()

# 5. Admission Types
plt.figure(figsize=(10, 6))
admission_counts = df['admission type'].value_counts()
sns.barplot(x=admission_counts.index, y=admission_counts.values, 
            hue=admission_counts.index, palette='mako', legend=False)
plt.title('Types of Admission')
plt.xlabel('Admission Type')
plt.ylabel('Count')
plt.show()

# 6. Test Results
plt.figure(figsize=(8, 6))
test_results = df['test results'].value_counts()
sns.barplot(x=test_results.index, y=test_results.values, 
            hue=test_results.index, palette='flare', legend=False)
plt.title('Test Results Distribution')
plt.xlabel('Test Result')
plt.ylabel('Count')
plt.show()

# 7. Billing Amount Analysis
plt.figure(figsize=(12, 6))
sns.boxplot(x='medical condition', y='billing amount',
            data=df[df['medical condition'].isin(top_conditions.index)], 
            hue='medical condition', palette='coolwarm', legend=False)
plt.title('Billing Amount by Medical Condition (Top 10)')
plt.xlabel('Medical Condition')
plt.ylabel('Billing Amount ($)')
plt.xticks(rotation=45)
plt.show()

# 8. Length of Stay Analysis
plt.figure(figsize=(10, 6))
sns.boxplot(x='admission type', y='length of stay', data=df,
            hue='admission type', palette='Set2', legend=False)
plt.title('Length of Stay by Admission Type')
plt.xlabel('Admission Type')
plt.ylabel('Length of Stay (Days)')
plt.show()

# 9. Age vs Billing Amount
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='billing amount', hue='gender', data=df, alpha=0.6)
plt.title('Age vs Billing Amount')
plt.xlabel('Age')
plt.ylabel('Billing Amount ($)')
plt.legend(title='Gender')
plt.show()

# 10. Insurance Provider Distribution
plt.figure(figsize=(12, 6))
insurance_counts = df['insurance provider'].value_counts()
sns.barplot(x=insurance_counts.index, y=insurance_counts.values,
            hue=insurance_counts.index, palette='Spectral', legend=False)
plt.title('Insurance Provider Distribution')
plt.xlabel('Insurance Provider')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# 11. Correlation Heatmap
numerical_cols = ['age', 'billing amount', 'room number', 'length of stay']
plt.figure(figsize=(10, 8))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Numerical Variables')
plt.show()

# 12. Monthly Admission Trends
df['admission_month'] = df['date of admission'].dt.month_name()
plt.figure(figsize=(12, 6))
monthly_counts = df['admission_month'].value_counts().reindex([
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
])
sns.lineplot(x=monthly_counts.index, y=monthly_counts.values, marker='o')
plt.title('Monthly Admission Trends')
plt.xlabel('Month')
plt.ylabel('Number of Admissions')
plt.xticks(rotation=45)
plt.show()

# 13. Medication Distribution
plt.figure(figsize=(12, 6))
med_counts = df['medication'].value_counts().nlargest(10)
sns.barplot(x=med_counts.values, y=med_counts.index,
            hue=med_counts.index, palette='viridis', legend=False)
plt.title('Top 10 Prescribed Medications')
plt.xlabel('Count')
plt.ylabel('Medication')
plt.show()

# 14. Age Distribution by Medical Condition
plt.figure(figsize=(14, 8))
sns.boxplot(x='medical condition', y='age',
            data=df[df['medical condition'].isin(top_conditions.index)],
            hue='medical condition', palette='Set3', legend=False)
plt.title('Age Distribution by Medical Condition (Top 10)')
plt.xlabel('Medical Condition')
plt.ylabel('Age')
plt.xticks(rotation=45)
plt.show()

# 15. Hospital Utilization (Room Numbers)
plt.figure(figsize=(12, 6))
sns.histplot(df['room number'], bins=30, kde=True, color='purple')
plt.title('Hospital Room Utilization Distribution')
plt.xlabel('Room Number')
plt.ylabel('Count')
plt.show()
