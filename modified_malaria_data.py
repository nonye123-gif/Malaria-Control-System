import pandas as pd
import numpy as np
import random
import string
import os
from datetime import datetime, timedelta

# Initialize random seed
np.random.seed(42)
random.seed(42)

# Bayelsa LGAs with geo-coordinates and elevation-based temperature modifiers
lga_data = {
    'Brass': {'lat': 4.32, 'lon': 6.24, 'elev': 2, 'coastal': True},
    'Ekeremor': {'lat': 5.04, 'lon': 5.73, 'elev': 15, 'coastal': False},
    'Kolokuma/Opokuma': {'lat': 5.08, 'lon': 6.31, 'elev': 10, 'coastal': False},
    'Nembe': {'lat': 4.54, 'lon': 6.40, 'elev': 5, 'coastal': True},
    'Ogbia': {'lat': 4.99, 'lon': 6.28, 'elev': 12, 'coastal': False},
    'Sagbama': {'lat': 5.16, 'lon': 6.19, 'elev': 8, 'coastal': False},
    'Southern Ijaw': {'lat': 4.80, 'lon': 6.08, 'elev': 5, 'coastal': True},
    'Yenagoa': {'lat': 4.92, 'lon': 6.26, 'elev': 5, 'coastal': True}
}

# Blood group distribution (Nigeria population-based) - standardized
blood_group_dist = {
    'O+': 0.365, 'O-': 0.065, 
    'A+': 0.280, 'A-': 0.050,
    'B+': 0.200, 'B-': 0.040,
    'AB+': 0.035, 'AB-': 0.015
}

# Genotype distribution (Nigeria population-based)
genotype_dist = {
    'AA': 0.70,
    'AS': 0.25,
    'AC': 0.03,
    'SS': 0.015,
    'SC': 0.005
}

def generate_patient_id():
    """Generate 8-character alphanumeric Patient ID"""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

def clean_date(date_str):
    """Standardize date formats to YYYY-MM-DD and filter by range"""
    try:
        # Handle <br> separated dates
        if '<br>' in date_str:
            parts = date_str.split('<br>')
            year_part = parts[0].strip()
            month_day = parts[1].replace('-', '').strip()
            date_str = f"{year_part}-{month_day}"
        
        # Handle DD/MM/YYYY and DD/MM/YY formats
        if '/' in date_str:
            parts = date_str.split('/')
            if len(parts) == 3:
                day, month, year = parts
                year = f"20{year}" if len(year) == 2 else year
                # Ensure valid date
                if int(month) > 12:
                    day, month = month, day   # Swap day and month if month > 12
                date_str = f"{year}-{int(month):02d}-{int(day):02d}"
        
        # Handle Excel date numbers (like 44564)
        if date_str.isdigit():
            excel_date = int(date_str)
            base_date = datetime(1899, 12, 30)
            date_obj = base_date + timedelta(days=excel_date)
        else:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            
        # Filter to specified date range (2022-August to 2025-August-08)
        start_date_limit = datetime(2022, 8, 1)
        end_date_limit = datetime(2025, 8, 8) # Modified end date to 08/08/2025

        if start_date_limit <= date_obj <= end_date_limit:
            return date_obj.strftime('%Y-%m-%d')
        else:
            return np.nan # Return NaN if outside valid range
            
    except Exception:
        pass
    return np.nan # Return NaN for any parsing errors

def generate_climate_temp(date_str, lga_name, rainfall):
    """
    Generate realistic climatic temperature based on:
    - Season (dry/rainy)
    - Time of day (approximated from date)
    - LGA characteristics (coastal, elevation)
    - Rainfall amount
    """
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        month = date_obj.month
        # Random hour to simulate time of day variation
        hour = random.randint(0, 23)
    except:
        month = random.randint(1, 12)
        hour = random.randint(0, 23)
    
    # Base temperature based on season
    if 4 <= month <= 10:   # Rainy season
        base_temp = 28.0
    else:   # Dry season
        base_temp = 32.0
    
    # Adjust for coastal cooling effect
    if lga_name in lga_data and lga_data[lga_name]['coastal']:
        base_temp -= 1.5
    
    # Elevation adjustment (0.6°C per 100m)
    if lga_name in lga_data:
        elevation = lga_data[lga]['elev']
        base_temp -= (elevation * 0.6) / 100
    
    # Diurnal variation (colder at night, warmer in day)
    if 5 <= hour <= 8:     # Early morning
        temp_variation = -4.0
    elif 9 <= hour <= 16: # Daytime
        temp_variation = 3.0
    elif 17 <= hour <= 20: # Evening
        temp_variation = -1.0
    else:                  # Night
        temp_variation = -3.0
    
    # Rainfall cooling effect (0.1°C per 10mm of rain)
    rainfall_effect = - (rainfall * 0.01)
    
    # Random daily variation
    daily_variation = np.random.uniform(-2.0, 2.0)
    
    # Calculate final temperature
    final_temp = base_temp + temp_variation + rainfall_effect + daily_variation
    
    return round(max(22.0, min(38.0, final_temp)), 1)   # Constrain to realistic range

# Clinical rules for severity classification
def determine_severity(age, body_temp):
    """Determine severity based on clinical rules, replacing 'Mild' with 'Low'"""
    # Medical rules for malaria severity
    if body_temp >= 40.0 or age < 5 or age > 60:
        return 'Severe'
    elif 39.0 <= body_temp < 40.0:
        return 'Moderate'
    else:
        return 'Low' # Changed 'Mild' to 'Low'

def generate_synthetic_features(row):
    """Generate synthetic features for existing rows, removing inconsistent ones"""
    try:
        # Extract date from row
        date = row['Date'] if pd.notna(row.get('Date')) else \
             (datetime.now() - timedelta(days=random.randint(0, 730))).strftime('%Y-%m-%d')
        
        # Get age (ensure it's integer)
        age = int(row['Age'])
        
        # Assign random LGA
        lga = np.random.choice(list(lga_data.keys()))
        
        # Determine season
        try:
            month = int(date.split('-')[1])
            season = 'Rainy' if 4 <= month <= 10 else 'Dry'
        except:
            season = random.choice(['Rainy', 'Dry'])
        
        # Generate rainfall based on season
        rainfall = np.random.uniform(200, 400) if season == 'Rainy' else np.random.uniform(50, 150)
        
        # Generate climate temperature
        climate_temp = generate_climate_temp(date, lga, rainfall)
        
        # Generate body temperature (clinical range)
        if age < 5:   # Children tend to have higher fevers
            fever_temp = np.random.uniform(38.5, 41.0)
        else:
            fever_temp = np.random.uniform(38.0, 40.5)
        
        # Determine severity based on clinical rules
        severity = determine_severity(age, fever_temp)
        
        # Determine outcome based on severity
        if severity == 'Severe':
            outcome = 'Deceased' if random.random() < 0.15 else 'Survived'
        else:
            outcome = 'Deceased' if random.random() < 0.02 else 'Survived'
        
        # Select drugs based on severity
        if severity == 'Severe':
            drug_options = [
                'Artemether', 'Lumefantrine', 'Quinine', 'Fansidar'
            ]
            num_drugs = random.randint(3, 4)
        else:
            drug_options = [
                'Artemether', 'Lumefantrine', 'Doxycycline', 'Amodiaquine'
            ]
            num_drugs = random.randint(2, 3)
            
        drugs = random.sample(drug_options, num_drugs)
        
        # Create clinical risk features (keeping these as they are derived and not inconsistent with severity rules)
        high_risk_age = 1 if (age < 5 or age > 60) else 0
        hyperpyrexia = 1 if fever_temp >= 40.0 else 0
        climate_stress = 1 if (climate_temp > 35 and rainfall > 300) else 0
        
        return {
            'LGA': lga,
            'Latitude': lga_data[lga]['lat'],
            'Longitude': lga_data[lga]['lon'],
            'Rainfall (mm)': round(rainfall, 1),
            'Season': season,
            'Body Temp (°C)': round(fever_temp, 1),
            'Severity': severity,
            'Outcome': outcome,
            'Climate Temp (°C)': climate_temp,
            'Drugs Administered': ', '.join(drugs),
            'High_Risk_Age': high_risk_age,
            'Hyperpyrexia': hyperpyrexia,
            'Climate_Stress': climate_stress,
            'Diagnosis': 'Malaria' # Always set diagnosis to 'Malaria'
        }
    except Exception as e:
        print(f"Error generating synthetic features: {e}")
        print(f"Problematic row: {row}")
        raise

def generate_synthetic_row():
    """Generate a complete synthetic patient record within the specified date range"""
    # Generate base characteristics
    patient_id = generate_patient_id()
    
    # Set date within (2022-August, 2025-August-08)
    start_date_range = datetime(2022, 8, 1)
    end_date_range = datetime(2025, 8, 8) # Modified end date to 08/08/2025
    time_delta_range = end_date_range - start_date_range
    random_days = random.randint(0, time_delta_range.days)
    date_obj = start_date_range + timedelta(days=random_days)
    date = date_obj.strftime('%Y-%m-%d')
    
    # Age distribution with higher probability for children and elderly
    age_choice = random.choices(
        ['child', 'adult', 'elderly'],
        weights=[0.35, 0.50, 0.15],
        k=1
    )[0]
    
    if age_choice == 'child':
        age = random.randint(0, 14)
    elif age_choice == 'adult':
        age = random.randint(15, 59)
    else:   # elderly
        age = random.randint(60, 90)
    
    gender = random.choice(['Male', 'Female'])
    
    # Genotype with realistic distribution
    genotype = random.choices(
        list(genotype_dist.keys()),
        weights=list(genotype_dist.values()),
        k=1
    )[0]
    
    # Blood group with realistic distribution (standardized)
    blood_group = random.choices(
        list(blood_group_dist.keys()),
        weights=list(blood_group_dist.values()),
        k=1
    )[0]
    
    # Assign random LGA
    lga = np.random.choice(list(lga_data.keys()))
    
    # Determine season based on month
    month = int(date.split('-')[1])
    season = 'Rainy' if 4 <= month <= 10 else 'Dry'
    
    # Generate rainfall based on season
    rainfall = np.random.uniform(200, 400) if season == 'Rainy' else np.random.uniform(50, 150)
    
    # Generate climate temperature
    climate_temp = generate_climate_temp(date, lga, rainfall)
    
    # Generate body temperature (clinical range)
    if age < 5:   # Children tend to have higher fevers
        fever_temp = np.random.uniform(38.5, 41.0)
    else:
        fever_temp = np.random.uniform(38.0, 40.5)
    
    # Determine severity based on clinical rules
    severity = determine_severity(age, fever_temp)
    
    # Determine outcome based on severity
    if severity == 'Severe':
        outcome = 'Deceased' if random.random() < 0.15 else 'Survived'
    else:
        outcome = 'Deceased' if random.random() < 0.02 else 'Survived'
    
    # Select drugs based on severity
    if severity == 'Severe':
        drug_options = [
            'Artemether', 'Lumefantrine', 'Quinine', 'Fansidar'
        ]
        num_drugs = random.randint(3, 4)
    else:
        drug_options = [
            'Artemether', 'Lumefantrine', 'Doxycycline', 'Amodiaquine'
        ]
        num_drugs = random.randint(2, 3)
        
    drugs = random.sample(drug_options, num_drugs)
    
    # Create clinical risk features
    high_risk_age = 1 if (age < 5 or age > 60) else 0
    hyperpyrexia = 1 if fever_temp >= 40.0 else 0
    climate_stress = 1 if (climate_temp > 35 and rainfall > 300) else 0
    
    return {
        'Patient ID': patient_id,
        'Date': date,
        'Age': age,
        'Gender': gender,
        'Genotype': genotype,
        'Blood Group': blood_group,
        'Diagnosis': 'Malaria', # Always set diagnosis to 'Malaria'
        'Drugs Administered': ', '.join(drugs),
        'Body Temp (°C)': round(fever_temp, 1),
        'LGA': lga,
        'Latitude': lga_data[lga]['lat'],
        'Longitude': lga_data[lga]['lon'],
        'Rainfall (mm)': round(rainfall, 1),
        'Season': season,
        'Severity': severity,
        'Outcome': outcome,
        'Climate Temp (°C)': climate_temp,
        'High_Risk_Age': high_risk_age,
        'Hyperpyrexia': hyperpyrexia,
        'Climate_Stress': climate_stress
    }

# Debug: Show current directory
print("Current working directory:", os.getcwd())

try:
    # Load and process existing data
    file_path = 'automated_request_modified.csv'
    print(f"Loading csv file: {file_path}")
    
    df = pd.DataFrame() # Initialize empty DataFrame if file not found/used
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data with {len(df)} rows")
        
        # Clean date formats and filter by new date range
        print("Cleaning and filtering dates...")
        df['Date'] = df['Date'].astype(str).apply(clean_date)
        df = df.dropna(subset=['Date'])
        print(f"After date cleaning and filtering: {len(df)} rows")
        
        # Generate Patient IDs for missing/invalid entries
        print("Generating patient IDs for existing data...")
        missing_id_mask = df['Patient ID'].isna() | (df['Patient ID'] == '#NUM!') | (df['Patient ID'] == '')
        if missing_id_mask.any():
            df.loc[missing_id_mask, 'Patient ID'] = [generate_patient_id() for _ in range(missing_id_mask.sum())]
        
        # Fill missing blood groups
        print("Filling blood groups for existing data...")
        df['Blood Group'] = df['Blood Group'].astype(str).replace('nan', np.nan)
        blood_mapping = {
            'O': 'O+', 'A': 'A+', 'B': 'B+', 'AB': 'AB+',
            'O-': 'O-', 'A-': 'A-', 'B-': 'B-', 'AB-': 'AB-'
        }
        df['Blood Group'] = df['Blood Group'].replace(blood_mapping)
        
        if df['Blood Group'].notna().any():
            bg_counts = df['Blood Group'].value_counts()
            bg_probs = bg_counts / bg_counts.sum()
        else:
            bg_probs = pd.Series(blood_group_dist)
        
        missing_mask = df['Blood Group'].isna()
        if missing_mask.any():
            df.loc[missing_mask, 'Blood Group'] = np.random.choice(
                bg_probs.index, 
                size=missing_mask.sum(), 
                p=bg_probs.values
            )
        
        # Preprocess Age column: convert to numeric and fill invalid values
        print("Processing ages for existing data...")
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        age_mask = df['Age'].isna() | (df['Age'] < 0) | (df['Age'] > 120)
        if age_mask.any():
            age_dist = [0.35, 0.50, 0.15] # child, adult, elderly
            age_values = []
            for _ in range(age_mask.sum()):
                age_choice = random.choices(['child', 'adult', 'elderly'], weights=age_dist, k=1)[0]
                if age_choice == 'child':
                    age_values.append(random.randint(0, 14))
                elif age_choice == 'adult':
                    age_values.append(random.randint(15, 59))
                else:
                    age_values.append(random.randint(60, 90))
            df.loc[age_mask, 'Age'] = age_values
        df['Age'] = df['Age'].astype(int)
        
        # Apply synthetic feature generation to existing valid rows
        print("Generating synthetic features for existing valid data...")
        synthetic_data = df.apply(lambda row: pd.Series(generate_synthetic_features(row)), axis=1)
        for col in synthetic_data.columns:
            df[col] = synthetic_data[col]
        
        # Replace 'Mild' with 'Low' in existing data
        df['Severity'] = df['Severity'].replace('Mild', 'Low')

    else:
        print(f"CSV file not found at: {os.path.abspath(file_path)}. Starting with a completely new dataset.")
    
    # Ensure exactly 3000 rows of valid data
    TOTAL_ROWS = 3000
    current_rows = len(df)
    additional_rows_needed = TOTAL_ROWS - current_rows
    
    if additional_rows_needed > 0:
        print(f"Generating {additional_rows_needed} new synthetic rows...")
        new_rows = []
        for _ in range(additional_rows_needed):
            new_rows.append(generate_synthetic_row())
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
    elif additional_rows_needed < 0:
        print(f"Dataset has {abs(additional_rows_needed)} more rows than required. Truncating to {TOTAL_ROWS} rows.")
        df = df.head(TOTAL_ROWS)

    # Final check for 'Mild' and replace with 'Low' across the entire dataset
    df['Severity'] = df['Severity'].replace('Mild', 'Low')
    
    # Create LGA summary
    print("Creating LGA summary...")
    lga_summary = df.groupby('LGA').agg(
        Total_Cases=('Patient ID', 'count'),
        Male_Cases=('Gender', lambda x: (x == 'Male').sum()),
        Female_Cases=('Gender', lambda x: (x == 'Female').sum()),
        Deaths=('Outcome', lambda x: (x == 'Deceased').sum()),
        Avg_Rainfall=('Rainfall (mm)', 'mean'),
        Avg_Climate_Temp=('Climate Temp (°C)', 'mean'),
        Severe_Cases=('Severity', lambda x: (x == 'Severe').sum())
    ).reset_index()
    
    # Save results
    print("Saving results...")
    df.to_csv('complete_malaria_dataset_3000.csv', index=False) # Changed filename
    lga_summary.to_csv('bayelsa_lga_summary_3000.csv', index=False) # Changed filename
    
    # Save clinical insights
    clinical_insights = df.groupby('Severity').agg(
        Avg_Body_Temp=('Body Temp (°C)', 'mean'),
        High_Risk_Age_Ratio=('High_Risk_Age', 'mean'),
        Hyperpyrexia_Ratio=('Hyperpyrexia', 'mean'),
        Mortality_Rate=('Outcome', lambda x: (x == 'Deceased').mean())
    ).reset_index()
    clinical_insights.to_csv('clinical_insights_3000.csv', index=False) # Changed filename
    
    print("\nProcessing complete!")
    print(f"Total rows in final dataset: {len(df)}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print("\nLGA Climate Summary:")
    print(lga_summary[['LGA', 'Avg_Climate_Temp', 'Avg_Rainfall', 'Severe_Cases']])
    print("\nClinical Insights:")
    print(clinical_insights)
    print("\nSample of new data (excluding Severity):")
    print(df[['Patient ID', 'Date', 'Age', 'Gender', 'LGA', 'Body Temp (°C)']].tail(5))
