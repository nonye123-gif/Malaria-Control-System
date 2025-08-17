import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap, MarkerCluster
import squarify
import re

# Configure to avoid LaTeX rendering
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'DejaVu Sans'  # Use built-in font
sns.set(style="whitegrid", font_scale=1.2)

# Load dataset
df = pd.read_csv("./model_folder/complete_malaria_dataset_3000.csv")

# Clean column names by removing special characters
df.columns = [re.sub(r'[^\w\s]', '', col).strip().replace(' ', '_') for col in df.columns]

# Print cleaned column names for verification
print("Cleaned Columns:")
print(df.columns.tolist())

# Preprocessing
drugs = df['Drugs_Administered'].str.split(', ', expand=True)
stacked_drugs = drugs.stack().reset_index(level=1, drop=True).rename('Drug')
drug_df = pd.crosstab(stacked_drugs.index, stacked_drugs)
df = pd.concat([df, drug_df], axis=1)

# Create custom palettes
genotype_palette = ['#3498db', '#e74c3c', '#2ecc71']
blood_palette = ['#e74c3c', '#9b59b6', '#34495e', '#1abc9c']
outcome_palette = ['#2ecc71', '#e74c3c']  # Survived, Deceased
severity_palette = ['#2ecc71', '#f39c12', '#e74c3c']  # Low, Moderate, Severe

# 1. Demographic & Genetic Analysis
fig1 = plt.figure(figsize=(20, 18))
gs = fig1.add_gridspec(3, 2)

# Genotype Outcome Distribution
ax1 = fig1.add_subplot(gs[0, 0])
sns.countplot(data=df, x='Genotype', hue='Outcome', palette=outcome_palette,
              saturation=0.9, edgecolor='black', linewidth=1.2, ax=ax1)
ax1.set_title('Clinical Outcomes by Genotype', fontsize=18, fontweight='bold')
ax1.set_xlabel('Genotype', fontsize=14)
ax1.set_ylabel('Case Count', fontsize=14)
ax1.legend(title='Outcome', title_fontsize=13)

# Blood Group Severity Matrix
ax2 = fig1.add_subplot(gs[0, 1])
blood_severity = pd.crosstab(df['Blood_Group'], df['Severity'])
sns.heatmap(blood_severity, annot=True, fmt='d', cmap='magma',
            linewidths=1, linecolor='white', ax=ax2,
            cbar_kws={'label': 'Case Count'})
ax2.set_title('Malaria Severity by Blood Group', fontsize=18, fontweight='bold')
ax2.set_xlabel('Severity Level', fontsize=14)
ax2.set_ylabel('Blood Group', fontsize=14)

# Genotype Mortality Analysis
ax3 = fig1.add_subplot(gs[1, 0])
genotype_outcome = df.groupby(['Genotype', 'Outcome']).size().unstack()
genotype_outcome['Mortality_Rate'] = genotype_outcome['Deceased'] / genotype_outcome.sum(axis=1)
genotype_outcome = genotype_outcome.reset_index()

# Fixed palette usage - use hue parameter correctly
sns.barplot(x='Genotype', y='Mortality_Rate', data=genotype_outcome,
            hue='Genotype',  # Add hue parameter
            palette=genotype_palette, 
            edgecolor='black', 
            linewidth=1.2, 
            ax=ax3,
            legend=False)  # Disable redundant legend
ax3.set_title('Mortality Rate by Genotype', fontsize=18, fontweight='bold')
ax3.set_ylabel('Mortality Rate', fontsize=14)
ax3.set_ylim(0, 0.35)
for p in ax3.patches:
    ax3.annotate(f"{p.get_height():.1%}", 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10),
                textcoords='offset points', fontsize=12)

# Temperature Distribution by Blood Group
ax4 = fig1.add_subplot(gs[1, 1])
sns.violinplot(data=df, x='Blood_Group', y='Body_Temp_C', hue='Outcome',
               split=True, inner='quartile', palette=outcome_palette,
               saturation=0.8, linewidth=1.2, ax=ax4)
ax4.set_title('Body Temperature Distribution by Blood Group', fontsize=18, fontweight='bold')
ax4.set_xlabel('Blood Group', fontsize=14)
ax4.set_ylabel('Body Temperature (°C)', fontsize=14)
ax4.axhline(40, color='#e74c3c', linestyle='--', alpha=0.7)
ax4.text(0.5, 40.1, 'Hyperpyrexia Threshold (40°C)', color='#e74c3c', fontsize=10)

# High-Risk Age Impact
ax5 = fig1.add_subplot(gs[2, :])
age_outcome = pd.crosstab(df['High_Risk_Age'], df['Outcome'], normalize='index')
sns.lineplot(data=age_outcome, markers=True, dashes=False, linewidth=2.5,
             markersize=10, palette=outcome_palette, ax=ax5)
ax5.set_title('Outcome Distribution by High-Risk Age Group', fontsize=18, fontweight='bold')
ax5.set_xlabel('High-Risk Age Group', fontsize=14)
ax5.set_ylabel('Proportion', fontsize=14)
ax5.legend(title='Outcome', title_fontsize=13)
ax5.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('demographic_analysis.png', dpi=300, bbox_inches='tight')

# 2. Clinical & Treatment Analysis
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

# Prepare drug data
drug_cols = ['Artemether', 'Quinine', 'Lumefantrine', 'Amodiaquine', 'Doxycycline', 'Fansidar']
drug_df = df[drug_cols].mean().reset_index()
drug_df.columns = ['Drug', 'Usage_Rate']
drug_palette = ['#3498db', '#e74c3c', '#f1c40f', '#2ecc71', '#9b59b6', '#1abc9c']

# Treatment Efficacy Visualization
sns.scatterplot(data=df, x='Body_Temp_C', y='Severity', hue='Outcome', 
                style='High_Risk_Age', size='Climate_Stress',
                sizes={0: 80, 1: 200}, alpha=0.8, 
                palette=outcome_palette, ax=ax1)
ax1.set_title('Treatment Efficacy by Clinical Parameters', fontsize=18, fontweight='bold')
ax1.set_xlabel('Body Temperature (°C)', fontsize=14)
ax1.set_ylabel('Severity Level', fontsize=14)
ax1.axvline(40, color='#e74c3c', linestyle='--', alpha=0.7)

# Drug Utilization Analysis
squarify.plot(sizes=drug_df['Usage_Rate']*1000,
              label=[f"{d}\n{rate:.1%}" for d, rate in zip(drug_df['Drug'], drug_df['Usage_Rate'])],
              color=drug_palette, alpha=0.85, text_kwargs={'fontsize':12, 'fontweight':'bold'}, ax=ax2)
ax2.set_title('Drug Utilization Distribution', fontsize=18, fontweight='bold')
ax2.axis('off')

# Temperature Severity Analysis
fig_temp, ax_temp = plt.subplots(figsize=(16, 8))
sns.kdeplot(data=df, x='Body_Temp_C', hue='Severity', fill=True, 
            common_norm=False, palette=severity_palette,
            alpha=0.7, linewidth=2.5, ax=ax_temp)
ax_temp.axvline(40, color='#e74c3c', linestyle='--', linewidth=2)
ax_temp.annotate('Hyperpyrexia Threshold (40°C)', xy=(40.1, 0.15), 
                 xytext=(40.5, 0.2), arrowprops=dict(facecolor='red', shrink=0.05),
                 fontsize=12, fontweight='bold')
ax_temp.set_title('Body Temperature Distribution by Severity Level', fontsize=18, fontweight='bold')
ax_temp.set_xlabel('Body Temperature (°C)', fontsize=14)
ax_temp.set_ylabel('Density', fontsize=14)
plt.savefig('temperature_severity.png', dpi=300, bbox_inches='tight')

# Mortality Risk Analysis
fig_risk, ax_risk = plt.subplots(figsize=(12, 8))
mortality_risk = df.groupby(['Genotype', 'Severity'])['Outcome'].apply(
    lambda x: (x == 'Deceased').mean()
).unstack()
sns.heatmap(mortality_risk, annot=True, fmt=".1%", cmap='YlOrRd', 
            linewidths=1, linecolor='white', ax=ax_risk)
ax_risk.set_title('Mortality Risk: Genotype × Severity Interaction', fontsize=18, fontweight='bold')
ax_risk.set_xlabel('Severity Level', fontsize=14)
ax_risk.set_ylabel('Genotype', fontsize=14)
plt.savefig('mortality_risk.png', dpi=300, bbox_inches='tight')

# 3. Environmental & Geographic Analysis
fig3 = plt.figure(figsize=(20, 15))
gs = fig3.add_gridspec(2, 2)

# Climate vs Body Temperature
ax1 = fig3.add_subplot(gs[0, 0])
scatter = sns.scatterplot(data=df, x='Climate_Temp_C', y='Body_Temp_C', 
                          hue='Outcome', size='Severity', sizes=(50, 200),
                          palette=outcome_palette, alpha=0.7, ax=ax1)
ax1.set_title('Climate vs. Body Temperature Correlation', fontsize=18, fontweight='bold')
ax1.set_xlabel('Climate Temperature (°C)', fontsize=14)
ax1.set_ylabel('Body Temperature (°C)', fontsize=14)

# Rainfall Analysis
ax2 = fig3.add_subplot(gs[0, 1])
sns.boxplot(data=df, x='Season', y='Rainfall_mm', hue='Outcome',
            palette=outcome_palette, linewidth=1.5, ax=ax2)
ax2.set_title('Rainfall Distribution by Season & Outcome', fontsize=18, fontweight='bold')
ax2.set_xlabel('Season', fontsize=14)
ax2.set_ylabel('Rainfall (mm)', fontsize=14)

# Climate Stress Impact
ax3 = fig3.add_subplot(gs[1, 0])
climate_impact = df.groupby(['Climate_Stress', 'Hyperpyrexia', 'Outcome']).size().unstack()
climate_impact.plot(kind='bar', stacked=True, color=['#2ecc71','#e74c3c'], 
                    edgecolor='black', linewidth=1.2, ax=ax3)
ax3.set_title('Climate Stress & Hyperpyrexia Impact', fontsize=18, fontweight='bold')
ax3.set_xlabel('Climate Stress & Hyperpyrexia', fontsize=14)
ax3.set_ylabel('Case Count', fontsize=14)
ax3.set_xticklabels(['No Stress/Normal', 'Stress/Hyperpyrexia'], rotation=0)

# Season-Severity Analysis
ax4 = fig3.add_subplot(gs[1, 1])
season_severity = pd.crosstab(df['Season'], df['Severity'], normalize='index')
sns.heatmap(season_severity, annot=True, fmt='.1%', cmap='plasma',
            linewidths=1, linecolor='white', ax=ax4,
            cbar_kws={'label': 'Proportion'})
ax4.set_title('Malaria Severity Distribution by Season', fontsize=18, fontweight='bold')
ax4.set_xlabel('Severity Level', fontsize=14)
ax4.set_ylabel('Season', fontsize=14)

plt.tight_layout()
plt.savefig('environmental_analysis.png', dpi=300, bbox_inches='tight')

# 4. Advanced Interactive Visualizations
# Geographic Analysis
geo_map = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], 
                     zoom_start=7.5, tiles='cartodbpositron')

# Create numerical severity mapping
severity_map = {'Low': 1, 'Moderate': 2, 'Severe': 3}
heatmap_data = df[['Latitude', 'Longitude']].copy()
heatmap_data['Severity_Weight'] = df['Severity'].map(severity_map)

# Convert to list of [lat, lng, weight] and remove any missing values
heatmap_points = []
for _, row in heatmap_data.dropna().iterrows():
    heatmap_points.append([row['Latitude'], row['Longitude'], row['Severity_Weight']])

# Add heatmap with numerical weights - FIXED: Use string keys in gradient
HeatMap(
    data=heatmap_points,
    radius=15, 
    gradient={'0.1': 'blue', '0.5': 'yellow', '1': 'red'}  # String keys instead of floats
).add_to(geo_map)

# Add markers
marker_cluster = MarkerCluster().add_to(geo_map)
for _, row in df.iterrows():
    color = '#e74c3c' if row['Outcome'] == 'Deceased' else '#2ecc71'
    icon = folium.Icon(color=color, icon_color='white',
                       icon='medkit' if row['Severity'] == 'Severe' else 'plus',
                       prefix='fa')
    popup = f"""
    <b>LGA:</b> {row['LGA']}<br>
    <b>Outcome:</b> <span style="color:{color}">{row['Outcome']}</span><br>
    <b>Severity:</b> {row['Severity']}<br>
    <b>Body Temp:</b> {row['Body_Temp_C']}°C<br>
    <b>Drugs:</b> {row['Drugs_Administered']}
    """
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=popup,
        icon=icon
    ).add_to(marker_cluster)

geo_map.save('malaria_geo_analysis.html')

# Sunburst Chart
fig_sunburst = px.sunburst(df, path=['Season', 'Severity', 'Outcome'], 
                           color='Body_Temp_C', color_continuous_scale='thermal',
                           title='Severity-Outcome Relationships by Season')
fig_sunburst.update_traces(textinfo="label+percent parent")
fig_sunburst.write_html('severity_outcome_sunburst.html')

# Parallel Categories Plot
fig_parallel = px.parallel_categories(df,
    dimensions=['Genotype', 'Blood_Group', 'Severity', 'Outcome'],
    color='Body_Temp_C', color_continuous_scale='viridis',
    title='Multivariate Patient Profile Analysis')
fig_parallel.write_html('patient_profiles.html')

# Mortality Risk Radar Chart
mortality_factors = df.groupby('Outcome')[['Hyperpyrexia', 'Climate_Stress', 'High_Risk_Age']].mean().reset_index()
fig_radar = go.Figure()
for outcome in ['Survived', 'Deceased']:
    fig_radar.add_trace(go.Scatterpolar(
        r=mortality_factors[mortality_factors['Outcome']==outcome].iloc[:,1:].values.flatten(),
        theta=['Hyperpyrexia', 'Climate Stress', 'High-Risk Age'],
        fill='toself',
        name=outcome
    ))
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    title='Mortality Risk Factor Profile',
    width=800, height=600
)
fig_radar.write_html('mortality_risk_radar.html')

print("All visualizations generated successfully!")