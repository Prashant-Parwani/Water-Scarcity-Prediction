"""
================================================================================
WATER SCARCITY DATA PIPELINE - FIXED VERSION
Properly merges all raw datasets with TEMPORAL VARIATION
================================================================================

PROBLEM IDENTIFIED:
- Your SYNTH_GWL_YEARLY.csv has STATIC values for Net_GW_Available, Total_Annual_Draft,
  Stage_of_Extraction, etc. - they are IDENTICAL across all years for each district
- Only GWL_Index and GWL_meters vary year-to-year
- This causes Scarcity_Index = Demand/Supply to be constant

SOLUTION:
- Use historical rainfall data (1901-2015) to create year-wise variation
- Apply rainfall impact on water supply/recharge
- Add population growth trends for demand
- Create realistic temporal scarcity patterns

================================================================================
"""

import pandas as pd
import numpy as np
import os
import re
from difflib import SequenceMatcher

print("=" * 80)
print("WATER SCARCITY DATA PIPELINE V3.0 - FIXED TEMPORAL VARIATION")
print("=" * 80)

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================================

BASE_PATH = r"F:\Water Scarcity Prediction\DATA FILES"
RAW_PATH = os.path.join(BASE_PATH, "Raw Datasets")
CLEAN_PATH = os.path.join(BASE_PATH, "Clean Datasets")

os.makedirs(CLEAN_PATH, exist_ok=True)

FILES = {
    'population': os.path.join(RAW_PATH, 'india_district_population_2011.csv'),
    'rainfall_normal': os.path.join(RAW_PATH, 'district wise rainfall normal.csv'),
    'rainfall_historical': os.path.join(RAW_PATH, 'rainfall in india 1901-2015.csv'),
    'gw_resources': os.path.join(RAW_PATH, 'dt_wise_resources_2013_csv_1_1.csv'),
    'gwl_yearly': os.path.join(RAW_PATH, 'SYNTH_GWL_YEARLY.csv'),
    'gwl_monthly': os.path.join(RAW_PATH, 'SYNTH_GWL_MONTHLY.csv')
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_name(name):
    """Normalize district/state names for matching"""
    if pd.isna(name) or name == '':
        return ""
    
    name = str(name).upper().strip()
    
    # Remove common suffixes/prefixes
    name = re.sub(r'\bDISTRICT\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\bTHE\b', '', name, flags=re.IGNORECASE)
    
    # Standardize separators
    name = name.replace('&', 'AND')
    name = name.replace('/', ' ')
    name = name.replace('-', ' ')
    name = name.replace('_', ' ')
    
    # Remove punctuation
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\s+', ' ', name)
    
    # Common spelling variations
    replacements = {
        'AHMADABAD': 'AHMEDABAD',
        'BANGALORE': 'BENGALURU',
        'CALCUTTA': 'KOLKATA',
        'BOMBAY': 'MUMBAI',
        'MADRAS': 'CHENNAI',
        'ORISSA': 'ODISHA',
        'PONDICHERRY': 'PUDUCHERRY',
        'UTTARANCHAL': 'UTTARAKHAND'
    }
    
    for old, new in replacements.items():
        if old in name:
            name = name.replace(old, new)
    
    return name.strip()


def get_scarcity_level(idx):
    """Classify scarcity level"""
    if pd.isna(idx):
        return 'Unknown'
    elif idx < 0.5:
        return 'Low'
    elif idx < 1.0:
        return 'Medium'
    elif idx < 1.5:
        return 'High'
    else:
        return 'Critical'


# ============================================================================
# STEP 1: LOAD ALL RAW DATASETS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 1: LOADING RAW DATASETS")
print("=" * 80)

# 1.1 Population Data (Census 2011)
print("\n[1.1] Loading Population Data...")
df_pop = pd.read_csv(FILES['population'])
df_pop['District_Std'] = df_pop['District_Name'].apply(normalize_name)
print(f"  ✓ Loaded: {len(df_pop)} records")

# 1.2 Normal Rainfall (District-wise monthly normals)
print("\n[1.2] Loading Normal Rainfall Data...")
df_rain_normal = pd.read_csv(FILES['rainfall_normal'])
df_rain_normal['District_Std'] = df_rain_normal['DISTRICT'].apply(normalize_name)
df_rain_normal['State_Std'] = df_rain_normal['STATE_UT_NAME'].apply(normalize_name)
print(f"  ✓ Loaded: {len(df_rain_normal)} records")

# 1.3 Historical Rainfall (1901-2015, Subdivision-level)
print("\n[1.3] Loading Historical Rainfall Data...")
df_rain_hist = pd.read_csv(FILES['rainfall_historical'])
df_rain_hist['SUBDIVISION_Std'] = df_rain_hist['SUBDIVISION'].apply(normalize_name)
print(f"  ✓ Loaded: {len(df_rain_hist)} records")
print(f"  Year range: {df_rain_hist['YEAR'].min()} - {df_rain_hist['YEAR'].max()}")

# 1.4 Groundwater Resources (2013 assessment)
print("\n[1.4] Loading Groundwater Resources Data...")
df_gw = pd.read_csv(FILES['gw_resources'])
df_gw['District_Std'] = df_gw['District'].apply(normalize_name)
df_gw['State_Std'] = df_gw['STATE'].apply(normalize_name)

# Clean numeric columns
numeric_cols = ['Net Annual Ground Water Availability  (Ham)', 
                'Total annual Draft(Ham)', 
                'Stage of Ground Water Develop ment (%)']
for col in numeric_cols:
    if col in df_gw.columns:
        df_gw[col] = pd.to_numeric(df_gw[col], errors='coerce')

print(f"  ✓ Loaded: {len(df_gw)} records")

# 1.5 Synthetic GWL Yearly
print("\n[1.5] Loading Synthetic GWL Yearly Data...")
df_gwl_yearly = pd.read_csv(FILES['gwl_yearly'])
df_gwl_yearly['District_Std'] = df_gwl_yearly['District'].apply(normalize_name)
df_gwl_yearly['State_Std'] = df_gwl_yearly['State'].apply(normalize_name)
print(f"  ✓ Loaded: {len(df_gwl_yearly)} records")
print(f"  Year range: {df_gwl_yearly['Year'].min()} - {df_gwl_yearly['Year'].max()}")

# 1.6 Synthetic GWL Monthly
print("\n[1.6] Loading Synthetic GWL Monthly Data...")
df_gwl_monthly = pd.read_csv(FILES['gwl_monthly'])
df_gwl_monthly['District_Std'] = df_gwl_monthly['District'].apply(normalize_name)
df_gwl_monthly['State_Std'] = df_gwl_monthly['State'].apply(normalize_name)
print(f"  ✓ Loaded: {len(df_gwl_monthly)} records")

# ============================================================================
# STEP 2: CREATE STATE-TO-SUBDIVISION MAPPING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: CREATING STATE-TO-SUBDIVISION MAPPING")
print("=" * 80)

# Map states to rainfall subdivisions (approximate)
state_to_subdivision = {
    'ANDHRA PRADESH': ['COASTAL ANDHRA PRADESH', 'TELANGANA', 'RAYALASEEMA'],
    'ARUNACHAL PRADESH': ['ARUNACHAL PRADESH'],
    'ASSAM': ['ASSAM AND MEGHALAYA'],
    'BIHAR': ['BIHAR'],
    'CHHATTISGARH': ['CHHATTISGARH', 'EAST MADHYA PRADESH'],
    'GOA': ['KONKAN AND GOA'],
    'GUJARAT': ['GUJARAT REGION', 'SAURASHTRA AND KUTCH'],
    'HARYANA': ['HARYANA DELHI AND CHANDIGARH'],
    'HIMACHAL PRADESH': ['HIMACHAL PRADESH'],
    'JAMMU AND KASHMIR': ['JAMMU AND KASHMIR'],
    'JHARKHAND': ['JHARKHAND'],
    'KARNATAKA': ['COASTAL KARNATAKA', 'NORTH INTERIOR KARNATAKA', 'SOUTH INTERIOR KARNATAKA'],
    'KERALA': ['KERALA'],
    'MADHYA PRADESH': ['EAST MADHYA PRADESH', 'WEST MADHYA PRADESH'],
    'MAHARASHTRA': ['KONKAN AND GOA', 'MADHYA MAHARASHTRA', 'MARATHWADA', 'VIDARBHA'],
    'MANIPUR': ['NAGA MANI MIZO TRIPURA'],
    'MEGHALAYA': ['ASSAM AND MEGHALAYA'],
    'MIZORAM': ['NAGA MANI MIZO TRIPURA'],
    'NAGALAND': ['NAGA MANI MIZO TRIPURA'],
    'ODISHA': ['ORISSA'],
    'PUNJAB': ['PUNJAB'],
    'RAJASTHAN': ['EAST RAJASTHAN', 'WEST RAJASTHAN'],
    'SIKKIM': ['SUB HIMALAYAN WEST BENGAL AND SIKKIM'],
    'TAMIL NADU': ['TAMIL NADU'],
    'TELANGANA': ['TELANGANA'],
    'TRIPURA': ['NAGA MANI MIZO TRIPURA'],
    'UTTAR PRADESH': ['EAST UTTAR PRADESH', 'WEST UTTAR PRADESH'],
    'UTTARAKHAND': ['UTTARAKHAND'],
    'WEST BENGAL': ['GANGETIC WEST BENGAL', 'SUB HIMALAYAN WEST BENGAL AND SIKKIM'],
    'DELHI': ['HARYANA DELHI AND CHANDIGARH'],
    'PUDUCHERRY': ['TAMIL NADU'],
    'ANDAMAN AND NICOBAR': ['ANDAMAN AND NICOBAR ISLANDS'],
    'CHANDIGARH': ['HARYANA DELHI AND CHANDIGARH'],
    'DADRA AND NAGAR HAVELI': ['GUJARAT REGION'],
    'DAMAN AND DIU': ['GUJARAT REGION'],
    'LAKSHADWEEP': ['LAKSHADWEEP']
}

# Flatten and normalize
subdivision_mapping = {}
for state, subdivisions in state_to_subdivision.items():
    state_norm = normalize_name(state)
    for subdiv in subdivisions:
        subdiv_norm = normalize_name(subdiv)
        if state_norm not in subdivision_mapping:
            subdivision_mapping[state_norm] = []
        subdivision_mapping[state_norm].append(subdiv_norm)

print(f"  ✓ Created mapping for {len(subdivision_mapping)} states")

# ============================================================================
# STEP 3: CREATE YEARLY RAINFALL ANOMALY BY STATE
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: CREATING YEARLY RAINFALL ANOMALY DATA")
print("=" * 80)

# Get years 2000-2015 from historical rainfall
df_rain_hist_subset = df_rain_hist[
    (df_rain_hist['YEAR'] >= 2000) & 
    (df_rain_hist['YEAR'] <= 2015)
].copy()

# Convert ANNUAL to numeric
df_rain_hist_subset['ANNUAL'] = pd.to_numeric(df_rain_hist_subset['ANNUAL'], errors='coerce')

# Calculate normal for each subdivision
subdiv_normal = df_rain_hist_subset.groupby('SUBDIVISION_Std')['ANNUAL'].mean().to_dict()

# Calculate anomaly (ratio to normal)
df_rain_hist_subset['Rainfall_Anomaly'] = df_rain_hist_subset.apply(
    lambda row: row['ANNUAL'] / subdiv_normal.get(row['SUBDIVISION_Std'], row['ANNUAL']) 
    if pd.notna(row['ANNUAL']) and subdiv_normal.get(row['SUBDIVISION_Std'], 0) > 0 
    else 1.0, 
    axis=1
)

print(f"  ✓ Calculated anomaly for {len(df_rain_hist_subset)} records")
print(f"  Anomaly range: {df_rain_hist_subset['Rainfall_Anomaly'].min():.2f} - {df_rain_hist_subset['Rainfall_Anomaly'].max():.2f}")

# Create state-year anomaly lookup
def get_state_year_anomaly(state_std, year):
    """Get rainfall anomaly for a state-year combination"""
    if state_std not in subdivision_mapping:
        return 1.0  # No anomaly if state not found
    
    subdivisions = subdivision_mapping[state_std]
    
    # Filter historical data for this state's subdivisions and year
    mask = (
        (df_rain_hist_subset['SUBDIVISION_Std'].isin(subdivisions)) & 
        (df_rain_hist_subset['YEAR'] == year)
    )
    
    anomalies = df_rain_hist_subset.loc[mask, 'Rainfall_Anomaly']
    
    if len(anomalies) > 0:
        return anomalies.mean()
    return 1.0

# ============================================================================
# STEP 4: BUILD YEARLY DATASET WITH TEMPORAL VARIATION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: BUILDING YEARLY DATASET WITH TEMPORAL VARIATION")
print("=" * 80)

# Start with GWL yearly as base (has District, State, Year)
df_yearly = df_gwl_yearly.copy()
print(f"\n[4.1] Base: {len(df_yearly)} records from GWL Yearly")

# 4.2 Merge Population
print("\n[4.2] Merging Population...")
pop_cols = ['District_Std', 'Total_Population', 'Male_Population', 'Female_Population']
df_yearly = df_yearly.merge(
    df_pop[pop_cols],
    on='District_Std',
    how='left'
)

# Fill missing with state averages
state_pop_avg = df_yearly.groupby('State_Std')['Total_Population'].transform('mean')
df_yearly['Total_Population'].fillna(state_pop_avg, inplace=True)
df_yearly['Total_Population'].fillna(1000000, inplace=True)  # Final fallback

print(f"  ✓ Population merged")

# 4.3 Merge Normal Rainfall (district-level)
print("\n[4.3] Merging Normal Rainfall...")
df_yearly = df_yearly.merge(
    df_rain_normal[['District_Std', 'ANNUAL']].rename(columns={'ANNUAL': 'Normal_Rainfall_District'}),
    on='District_Std',
    how='left'
)

# Fill missing with existing Normal_Annual_Rainfall or state avg
df_yearly['Normal_Rainfall_District'].fillna(df_yearly['Normal_Annual_Rainfall'], inplace=True)
state_rain_avg = df_yearly.groupby('State_Std')['Normal_Rainfall_District'].transform('mean')
df_yearly['Normal_Rainfall_District'].fillna(state_rain_avg, inplace=True)
df_yearly['Normal_Rainfall_District'].fillna(1000, inplace=True)

print(f"  ✓ Normal rainfall merged")

# 4.4 Calculate ACTUAL ANNUAL RAINFALL using historical anomaly
print("\n[4.4] Calculating Actual Annual Rainfall with temporal variation...")

# Get anomaly for each row
df_yearly['Rainfall_Anomaly'] = df_yearly.apply(
    lambda row: get_state_year_anomaly(row['State_Std'], row['Year']),
    axis=1
)

# For years beyond 2015, extrapolate using recent trends + random variation
# Add some realistic variation for 2016-2024
np.random.seed(42)  # For reproducibility
years_after_2015 = df_yearly[df_yearly['Year'] > 2015].index

# Create variation based on climate trends (slightly decreasing with higher volatility)
for idx in years_after_2015:
    year = df_yearly.loc[idx, 'Year']
    # Base trend: slight decrease over time (climate change effect)
    base_trend = 1.0 - (year - 2015) * 0.005
    # Add random variation (±20%)
    random_factor = np.random.normal(0, 0.15)
    df_yearly.loc[idx, 'Rainfall_Anomaly'] = np.clip(base_trend + random_factor, 0.5, 1.5)

# Calculate actual rainfall
df_yearly['Annual_Rainfall'] = df_yearly['Normal_Rainfall_District'] * df_yearly['Rainfall_Anomaly']

print(f"  ✓ Annual rainfall calculated with variation")
print(f"  Rainfall range: {df_yearly['Annual_Rainfall'].min():.0f} - {df_yearly['Annual_Rainfall'].max():.0f} mm")

# 4.5 Calculate Water Demand with population growth
print("\n[4.5] Calculating Water Demand with temporal growth...")

# Per capita water requirement: 135 liters/day
BASE_LPCD = 135

# Add population growth effect (1.2% annual growth from 2011 baseline)
df_yearly['Population_Growth_Factor'] = 1 + (df_yearly['Year'] - 2011) * 0.012
df_yearly['Adjusted_Population'] = df_yearly['Total_Population'] * df_yearly['Population_Growth_Factor']

# Water demand in MCM (Million Cubic Meters)
df_yearly['Water_Demand_MCM'] = (
    df_yearly['Adjusted_Population'] * BASE_LPCD * 365 / 1_000_000_000
).round(2)

print(f"  ✓ Water demand calculated")
print(f"  Demand range: {df_yearly['Water_Demand_MCM'].min():.2f} - {df_yearly['Water_Demand_MCM'].max():.2f} MCM")

# 4.6 Calculate Water Supply with rainfall impact
print("\n[4.6] Calculating Water Supply with rainfall variation...")

# Base supply = Net GW Available (static from 2013 assessment)
# But adjust based on rainfall anomaly (good rainfall = more recharge)

# Supply varies with rainfall:
# - Normal rainfall (anomaly=1.0): 100% of base supply
# - Good rainfall (anomaly=1.2): 110% of base supply
# - Poor rainfall (anomaly=0.8): 85% of base supply

df_yearly['Supply_Adjustment'] = 0.5 + (df_yearly['Rainfall_Anomaly'] * 0.5)
df_yearly['Supply_Adjustment'] = df_yearly['Supply_Adjustment'].clip(0.7, 1.3)

# Convert Net_GW_Available from Ham to MCM (1 Ham = 0.01 MCM)
df_yearly['Net_GW_Available_MCM'] = df_yearly['Net_GW_Available'] * 0.01

# Apply rainfall adjustment
df_yearly['Water_Supply_MCM'] = (
    df_yearly['Net_GW_Available_MCM'] * df_yearly['Supply_Adjustment']
).round(2)

# Ensure positive supply
df_yearly['Water_Supply_MCM'] = df_yearly['Water_Supply_MCM'].clip(lower=1.0)

print(f"  ✓ Water supply calculated with temporal variation")
print(f"  Supply range: {df_yearly['Water_Supply_MCM'].min():.2f} - {df_yearly['Water_Supply_MCM'].max():.2f} MCM")

# 4.7 Calculate Scarcity Index
print("\n[4.7] Calculating Scarcity Index...")

df_yearly['Water_Deficit_MCM'] = (
    df_yearly['Water_Demand_MCM'] - df_yearly['Water_Supply_MCM']
).round(2)

df_yearly['Scarcity_Index'] = (
    df_yearly['Water_Demand_MCM'] / df_yearly['Water_Supply_MCM']
).round(4)

# Cap extreme values
df_yearly['Scarcity_Index'] = df_yearly['Scarcity_Index'].clip(0.01, 10.0)

df_yearly['Scarcity_Level'] = df_yearly['Scarcity_Index'].apply(get_scarcity_level)

print(f"  ✓ Scarcity Index calculated")
print(f"  Index range: {df_yearly['Scarcity_Index'].min():.4f} - {df_yearly['Scarcity_Index'].max():.4f}")

# 4.8 Verify temporal variation
print("\n[4.8] Verifying temporal variation...")

# Check a sample district
sample_district = 'GHAZIABAD'
sample_data = df_yearly[df_yearly['District_Std'] == sample_district].sort_values('Year')

if len(sample_data) > 0:
    print(f"\n  Sample: {sample_district}")
    print(f"  Scarcity Index range: {sample_data['Scarcity_Index'].min():.4f} - {sample_data['Scarcity_Index'].max():.4f}")
    print(f"  Scarcity Index std: {sample_data['Scarcity_Index'].std():.4f}")
    print(f"\n  First 5 years:")
    for _, row in sample_data.head(5).iterrows():
        print(f"    {int(row['Year'])}: Scarcity={row['Scarcity_Index']:.3f}, "
              f"Rainfall={row['Annual_Rainfall']:.0f}mm, "
              f"Anomaly={row['Rainfall_Anomaly']:.2f}")

# Check overall variation
district_std = df_yearly.groupby('District_Std')['Scarcity_Index'].std()
districts_with_variation = (district_std > 0.01).sum()
total_districts = len(district_std)

print(f"\n  Districts with meaningful variation: {districts_with_variation}/{total_districts} ({districts_with_variation/total_districts*100:.1f}%)")

# ============================================================================
# STEP 5: PREPARE FINAL YEARLY DATASET
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: PREPARING FINAL YEARLY DATASET")
print("=" * 80)

# Select and rename columns
final_yearly_cols = {
    'State': 'State',
    'District': 'District',
    'Year': 'Year',
    'Normal_Rainfall_District': 'Normal_Annual_Rainfall',
    'Annual_Rainfall': 'Annual_Rainfall',
    'Rainfall_Anomaly': 'Rainfall_Anomaly',
    'Net_GW_Available_MCM': 'Net_GW_Available_MCM',
    'Total_Annual_Draft': 'Total_Annual_Draft_MCM',
    'Stage_of_Extraction': 'Stage_of_Extraction',
    'Recharge_Monsoon': 'Recharge_Monsoon',
    'Recharge_NonMonsoon': 'Recharge_NonMonsoon',
    'Total_Population': 'Total_Population',
    'Adjusted_Population': 'Adjusted_Population',
    'Water_Demand_MCM': 'Water_Demand_MCM',
    'Water_Supply_MCM': 'Water_Supply_MCM',
    'Water_Deficit_MCM': 'Water_Deficit_MCM',
    'Scarcity_Index': 'Scarcity_Index',
    'Scarcity_Level': 'Scarcity_Level',
    'GWL_meters': 'GWL_meters',
    'GWL_Index': 'GWL_Index',
    'GWL_Trend': 'GWL_Trend'
}

# Get available columns
available_cols = [col for col in final_yearly_cols.keys() if col in df_yearly.columns]

df_final_yearly = df_yearly[available_cols].copy()
df_final_yearly = df_final_yearly.rename(columns=final_yearly_cols)

print(f"  ✓ Final Yearly Dataset: {len(df_final_yearly)} records × {len(df_final_yearly.columns)} columns")

# ============================================================================
# STEP 6: BUILD MONTHLY DATASET
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: BUILDING MONTHLY DATASET")
print("=" * 80)

df_monthly = df_gwl_monthly.copy()
print(f"\n[6.1] Base: {len(df_monthly)} records from GWL Monthly")

# Merge population
print("\n[6.2] Merging Population...")
df_monthly = df_monthly.merge(
    df_pop[['District_Std', 'Total_Population']],
    on='District_Std',
    how='left'
)
state_pop_avg = df_monthly.groupby('State_Std')['Total_Population'].transform('mean')
df_monthly['Total_Population'].fillna(state_pop_avg, inplace=True)
df_monthly['Total_Population'].fillna(1000000, inplace=True)

# Merge monthly normal rainfall
print("\n[6.3] Merging Monthly Rainfall Normals...")
month_cols = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
df_rain_melted = df_rain_normal.melt(
    id_vars=['District_Std', 'State_Std'],
    value_vars=month_cols,
    var_name='Month_Name',
    value_name='Rainfall_Normal'
)

month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
             'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
df_rain_melted['Month'] = df_rain_melted['Month_Name'].map(month_map)

df_monthly = df_monthly.merge(
    df_rain_melted[['District_Std', 'Month', 'Rainfall_Normal']],
    on=['District_Std', 'Month'],
    how='left'
)

# Fill missing rainfall with month averages
month_avg = df_rain_melted.groupby('Month')['Rainfall_Normal'].mean()
for month in range(1, 13):
    mask = (df_monthly['Month'] == month) & (df_monthly['Rainfall_Normal'].isna())
    df_monthly.loc[mask, 'Rainfall_Normal'] = month_avg.get(month, 100.0)

print(f"  ✓ Monthly rainfall merged")

# Get yearly rainfall anomaly for actual rainfall calculation
print("\n[6.4] Applying Yearly Rainfall Anomaly...")

yearly_anomaly = df_final_yearly[['District', 'Year', 'Rainfall_Anomaly']].drop_duplicates()
yearly_anomaly['District_Std'] = yearly_anomaly['District'].apply(normalize_name)

df_monthly = df_monthly.merge(
    yearly_anomaly[['District_Std', 'Year', 'Rainfall_Anomaly']],
    on=['District_Std', 'Year'],
    how='left'
)
df_monthly['Rainfall_Anomaly'].fillna(1.0, inplace=True)

# Add monthly variation (monsoon months get more/less variation)
np.random.seed(42)
monthly_variation = np.random.normal(0, 0.1, len(df_monthly))
df_monthly['Monthly_Variation'] = 1 + monthly_variation

# Monsoon months (Jun-Sep) more affected by anomaly
df_monthly['Monsoon_Factor'] = 1.0
df_monthly.loc[df_monthly['Month'].isin([6, 7, 8, 9]), 'Monsoon_Factor'] = 1.2

# Calculate actual monthly rainfall
df_monthly['Rainfall_mm'] = (
    df_monthly['Rainfall_Normal'] * 
    df_monthly['Rainfall_Anomaly'] * 
    df_monthly['Monthly_Variation'] *
    (1 + (df_monthly['Monsoon_Factor'] - 1) * (df_monthly['Rainfall_Anomaly'] - 1))
).round(2)

df_monthly['Rainfall_mm'] = df_monthly['Rainfall_mm'].clip(lower=0)

print(f"  ✓ Actual rainfall calculated")

# Calculate monthly demand and supply
print("\n[6.5] Calculating Monthly Demand and Supply...")

# Population growth
df_monthly['Pop_Growth'] = 1 + (df_monthly['Year'] - 2011) * 0.012
df_monthly['Adjusted_Pop'] = df_monthly['Total_Population'] * df_monthly['Pop_Growth']

# Monthly demand
df_monthly['Water_Demand_MCM'] = (
    df_monthly['Adjusted_Pop'] * 135 * 30 / 1_000_000_000
).round(3)

# Get yearly supply and distribute monthly
yearly_supply = df_final_yearly[['District', 'Year', 'Water_Supply_MCM']].drop_duplicates()
yearly_supply['District_Std'] = yearly_supply['District'].apply(normalize_name)
yearly_supply = yearly_supply.rename(columns={'Water_Supply_MCM': 'Yearly_Supply_MCM'})

df_monthly = df_monthly.merge(
    yearly_supply[['District_Std', 'Year', 'Yearly_Supply_MCM']],
    on=['District_Std', 'Year'],
    how='left'
)

# Distribute yearly supply to months (not equal - more in monsoon)
monthly_weights = {1: 0.06, 2: 0.06, 3: 0.05, 4: 0.05, 5: 0.05, 6: 0.10,
                   7: 0.14, 8: 0.14, 9: 0.12, 10: 0.08, 11: 0.08, 12: 0.07}

df_monthly['Monthly_Weight'] = df_monthly['Month'].map(monthly_weights)
df_monthly['Water_Supply_MCM'] = (
    df_monthly['Yearly_Supply_MCM'] * df_monthly['Monthly_Weight']
).round(3)

# Ensure positive
df_monthly['Water_Supply_MCM'].fillna(100, inplace=True)
df_monthly['Water_Supply_MCM'] = df_monthly['Water_Supply_MCM'].clip(lower=0.1)

# Calculate scarcity
df_monthly['Water_Deficit_MCM'] = (
    df_monthly['Water_Demand_MCM'] - df_monthly['Water_Supply_MCM']
).round(3)

df_monthly['Scarcity_Index'] = (
    df_monthly['Water_Demand_MCM'] / df_monthly['Water_Supply_MCM']
).round(4)

df_monthly['Scarcity_Index'] = df_monthly['Scarcity_Index'].clip(0.01, 10.0)
df_monthly['Scarcity_Level'] = df_monthly['Scarcity_Index'].apply(get_scarcity_level)

print(f"  ✓ Monthly scarcity calculated")

# Prepare final monthly columns
final_monthly_cols = [
    'State', 'District', 'Year', 'Month',
    'Rainfall_mm', 'GWL_meters', 'GWL_Index',
    'Water_Demand_MCM', 'Water_Supply_MCM', 'Water_Deficit_MCM',
    'Scarcity_Index', 'Scarcity_Level'
]

available_monthly = [col for col in final_monthly_cols if col in df_monthly.columns]
df_final_monthly = df_monthly[available_monthly].copy()

print(f"\n  ✓ Final Monthly Dataset: {len(df_final_monthly)} records × {len(df_final_monthly.columns)} columns")

# ============================================================================
# STEP 7: SAVE DATASETS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: SAVING DATASETS")
print("=" * 80)

yearly_path = os.path.join(CLEAN_PATH, 'FINAL_YEARLY_WATER_SCARCITY_DATASET.csv')
monthly_path = os.path.join(CLEAN_PATH, 'FINAL_MONTHLY_WATER_SCARCITY_DATASET.csv')

df_final_yearly.to_csv(yearly_path, index=False)
print(f"\n✓ Saved: {yearly_path}")

df_final_monthly.to_csv(monthly_path, index=False)
print(f"✓ Saved: {monthly_path}")

# ============================================================================
# STEP 8: VALIDATION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: VALIDATION REPORT")
print("=" * 80)

print("\n📊 YEARLY DATASET:")
print(f"  Records: {len(df_final_yearly):,}")
print(f"  Year Range: {df_final_yearly['Year'].min()} - {df_final_yearly['Year'].max()}")
print(f"  Districts: {df_final_yearly['District'].nunique()}")

print("\n  Scarcity Index Statistics:")
print(f"    Min: {df_final_yearly['Scarcity_Index'].min():.4f}")
print(f"    Max: {df_final_yearly['Scarcity_Index'].max():.4f}")
print(f"    Mean: {df_final_yearly['Scarcity_Index'].mean():.4f}")
print(f"    Std: {df_final_yearly['Scarcity_Index'].std():.4f}")

print("\n  Scarcity Level Distribution:")
print(df_final_yearly['Scarcity_Level'].value_counts().to_string())

# Check temporal variation
district_variation = df_final_yearly.groupby('District')['Scarcity_Index'].agg(['std', 'min', 'max'])
good_variation = (district_variation['std'] > 0.01).sum()
print(f"\n  Districts with temporal variation: {good_variation}/{len(district_variation)} ({good_variation/len(district_variation)*100:.1f}%)")

print("\n📊 MONTHLY DATASET:")
print(f"  Records: {len(df_final_monthly):,}")
print(f"  Year Range: {df_final_monthly['Year'].min()} - {df_final_monthly['Year'].max()}")

print("\n" + "=" * 80)
print("✅ DATA PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 80)
print("\n🎯 The datasets now have TEMPORAL VARIATION based on:")
print("  1. Historical rainfall anomalies (2000-2015)")
print("  2. Extended rainfall variation (2016-2024)")
print("  3. Population growth trends")
print("  4. Seasonal water availability patterns")
print("\n📈 Your visualizations should now show meaningful trends!")
print("=" * 80)