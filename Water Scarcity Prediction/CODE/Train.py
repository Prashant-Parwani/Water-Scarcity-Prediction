"""
================================================================================
UNIFIED WATER SCARCITY PREDICTION MODEL TRAINER - FIXED VERSION
Trains both Ensemble Regression & XGBoost Classification Models
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("UNIFIED WATER SCARCITY PREDICTION MODEL TRAINER")
print("Training: Ensemble Regression + XGBoost Classification")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_PATH = r"F:\Water Scarcity Prediction\DATA FILES"
DATA_PATH = os.path.join(BASE_PATH, "Clean Datasets")
OUTPUT_DIR = os.path.join(BASE_PATH, "ML Models")

os.makedirs(OUTPUT_DIR, exist_ok=True)

MONTHLY_DATA = os.path.join(DATA_PATH, "FINAL_MONTHLY_WATER_SCARCITY_DATASET.csv")
YEARLY_DATA = os.path.join(DATA_PATH, "FINAL_YEARLY_WATER_SCARCITY_DATASET.csv")

# ============================================================================
# PART 1: ENSEMBLE REGRESSION MODEL (TIME-SERIES PREDICTION)
# ============================================================================

print("\n" + "=" * 80)
print("PART 1: ENSEMBLE REGRESSION MODEL - TIME SERIES PREDICTION")
print("=" * 80)

# ----------------------------------------------------------------------------
# STEP 1.1: LOAD MONTHLY DATA
# ----------------------------------------------------------------------------

print("\n[STEP 1.1] Loading Monthly Dataset...")
df_monthly = pd.read_csv(MONTHLY_DATA)

print(f"✓ Loaded: {len(df_monthly):,} records")
print(f"  Year Range: {df_monthly['Year'].min()}-{df_monthly['Year'].max()}")
print(f"  Districts: {df_monthly['District'].nunique()}")
print(f"  Columns: {df_monthly.columns.tolist()}")

# ----------------------------------------------------------------------------
# STEP 1.2: OUTLIER HANDLING
# ----------------------------------------------------------------------------

print("\n[STEP 1.2] Handling Outliers...")

print(f"  Original Scarcity Index range: [{df_monthly['Scarcity_Index'].min():.2f}, {df_monthly['Scarcity_Index'].max():.2f}]")
df_monthly = df_monthly[df_monthly['Scarcity_Index'] <= 10].copy()
print(f"  Cleaned Scarcity Index range: [{df_monthly['Scarcity_Index'].min():.2f}, {df_monthly['Scarcity_Index'].max():.2f}]")

# ----------------------------------------------------------------------------
# STEP 1.3: ENHANCED FEATURE ENGINEERING
# ----------------------------------------------------------------------------

print("\n[STEP 1.3] Engineering Features...")

# Create datetime
df_monthly['Date'] = pd.to_datetime(df_monthly[['Year', 'Month']].assign(day=1))
df_monthly = df_monthly.sort_values(['District', 'Date']).reset_index(drop=True)

# Lag features (1, 2, 3, 6, 12 months)
def create_comprehensive_lags(group):
    for lag in [1, 2, 3, 6, 12]:
        group[f'Scarcity_Lag_{lag}'] = group['Scarcity_Index'].shift(lag)
        group[f'Rainfall_Lag_{lag}'] = group['Rainfall_mm'].shift(lag)
        group[f'GWL_Lag_{lag}'] = group['GWL_meters'].shift(lag)
        group[f'Supply_Lag_{lag}'] = group['Water_Supply_MCM'].shift(lag)
        group[f'Demand_Lag_{lag}'] = group['Water_Demand_MCM'].shift(lag)
    return group

df_monthly = df_monthly.groupby('District', group_keys=False).apply(create_comprehensive_lags)

# Rolling statistics (3, 6, 12 months)
def create_comprehensive_rolling(group):
    for window in [3, 6, 12]:
        group[f'Scarcity_MA_{window}'] = group['Scarcity_Index'].rolling(window, min_periods=1).mean()
        group[f'Scarcity_STD_{window}'] = group['Scarcity_Index'].rolling(window, min_periods=1).std()
        group[f'Rainfall_MA_{window}'] = group['Rainfall_mm'].rolling(window, min_periods=1).mean()
        group[f'Rainfall_STD_{window}'] = group['Rainfall_mm'].rolling(window, min_periods=1).std()
        group[f'GWL_MA_{window}'] = group['GWL_meters'].rolling(window, min_periods=1).mean()
        group[f'Supply_MA_{window}'] = group['Water_Supply_MCM'].rolling(window, min_periods=1).mean()
    
    # Trends
    group['Scarcity_Trend_3'] = group['Scarcity_Index'].diff(3)
    group['Scarcity_Trend_6'] = group['Scarcity_Index'].diff(6)
    group['Scarcity_Trend_12'] = group['Scarcity_Index'].diff(12)
    group['Rainfall_Trend_3'] = group['Rainfall_mm'].diff(3)
    group['Rainfall_Trend_6'] = group['Rainfall_mm'].diff(6)
    
    return group

df_monthly = df_monthly.groupby('District', group_keys=False).apply(create_comprehensive_rolling)

# Seasonal features
df_monthly['Month_Sin'] = np.sin(2 * np.pi * df_monthly['Month'] / 12)
df_monthly['Month_Cos'] = np.cos(2 * np.pi * df_monthly['Month'] / 12)
df_monthly['Is_Monsoon'] = df_monthly['Month'].isin([6, 7, 8, 9]).astype(int)
df_monthly['Is_Summer'] = df_monthly['Month'].isin([3, 4, 5]).astype(int)
df_monthly['Is_Winter'] = df_monthly['Month'].isin([11, 12, 1, 2]).astype(int)
df_monthly['Is_PostMonsoon'] = df_monthly['Month'].isin([10, 11]).astype(int)

# Year features for trend
df_monthly['Year_Normalized'] = (df_monthly['Year'] - df_monthly['Year'].min()) / (df_monthly['Year'].max() - df_monthly['Year'].min())
df_monthly['Year_Squared'] = df_monthly['Year_Normalized'] ** 2

# Interaction features
df_monthly['Demand_Supply_Ratio'] = df_monthly['Water_Demand_MCM'] / (df_monthly['Water_Supply_MCM'] + 0.001)
df_monthly['Rainfall_GWL_Interaction'] = df_monthly['Rainfall_mm'] * df_monthly['GWL_Index']
df_monthly['Monsoon_Rainfall'] = df_monthly['Is_Monsoon'] * df_monthly['Rainfall_mm']
df_monthly['Supply_Demand_Gap'] = df_monthly['Water_Supply_MCM'] - df_monthly['Water_Demand_MCM']

# Year-on-Year changes
def add_yoy_changes(group):
    group['Scarcity_YoY'] = group['Scarcity_Index'].diff(12)
    group['Rainfall_YoY'] = group['Rainfall_mm'].diff(12)
    return group

df_monthly = df_monthly.groupby('District', group_keys=False).apply(add_yoy_changes)

df_model_monthly = df_monthly.dropna()

print(f"✓ After feature engineering: {len(df_model_monthly):,} samples")

# ----------------------------------------------------------------------------
# STEP 1.4: FEATURE SELECTION FOR ENSEMBLE (FIXED - removed Net_GW_Available_MCM)
# ----------------------------------------------------------------------------

print("\n[STEP 1.4] Selecting Features for Ensemble Model...")

# FIXED: Only use columns that exist in monthly dataset
ensemble_feature_cols = [
    # Base features (available in monthly)
    'Rainfall_mm', 'GWL_meters', 'GWL_Index', 'Water_Demand_MCM', 'Water_Supply_MCM',
    
    # Lag features (most important)
    'Scarcity_Lag_1', 'Scarcity_Lag_2', 'Scarcity_Lag_3', 'Scarcity_Lag_6', 'Scarcity_Lag_12',
    'Rainfall_Lag_1', 'Rainfall_Lag_2', 'Rainfall_Lag_3',
    'GWL_Lag_1', 'GWL_Lag_3',
    'Supply_Lag_1', 'Demand_Lag_1',
    
    # Rolling statistics
    'Scarcity_MA_3', 'Scarcity_MA_6', 'Scarcity_MA_12',
    'Scarcity_STD_3', 'Scarcity_STD_6',
    'Rainfall_MA_3', 'Rainfall_MA_6', 'Rainfall_MA_12',
    'Rainfall_STD_3', 'Rainfall_STD_6',
    'GWL_MA_3', 'GWL_MA_6',
    'Supply_MA_3', 'Supply_MA_6',
    
    # Trends
    'Scarcity_Trend_3', 'Scarcity_Trend_6', 'Scarcity_Trend_12',
    'Rainfall_Trend_3', 'Rainfall_Trend_6',
    
    # Seasonal
    'Month_Sin', 'Month_Cos', 'Is_Monsoon', 'Is_Summer', 'Is_Winter', 'Is_PostMonsoon',
    
    # Year trends
    'Year_Normalized', 'Year_Squared',
    
    # Interactions
    'Demand_Supply_Ratio', 'Rainfall_GWL_Interaction', 'Monsoon_Rainfall', 'Supply_Demand_Gap',
    
    # YoY changes
    'Scarcity_YoY', 'Rainfall_YoY'
]

# Verify all columns exist
missing_cols = [col for col in ensemble_feature_cols if col not in df_model_monthly.columns]
if missing_cols:
    print(f"  ⚠️ Missing columns (removing): {missing_cols}")
    ensemble_feature_cols = [col for col in ensemble_feature_cols if col in df_model_monthly.columns]

X_ensemble = df_model_monthly[ensemble_feature_cols].values
y_ensemble = df_model_monthly['Scarcity_Index'].values

print(f"✓ Features: {len(ensemble_feature_cols)}")
print(f"✓ Shape: {X_ensemble.shape}")

# ----------------------------------------------------------------------------
# STEP 1.5: TRAIN-TEST SPLIT (TEMPORAL)
# ----------------------------------------------------------------------------

print("\n[STEP 1.5] Train-Test Split (Temporal)...")

split = int(len(X_ensemble) * 0.85)
X_train_ens, X_test_ens = X_ensemble[:split], X_ensemble[split:]
y_train_ens, y_test_ens = y_ensemble[:split], y_ensemble[split:]

print(f"✓ Train: {len(X_train_ens):,} | Test: {len(X_test_ens):,}")

# ----------------------------------------------------------------------------
# STEP 1.6: SCALING
# ----------------------------------------------------------------------------

print("\n[STEP 1.6] Applying RobustScaler...")

scaler_X_ens = RobustScaler()
scaler_y_ens = RobustScaler()

X_train_ens_sc = scaler_X_ens.fit_transform(X_train_ens)
X_test_ens_sc = scaler_X_ens.transform(X_test_ens)
y_train_ens_sc = scaler_y_ens.fit_transform(y_train_ens.reshape(-1, 1)).flatten()

# ----------------------------------------------------------------------------
# STEP 1.7: TRAIN ENSEMBLE MODELS
# ----------------------------------------------------------------------------

print("\n[STEP 1.7] Training Ensemble Models...")

# Gradient Boosting
print("  [1/3] Gradient Boosting Regressor...")
gb_model = GradientBoostingRegressor(
    n_estimators=400,
    max_depth=7,
    learning_rate=0.03,
    subsample=0.85,
    min_samples_split=8,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42,
    verbose=0
)
gb_model.fit(X_train_ens_sc, y_train_ens_sc)

# Random Forest
print("  [2/3] Random Forest Regressor...")
rf_model = RandomForestRegressor(
    n_estimators=400,
    max_depth=25,
    min_samples_split=8,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    verbose=0
)
rf_model.fit(X_train_ens_sc, y_train_ens_sc)

# Extra Trees
print("  [3/3] Extra Trees Regressor...")
et_model = ExtraTreesRegressor(
    n_estimators=300,
    max_depth=25,
    min_samples_split=8,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    verbose=0
)
et_model.fit(X_train_ens_sc, y_train_ens_sc)

print("✓ All ensemble models trained!")

# ----------------------------------------------------------------------------
# STEP 1.8: EVALUATE ENSEMBLE
# ----------------------------------------------------------------------------

print("\n[STEP 1.8] Evaluating Ensemble...")

gb_pred = gb_model.predict(X_test_ens_sc)
rf_pred = rf_model.predict(X_test_ens_sc)
et_pred = et_model.predict(X_test_ens_sc)

# Weighted ensemble (optimized weights)
y_pred_ens_sc = 0.45 * gb_pred + 0.35 * rf_pred + 0.20 * et_pred
y_pred_ens = scaler_y_ens.inverse_transform(y_pred_ens_sc.reshape(-1, 1)).flatten()
y_pred_ens = np.clip(y_pred_ens, 0, 10)

rmse_ens = np.sqrt(mean_squared_error(y_test_ens, y_pred_ens))
mae_ens = mean_absolute_error(y_test_ens, y_pred_ens)
r2_ens = r2_score(y_test_ens, y_pred_ens)

print(f"\n📊 ENSEMBLE PERFORMANCE:")
print(f"  RMSE: {rmse_ens:.4f}")
print(f"  MAE: {mae_ens:.4f}")
print(f"  R²: {r2_ens:.4f}")

tolerance = 0.15
rel_error = np.abs(y_pred_ens - y_test_ens) / (y_test_ens + 0.1)
accuracy_ens = (rel_error <= tolerance).sum() / len(y_test_ens) * 100
print(f"  Accuracy (±15%): {accuracy_ens:.2f}%")

# ----------------------------------------------------------------------------
# STEP 1.9: SAVE ENSEMBLE MODELS
# ----------------------------------------------------------------------------

print("\n[STEP 1.9] Saving Ensemble Models...")

with open(os.path.join(OUTPUT_DIR, 'gradient_boosting_final.pkl'), 'wb') as f:
    pickle.dump(gb_model, f)
print("✓ Saved: gradient_boosting_final.pkl")

with open(os.path.join(OUTPUT_DIR, 'random_forest_final.pkl'), 'wb') as f:
    pickle.dump(rf_model, f)
print("✓ Saved: random_forest_final.pkl")

with open(os.path.join(OUTPUT_DIR, 'extra_trees_final.pkl'), 'wb') as f:
    pickle.dump(et_model, f)
print("✓ Saved: extra_trees_final.pkl")

with open(os.path.join(OUTPUT_DIR, 'ensemble_scaler_X.pkl'), 'wb') as f:
    pickle.dump(scaler_X_ens, f)
print("✓ Saved: ensemble_scaler_X.pkl")

with open(os.path.join(OUTPUT_DIR, 'ensemble_scaler_y.pkl'), 'wb') as f:
    pickle.dump(scaler_y_ens, f)
print("✓ Saved: ensemble_scaler_y.pkl")

with open(os.path.join(OUTPUT_DIR, 'ensemble_features.pkl'), 'wb') as f:
    pickle.dump(ensemble_feature_cols, f)
print("✓ Saved: ensemble_features.pkl")

# Save predictions
results_ens = pd.DataFrame({
    'Actual': y_test_ens,
    'Predicted': y_pred_ens,
    'Error': y_pred_ens - y_test_ens,
    'Abs_Error': np.abs(y_pred_ens - y_test_ens)
})
results_ens.to_csv(os.path.join(OUTPUT_DIR, 'Ensemble_Predictions.csv'), index=False)

print("\n✅ ENSEMBLE REGRESSION MODEL COMPLETE!")

# ============================================================================
# PART 2: XGBOOST CLASSIFICATION MODEL
# ============================================================================

print("\n" + "=" * 80)
print("PART 2: XGBOOST CLASSIFICATION MODEL - SCARCITY LEVEL PREDICTION")
print("=" * 80)

# ----------------------------------------------------------------------------
# STEP 2.1: LOAD YEARLY DATA
# ----------------------------------------------------------------------------

print("\n[STEP 2.1] Loading Yearly Dataset...")
df_yearly = pd.read_csv(YEARLY_DATA)

print(f"✓ Loaded: {len(df_yearly):,} records")
print(f"  Years: {df_yearly['Year'].min()}-{df_yearly['Year'].max()}")
print(f"  Districts: {df_yearly['District'].nunique()}")
print(f"  Columns: {df_yearly.columns.tolist()}")

# ----------------------------------------------------------------------------
# STEP 2.2: DATA CLEANING
# ----------------------------------------------------------------------------

print("\n[STEP 2.2] Cleaning Data...")

initial = len(df_yearly)
df_yearly = df_yearly[df_yearly['Scarcity_Level'] != 'Unknown'].copy()
print(f"  Removed {initial - len(df_yearly)} 'Unknown' records")

# Cap outliers
df_yearly.loc[df_yearly['Scarcity_Index'] > 10, 'Scarcity_Index'] = 10

print("\n📊 Scarcity Level Distribution:")
for level, count in df_yearly['Scarcity_Level'].value_counts().items():
    pct = count / len(df_yearly) * 100
    print(f"  {level:12s}: {count:6,} ({pct:5.2f}%)")

# ----------------------------------------------------------------------------
# STEP 2.3: FEATURE ENGINEERING FOR CLASSIFICATION (FIXED)
# ----------------------------------------------------------------------------

print("\n[STEP 2.3] Feature Engineering for Classification...")

# Base features that exist in yearly dataset
base_features_xgb = []

# Check which columns exist and add them
possible_base_features = [
    'Normal_Annual_Rainfall', 'Annual_Rainfall', 'Rainfall_Anomaly',
    'Net_GW_Available_MCM', 'Total_Annual_Draft_MCM',
    'Stage_of_Extraction', 'Recharge_Monsoon', 'Recharge_NonMonsoon',
    'Water_Demand_MCM', 'Water_Supply_MCM', 'Water_Deficit_MCM',
    'GWL_meters', 'GWL_Index', 'GWL_Trend', 
    'Total_Population', 'Adjusted_Population'
]

for col in possible_base_features:
    if col in df_yearly.columns:
        base_features_xgb.append(col)
    else:
        print(f"  ⚠️ Column not found (skipping): {col}")

print(f"  ✓ Base features available: {len(base_features_xgb)}")

# Create derived features
df_yearly['Monsoon_Dependency'] = df_yearly['Recharge_Monsoon'] / (df_yearly['Recharge_Monsoon'] + df_yearly['Recharge_NonMonsoon'] + 0.001)
df_yearly['Supply_Demand_Ratio'] = df_yearly['Water_Supply_MCM'] / (df_yearly['Water_Demand_MCM'] + 0.001)
df_yearly['Deficit_Percentage'] = (df_yearly['Water_Deficit_MCM'] / (df_yearly['Water_Demand_MCM'] + 0.001)) * 100
df_yearly['Extraction_Pressure'] = df_yearly['Total_Annual_Draft_MCM'] / (df_yearly['Net_GW_Available_MCM'] + 0.001)
df_yearly['Water_Stress_Index'] = df_yearly['Stage_of_Extraction'] * df_yearly['Deficit_Percentage'].abs() / 100
df_yearly['Per_Capita_Water'] = df_yearly['Water_Supply_MCM'] / (df_yearly['Total_Population'] + 1) * 1000000
df_yearly['GWL_Stress'] = df_yearly['GWL_Index'] * df_yearly['Stage_of_Extraction']

# Year-based features
df_yearly['Year_Normalized'] = (df_yearly['Year'] - df_yearly['Year'].min()) / (df_yearly['Year'].max() - df_yearly['Year'].min())

# Rainfall-based features (new dataset has these)
if 'Rainfall_Anomaly' in df_yearly.columns:
    df_yearly['Rainfall_Stress'] = 1 / (df_yearly['Rainfall_Anomaly'] + 0.001)
    df_yearly['Rainfall_Deficit'] = df_yearly['Normal_Annual_Rainfall'] - df_yearly['Annual_Rainfall']

derived_features = [
    'Monsoon_Dependency', 'Supply_Demand_Ratio', 'Deficit_Percentage',
    'Extraction_Pressure', 'Water_Stress_Index', 'Per_Capita_Water',
    'GWL_Stress', 'Year_Normalized'
]

# Add rainfall features if available
if 'Rainfall_Anomaly' in df_yearly.columns:
    derived_features.extend(['Rainfall_Stress', 'Rainfall_Deficit'])

all_features_xgb = base_features_xgb + derived_features

# Remove any features with all NaN
valid_features = []
for col in all_features_xgb:
    if col in df_yearly.columns and df_yearly[col].notna().sum() > 0:
        valid_features.append(col)
    else:
        print(f"  ⚠️ Removing invalid feature: {col}")

all_features_xgb = valid_features

df_model_yearly = df_yearly[all_features_xgb + ['Scarcity_Level']].dropna()

print(f"✓ Features: {len(all_features_xgb)}")
print(f"✓ Samples: {len(df_model_yearly):,}")

# ----------------------------------------------------------------------------
# STEP 2.4: ENCODE TARGET
# ----------------------------------------------------------------------------

print("\n[STEP 2.4] Encoding Target...")

le_xgb = LabelEncoder()
y_xgb = le_xgb.fit_transform(df_model_yearly['Scarcity_Level'])

label_map = dict(enumerate(le_xgb.classes_))
print("\n📋 Encoding Map:")
for code, name in label_map.items():
    print(f"  {code} → {name}")

# ----------------------------------------------------------------------------
# STEP 2.5: TRAIN-TEST SPLIT
# ----------------------------------------------------------------------------

print("\n[STEP 2.5] Train-Test Split (Stratified)...")

X_xgb = df_model_yearly[all_features_xgb].values
X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
    X_xgb, y_xgb, test_size=0.2, random_state=42, stratify=y_xgb
)

print(f"✓ Train: {len(X_train_xgb):,} | Test: {len(X_test_xgb):,}")

# ----------------------------------------------------------------------------
# STEP 2.6: SMOTE BALANCING
# ----------------------------------------------------------------------------

print("\n[STEP 2.6] SMOTE Balancing...")

try:
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_xgb_bal, y_train_xgb_bal = smote.fit_resample(X_train_xgb, y_train_xgb)
    print(f"✓ After SMOTE: {len(X_train_xgb_bal):,} samples")
    
    print("\n  Class distribution after SMOTE:")
    unique, counts = np.unique(y_train_xgb_bal, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"    {label_map[cls]}: {cnt}")
        
except ImportError:
    print("⚠️  SMOTE not available. Install: pip install imbalanced-learn")
    X_train_xgb_bal, y_train_xgb_bal = X_train_xgb, y_train_xgb

# ----------------------------------------------------------------------------
# STEP 2.7: SCALING
# ----------------------------------------------------------------------------

print("\n[STEP 2.7] Applying RobustScaler...")

scaler_xgb = RobustScaler()
X_train_xgb_sc = scaler_xgb.fit_transform(X_train_xgb_bal)
X_test_xgb_sc = scaler_xgb.transform(X_test_xgb)

print("✓ RobustScaler applied")

# ----------------------------------------------------------------------------
# STEP 2.8: TRAIN XGBOOST
# ----------------------------------------------------------------------------

print("\n[STEP 2.8] Training XGBoost Classifier...")

try:
    import xgboost as xgb
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=1,
        gamma=0.15,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective='multi:softprob',
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
        verbosity=0
    )
    
    xgb_model.fit(
        X_train_xgb_sc, y_train_xgb_bal,
        eval_set=[(X_test_xgb_sc, y_test_xgb)],
        verbose=False
    )
    
    model_name = "XGBoost"
    
except ImportError:
    print("⚠️  XGBoost not installed. Using Random Forest as fallback.")
    print("   Install with: pip install xgboost")
    
    from sklearn.ensemble import RandomForestClassifier
    xgb_model = RandomForestClassifier(
        n_estimators=400,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train_xgb_sc, y_train_xgb_bal)
    model_name = "Random Forest (Fallback)"

print(f"✓ {model_name} trained!")

# ----------------------------------------------------------------------------
# STEP 2.9: EVALUATE CLASSIFICATION MODEL
# ----------------------------------------------------------------------------

print("\n[STEP 2.9] Evaluating Classification Model...")

y_pred_xgb = xgb_model.predict(X_test_xgb_sc)
y_pred_proba_xgb = xgb_model.predict_proba(X_test_xgb_sc)

accuracy_xgb = accuracy_score(y_test_xgb, y_pred_xgb)
precision_xgb, recall_xgb, f1_xgb, _ = precision_recall_fscore_support(
    y_test_xgb, y_pred_xgb, average='weighted', zero_division=0
)

print(f"\n📊 {model_name.upper()} PERFORMANCE:")
print(f"  Accuracy: {accuracy_xgb:.4f} ({accuracy_xgb*100:.2f}%)")
print(f"  Precision: {precision_xgb:.4f}")
print(f"  Recall: {recall_xgb:.4f}")
print(f"  F1-Score: {f1_xgb:.4f}")

# Per-class metrics
print("\n📋 PER-CLASS METRICS:")
print("─" * 75)
print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
print("─" * 75)

for i in np.unique(y_test_xgb):
    mask = y_test_xgb == i
    pred_mask = y_pred_xgb == i
    
    tp = np.sum(mask & pred_mask)
    fp = np.sum(~mask & pred_mask)
    fn = np.sum(mask & ~pred_mask)
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_val = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    supp = mask.sum()
    
    print(f"{label_map[i]:<15} {prec:<12.4f} {rec:<12.4f} {f1_val:<12.4f} {supp:<10}")

print("─" * 75)

# ----------------------------------------------------------------------------
# STEP 2.10: SAVE CLASSIFICATION MODELS
# ----------------------------------------------------------------------------

print("\n[STEP 2.10] Saving Classification Models...")

with open(os.path.join(OUTPUT_DIR, 'xgboost_classifier_final.pkl'), 'wb') as f:
    pickle.dump(xgb_model, f)
print("✓ Saved: xgboost_classifier_final.pkl")

with open(os.path.join(OUTPUT_DIR, 'xgb_label_encoder.pkl'), 'wb') as f:
    pickle.dump(le_xgb, f)
print("✓ Saved: xgb_label_encoder.pkl")

with open(os.path.join(OUTPUT_DIR, 'xgb_scaler.pkl'), 'wb') as f:
    pickle.dump(scaler_xgb, f)
print("✓ Saved: xgb_scaler.pkl")

with open(os.path.join(OUTPUT_DIR, 'xgb_features.pkl'), 'wb') as f:
    pickle.dump(all_features_xgb, f)
print("✓ Saved: xgb_features.pkl")

# Save predictions
results_xgb = pd.DataFrame({
    'Actual': [label_map[i] for i in y_test_xgb],
    'Predicted': [label_map[i] for i in y_pred_xgb],
    'Correct': y_test_xgb == y_pred_xgb,
    'Confidence_%': (y_pred_proba_xgb.max(axis=1) * 100).round(2)
})
results_xgb.to_csv(os.path.join(OUTPUT_DIR, 'XGBoost_Predictions_Final.csv'), index=False)

# Save confusion matrix
cm_xgb = confusion_matrix(y_test_xgb, y_pred_xgb)
cm_df_xgb = pd.DataFrame(
    cm_xgb,
    index=[label_map[i] for i in range(len(label_map))],
    columns=[label_map[i] for i in range(len(label_map))]
)
cm_df_xgb.to_csv(os.path.join(OUTPUT_DIR, 'XGBoost_Confusion_Matrix.csv'))

print("\n✅ XGBOOST CLASSIFICATION MODEL COMPLETE!")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
print("=" * 80)

print("\n📊 FINAL PERFORMANCE SUMMARY:")
print("\n1. ENSEMBLE REGRESSION MODEL:")
print(f"   - R²: {r2_ens:.4f}")
print(f"   - RMSE: {rmse_ens:.4f}")
print(f"   - MAE: {mae_ens:.4f}")
print(f"   - Accuracy (±15%): {accuracy_ens:.2f}%")

print(f"\n2. {model_name.upper()} CLASSIFICATION MODEL:")
print(f"   - Accuracy: {accuracy_xgb*100:.2f}%")
print(f"   - Precision: {precision_xgb:.4f}")
print(f"   - Recall: {recall_xgb:.4f}")
print(f"   - F1-Score: {f1_xgb:.4f}")

print("\n📁 SAVED FILES:")
print("   Ensemble Models:")
print("   - gradient_boosting_final.pkl")
print("   - random_forest_final.pkl")
print("   - extra_trees_final.pkl")
print("   - ensemble_scaler_X.pkl")
print("   - ensemble_scaler_y.pkl")
print("   - ensemble_features.pkl")
print("\n   Classification Models:")
print("   - xgboost_classifier_final.pkl")
print("   - xgb_label_encoder.pkl")
print("   - xgb_scaler.pkl")
print("   - xgb_features.pkl")

print("\n🚀 READY FOR STREAMLIT DEPLOYMENT!")
print("=" * 80)