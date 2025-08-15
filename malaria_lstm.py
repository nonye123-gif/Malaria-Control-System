import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, LayerNormalization, Attention, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
import os
import warnings

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define folder paths
VISUALIZATION_FOLDER = 'stacked_charts'
MODEL_FOLDER = 'model_folder'

# Create folders if they don't exist
os.makedirs(VISUALIZATION_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Enhanced date parsing function (copied from dataset generation for consistency)
def safe_parse_date(date_str):
    try:
        if pd.isna(date_str):
            return pd.NaT
        
        if isinstance(date_str, (int, float)):
            base_date = pd.Timestamp('1899-12-30')
            return base_date + pd.Timedelta(days=date_str)
        
        if isinstance(date_str, str):
            date_str = date_str.replace('<br>', '-').replace('--', '-').strip()
            
            for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d', '%d-%m-%Y', '%m-%d-%Y'):
                try:
                    return pd.to_datetime(date_str, format=fmt, errors='raise')
                except:
                    continue
        
        return pd.to_datetime(date_str, errors='coerce')
    except:
        return pd.NaT

# Load and preprocess data with robust date handling and visualizations
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    print("Dataset Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    if 'Severity' not in df.columns:
        raise ValueError(
            "The 'Severity' column is missing from the dataset. "
            "Please ensure your CSV file contains a column named 'Severity' "
            "for the target variable."
        )
    
    # Replace 'Mild' with 'Low' to ensure consistency with dataset generation
    df['Severity'] = df['Severity'].replace('Mild', 'Low')

    if df['Severity'].isnull().all():
        raise ValueError(
            "The 'Severity' column contains only missing values after preprocessing. "
            "Cannot encode an empty target variable."
        )
    
    print("\nProcessing dates...")
    df['Date'] = df['Date'].apply(safe_parse_date)
    
    invalid_dates = df[df['Date'].isna()]
    print(f"Found {len(invalid_dates)} rows with invalid dates")
    
    if not invalid_dates.empty:
        print("Sample of rows with invalid dates:")
        print(invalid_dates[['Patient ID', 'Date']].head())
        
        # Fill invalid dates with mode, but only if there are valid dates to compute mode from
        if not df['Date'].mode().empty:
            mode_date = df['Date'].mode()[0]
            print(f"\nFilling invalid dates with: {mode_date}")
            df['Date'] = df['Date'].fillna(mode_date)
        else:
            print("\nNo valid dates to compute mode. Dropping rows with invalid dates.")
            df.dropna(subset=['Date'], inplace=True) # Drop if no mode can be found

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Day_of_year'] = df['Date'].dt.dayofyear
    df['Week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
    
    # Visualize date distribution
    plt.figure(figsize=(12, 6))
    df['Year'].value_counts().sort_index().plot(kind='bar')
    plt.title('Case Distribution by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Cases')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_FOLDER, 'year_distribution.png'))
    plt.close()
    print("Visualization saved: year_distribution.png")
    
    print("\nHandling missing values...")
    missing_report = df.isnull().sum()
    print("Missing values per column:")
    print(missing_report[missing_report > 0])
    
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == 'object':
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                print(f"Filled missing {col} with mode: {mode_val}")
            else:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"Filled missing {col} with median: {median_val}")
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values After Handling')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_FOLDER, 'missing_values_after.png'))
    plt.close()
    print("Visualization saved: missing_values_after.png")
    
    # Create drug features (still needed for the model input)
    drugs = ['Artemether', 'Lumefantrine', 'Quinine', 'Fansidar', 'Mefloquine', 
             'Doxycycline', 'Primaquine', 'Chlorproguanil', 'Amodiaquine']
    
    print("\nCreating drug features...")
    for drug in drugs:
        df[drug] = df['Drugs Administered'].str.contains(drug, case=False, na=False).astype(int)
        print(f"- Created {drug} feature: {df[drug].sum()} cases")
    
    df['Age_BodyTemp_Interaction'] = df['Age'] * df['Body Temp (°C)']
    print(f"- Created Age_BodyTemp_Interaction feature.")

    # --- Data Leakage Prevention ---
    # Features to be removed to avoid data leakage
    leakage_features = ['Outcome', 'Drugs Administered', 'High_Risk_Age', 'Hyperpyrexia']
    
    # If 'Symptom_Score' is still present in the loaded CSV (e.g., from older data), remove it
    if 'Symptom_Score' in df.columns:
        leakage_features.append('Symptom_Score')
        print("Warning: 'Symptom_Score' found in input data and will be removed to prevent leakage.")

    # Drop leakage features from the DataFrame
    df_processed = df.drop(columns=leakage_features, errors='ignore')
    print(f"Removed leakage features: {leakage_features}")

    # Define features for preprocessing (excluding leakage features)
    # Diagnosis is kept as it's consistently 'Malaria' in the generated data
    categorical_features = ['Gender', 'Genotype', 'Blood Group', 'LGA', 'Season', 'Diagnosis'] 
    numerical_features = ['Age', 'Body Temp (°C)', 'Latitude', 'Longitude', 
                          'Rainfall (mm)', 'Climate Temp (°C)', 
                          'Year', 'Month', 'Day', 'Day_of_year', 'Week_of_year',
                          'Age_BodyTemp_Interaction']
    
    # Ensure drug features are added to the list of features for the model,
    # but the original 'Drugs Administered' column is dropped.
    feature_cols_for_model = categorical_features + numerical_features + drugs 

    # Create a robust preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('drugs_pass', 'passthrough', drugs) # Pass through the newly created drug binary features
        ],
        remainder='drop' # Drop any other columns not specified
    )
    
    # Apply preprocessing
    X = preprocessor.fit_transform(df_processed[feature_cols_for_model])
    
    severity_encoder = LabelEncoder()
    df_processed['Severity_encoded'] = severity_encoder.fit_transform(df_processed['Severity'])
    y = df_processed['Severity_encoded'] 
    
    class_weights = compute_class_weight('balanced', classes=np.unique(df_processed['Severity_encoded']), 
                                         y=df_processed['Severity_encoded'])
    class_weights = dict(enumerate(class_weights))
    print(f"\nClass weights: {class_weights}")

    # Get feature names after one-hot encoding
    num_feature_names = numerical_features
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    drug_feature_names_processed = drugs 
    
    feature_names = np.concatenate([
        np.array(num_feature_names, dtype=str),
        cat_feature_names.astype(str),
        np.array(drug_feature_names_processed, dtype=str)
    ])
    
    # Visualize class distribution before SMOTE
    plt.figure(figsize=(8, 6))
    sns.countplot(x=df_processed['Severity'], order=severity_encoder.classes_)
    plt.title('Severity Distribution Before SMOTE')
    plt.xlabel('Severity')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_FOLDER, 'severity_distribution_before_smote.png'))
    plt.close()
    print("Visualization saved: severity_distribution_before_smote.png")

    return X, y, df_processed, feature_names, class_weights, preprocessor, severity_encoder

# Plot training history
def plot_training_history(history, model_type="LSTM"):
    plt.figure(figsize=(15, 12))
    
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'auc']
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(3, 2, i)
        plt.plot(history.history[metric], label=f'Training {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
        plt.title(f'Training and Validation {metric.capitalize()} ({model_type})')
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_FOLDER, f'training_history_{model_type.lower()}.png'))
    plt.close()
    print(f"Training history plot saved as 'training_history_{model_type.lower()}.png'")

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_FOLDER, 'confusion_matrix.png'))
    plt.close()
    print("Visualization saved: confusion_matrix.png")

# Plot ROC curves
def plot_roc_curves(y_test_labels, y_score, classes):
    plt.figure(figsize=(10, 8))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = len(classes)
    
    y_test_one_hot = to_categorical(y_test_labels, num_classes=n_classes)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    colors = ['blue', 'red', 'green']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{classes[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_FOLDER, 'roc_curves.png'))
    plt.close()
    print("Visualization saved: roc_curves.png")

# Plot Precision-Recall curves
def plot_pr_curves(y_test_labels, y_score, classes):
    plt.figure(figsize=(10, 8))
    
    precision = dict()
    recall = dict()
    average_precision = dict()
    n_classes = len(classes)

    y_test_one_hot = to_categorical(y_test_labels, num_classes=n_classes)
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_one_hot[:, i], y_score[:, i])
        average_precision[i] = auc(recall[i], precision[i])
    
    colors = ['blue', 'red', 'green']
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label=f'{classes[i]} (AP = {average_precision[i]:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_FOLDER, 'pr_curves.png'))
    plt.close()
    print("Visualization saved: pr_curves.png")

# Evaluate model
def evaluate_model(model, X_test, y_test_labels, severity_classes, model_type="LSTM"):
    y_pred_proba = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_proba, axis=1)
    
    print(f"\nClassification Report ({model_type}):")
    print(classification_report(y_test_labels, y_pred_classes, 
                                target_names=severity_classes))
    
    plot_confusion_matrix(y_test_labels, y_pred_classes, severity_classes)
    
    plot_roc_curves(y_test_labels, y_pred_proba, severity_classes)
    
    plot_pr_curves(y_test_labels, y_pred_proba, severity_classes)

# Function to build the LSTM model with Attention
def build_lstm_model_with_attention(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    # LSTM Layer
    # Using return_sequences=True to pass sequence to Attention layer
    lstm_out = LSTM(units=128, return_sequences=True)(inputs) 
    lstm_out = LayerNormalization()(lstm_out)
    lstm_out = Dropout(0.3)(lstm_out)
    
    # Attention Mechanism
    # Query, Key, Value are all from the LSTM output
    attention_output = Attention(name='attention_layer')([lstm_out, lstm_out])
    
    # Flatten the attention output for Dense layers
    attention_output = tf.keras.layers.Flatten()(attention_output) # Flatten after attention

    # Dense layers for classification
    x = Dense(128, activation='relu')(attention_output)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax', name='classification_output')(x)
    
    model = Model(inputs=inputs, outputs=output)
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')])
    return model

# Test model on new data
def test_model(model, new_data, preprocessor, severity_encoder):
    new_df_raw = pd.DataFrame([new_data])

    # Re-create drug features for the new data point
    drugs_list = ['Artemether', 'Lumefantrine', 'Quinine', 'Fansidar', 'Mefloquine', 
                  'Doxycycline', 'Primaquine', 'Chlorproguanil', 'Amodiaquine']
    for drug in drugs_list:
        new_df_raw[drug] = 1 if drug in new_data.get('Drugs Administered', '') else 0
    
    # Add date-derived features
    new_df_raw['Date'] = safe_parse_date(new_data['Date'])
    new_df_raw['Year'] = new_df_raw['Date'].dt.year
    new_df_raw['Month'] = new_df_raw['Date'].dt.month
    new_df_raw['Day'] = new_df_raw['Date'].dt.day
    new_df_raw['Day_of_year'] = new_df_raw['Date'].dt.dayofyear
    new_df_raw['Week_of_year'] = new_df_raw['Date'].dt.isocalendar().week.astype(int)
    
    # Add engineered interaction feature
    new_df_raw['Age_BodyTemp_Interaction'] = new_df_raw['Age'] * new_df_raw['Body Temp (°C)']

    # Define features that the preprocessor expects (excluding leakage features)
    categorical_features_expected = ['Gender', 'Genotype', 'Blood Group', 'LGA', 'Season', 'Diagnosis']
    numerical_features_expected = ['Age', 'Body Temp (°C)', 'Latitude', 'Longitude', 
                                   'Rainfall (mm)', 'Climate Temp (°C)', 
                                   'Year', 'Month', 'Day', 'Day_of_year', 'Week_of_year', 
                                   'Age_BodyTemp_Interaction']
    
    # Combine all expected features for the preprocessor
    all_expected_features = numerical_features_expected + categorical_features_expected + drugs_list

    # Create a DataFrame with all expected columns, filling missing ones with defaults
    df_for_transform = pd.DataFrame(columns=all_expected_features)
    for col in all_expected_features:
        if col in new_df_raw.columns:
            df_for_transform[col] = new_df_raw[col]
        else:
            if col in drugs_list:
                df_for_transform[col] = 0 # Default for drug flags
            elif col in numerical_features_expected:
                df_for_transform[col] = 0.0 # Default for numerical
            elif col in categorical_features_expected:
                df_for_transform[col] = 'Unknown' # Default for categorical
    
    # Ensure correct dtypes before transformation
    for col in numerical_features_expected:
        df_for_transform[col] = pd.to_numeric(df_for_transform[col], errors='coerce').fillna(0.0)
    for col in categorical_features_expected:
        df_for_transform[col] = df_for_transform[col].astype(str).fillna('Unknown')

    # Transform the new data using the fitted preprocessor
    X_new_processed = preprocessor.transform(df_for_transform)
    
    # Reshape for LSTM input: (samples, timesteps, features)
    X_new_lstm_input = X_new_processed.reshape(X_new_processed.shape[0], 1, X_new_processed.shape[1])

    # Predict using the LSTM model
    prediction_proba = model.predict(X_new_lstm_input) 
    predicted_class_label = np.argmax(prediction_proba, axis=1)[0]
    severity = severity_encoder.inverse_transform([predicted_class_label])[0]
    
    probabilities_dict = {
        severity_encoder.classes_[i]: float(prediction_proba[0][i]) 
        for i in range(len(severity_encoder.classes_))
    }

    return {
        'severity': severity,
        'probabilities': probabilities_dict
    }

# Main execution
if __name__ == "__main__":
    print("Starting malaria severity prediction pipeline (LSTM + Attention)...")
    print("="*60)
    
    X_2d, y_labels, df, feature_names, class_weights, preprocessor, severity_encoder = \
        None, None, None, None, None, None, None

    try:
        print("\nLoading and preprocessing data...")
        # Ensure the path matches where your dataset generation script saves the CSV
        X_2d, y_labels, df, feature_names, class_weights, preprocessor, severity_encoder = load_and_preprocess_data(
            os.path.join(MODEL_FOLDER, 'complete_malaria_dataset_3000.csv')
        )
        print(f"Data shape after preprocessing (2D): {X_2d.shape}")
        print(f"Target shape (labels): {y_labels.shape}")
        print(f"Feature names: {feature_names[:10]}... (total: {len(feature_names)})")
    except Exception as e:
        print(f"Error during data loading: {e}")
        import traceback
        traceback.print_exc()
        print("\nProgram terminated due to errors during data loading.")
        exit() # Exit if data loading fails

    print("\nSplitting data...")
    X_train_2d, X_test_2d, y_train_labels, y_test_labels = train_test_split(
        X_2d, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    print(f"Train shape (2D): {X_train_2d.shape}, Test shape (2D): {X_test_2d.shape}")

    print("\nApplying SMOTE to training data...")
    sm = SMOTE(random_state=42) 
    X_train_res, y_train_res_labels = sm.fit_resample(X_train_2d, y_train_labels)
    print(f"Train shape after SMOTE: {X_train_res.shape}, {y_train_res_labels.shape}")

    # Visualize class distribution after SMOTE
    plt.figure(figsize=(8, 6))
    sns.countplot(x=severity_encoder.inverse_transform(y_train_res_labels), 
                  order=severity_encoder.classes_)
    plt.title('Severity Distribution After SMOTE')
    plt.xlabel('Severity')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_FOLDER, 'severity_distribution_after_smote.png'))
    plt.close()
    print("Visualization saved: severity_distribution_after_smote.png")

    # Reshape data for LSTM input: (samples, timesteps, features)
    # For tabular data, timesteps is usually 1
    num_features = X_train_res.shape[1]
    X_train_lstm_input = X_train_res.reshape(X_train_res.shape[0], 1, num_features)
    X_test_lstm_input = X_test_2d.reshape(X_test_2d.shape[0], 1, num_features)
    
    # Convert y to one-hot encoding for LSTM training
    y_train_one_hot = to_categorical(y_train_res_labels, num_classes=len(severity_encoder.classes_))
    y_test_one_hot = to_categorical(y_test_labels, num_classes=len(severity_encoder.classes_))


    # --- Build and Train LSTM Model with Attention ---
    print("\n--- Building and Training LSTM Model with Attention ---")
    lstm_model = build_lstm_model_with_attention(input_shape=(1, num_features), 
                                                num_classes=len(severity_encoder.classes_))
    
    lstm_callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=1e-7, verbose=1),
        ModelCheckpoint(os.path.join(MODEL_FOLDER, 'best_lstm_attention_model.keras'), save_best_only=True, 
                        monitor='val_loss', verbose=1)
    ]
    
    lstm_history = lstm_model.fit(
        X_train_lstm_input, y_train_one_hot,
        epochs=200, 
        batch_size=32,
        validation_split=0.2,
        class_weight=class_weights,
        callbacks=lstm_callbacks,
        verbose=1
    )
    
    plot_training_history(lstm_history, model_type="LSTM_Attention")

    print("\nEvaluating the LSTM model with Attention...")
    severity_classes = severity_encoder.classes_
    evaluate_model(lstm_model, X_test_lstm_input, y_test_labels, severity_classes, model_type="LSTM_Attention") 
    
    print("\nTesting with new patient data (LSTM + Attention pipeline)...")
    new_patient = {
        'Age': 8,
        'Gender': 'Male',
        'Genotype': 'AS',
        'Blood Group': 'O+',
        'Body Temp (°C)': 40.2,
        'Latitude': 4.92,
        'Longitude': 6.26,
        'Rainfall (mm)': 320.0,
        'Climate Temp (°C)': 29.8,
        'LGA': 'Yenagoa',
        'Season': 'Rainy',
        'Date': '2023-07-15',
        'Drugs Administered': 'Artemether, Lumefantrine', 
        'Diagnosis': 'Malaria' 
    }
    
    result = test_model(lstm_model, new_patient, preprocessor, severity_encoder)
    print("\nNew Patient Prediction:")
    print(f"Predicted Severity: {result['severity']}")
    print(f"Probabilities:")
    for cls, prob in result['probabilities'].items():
        print(f"- {cls}: {prob:.4f}")
    
    # Save the LSTM model
    lstm_model.save(os.path.join(MODEL_FOLDER, 'malaria_severity_lstm_attention_model.keras'))
    print(f"\nSaved LSTM model to '{MODEL_FOLDER}/malaria_severity_lstm_attention_model.keras'")
    
    print("\nPipeline completed successfully!")
