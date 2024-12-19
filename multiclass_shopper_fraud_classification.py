import os
import time
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import snowflake.connector
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_recall_fscore_support,
    average_precision_score,
    precision_recall_curve,
    log_loss  # Added for computing training loss
)
from sklearn.preprocessing import LabelEncoder
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from dotenv import load_dotenv
from sqlalchemy import create_engine
from lightgbm import log_evaluation  
import warnings

warnings.filterwarnings('ignore')

# ================== Data Loading ==================

start_time = time.time()

# Load environment variables from .env file
load_dotenv()

# Verify that essential environment variables are loaded
required_vars = [
    'SNOWFLAKE_USER',
    'SNOWFLAKE_ACCOUNT',
    'SNOWFLAKE_DATABASE',
    'SNOWFLAKE_WAREHOUSE',
    'SNOWFLAKE_ROLE',
    'SNOWFLAKE_SCHEMA',
    'SNOWFLAKE_PRIVATE_KEY_PATH'
]
for var in required_vars:
    if not os.getenv(var):
        raise EnvironmentError(f"Required environment variable {var} is not set.")

# Load the private key for Snowflake connection
private_key_path = os.path.expanduser(os.getenv('SNOWFLAKE_PRIVATE_KEY_PATH'))
with open(private_key_path, "rb") as key_file:
    private_key = serialization.load_pem_private_key(
        key_file.read(),
        password=os.getenv('SNOWFLAKE_PRIVATE_KEY_PASSPHRASE').encode()
        if os.getenv('SNOWFLAKE_PRIVATE_KEY_PASSPHRASE') else None,
        backend=default_backend()
    )
    pkb = private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )

# Create a SQLAlchemy engine using key pair authentication
engine = create_engine(
    f'snowflake://{os.getenv("SNOWFLAKE_USER")}'
    f'@{os.getenv("SNOWFLAKE_ACCOUNT")}/{os.getenv("SNOWFLAKE_DATABASE")}',
    connect_args={
        'private_key': pkb,
        'warehouse': os.getenv("SNOWFLAKE_WAREHOUSE"),
        'role': os.getenv("SNOWFLAKE_ROLE"),
        'schema': os.getenv("SNOWFLAKE_SCHEMA"),
    }
)

# Define your SQL query for three datasets (up to 30K, up to 500K, up to 1.5M)
query = """
SELECT *
FROM SANDBOX_DB.AHMADSAJEDI.SHOPPER_FRAUD_CLASSIFICATION_AGG_M1
ORDER BY delivery_created_date_time_utc ASC
"""

# query = """
# SELECT *
# FROM SANDBOX_DB.AHMADSAJEDI.SHOPPER_FRAUD_CLASSIFICATION_FULL_NONFRAUD_AGG_M1
# ORDER BY delivery_created_date_time_utc ASC
# """

# query = """
# SELECT *
# FROM SANDBOX_DB.AHMADSAJEDI.SHOPPER_FRAUD_CLASSIFICATION_FULL_AGG_M1
# ORDER BY delivery_created_date_time_utc ASC
# """

# Execute the query and fetch the data into a pandas DataFrame
data = pd.read_sql(query, engine)

end_time = time.time()
print(f"Data Loading Time: {end_time - start_time:.2f} seconds")

# ================== Data Preparation ==================

start_time = time.time()

# Remove 'delivery_created_date_time_utc' from the dataset
if 'delivery_created_date_time_utc' in data.columns:
    data.drop('delivery_created_date_time_utc', axis=1, inplace=True)

# Identify the target column and ID columns
target_column = 'fraud_vector'
id_columns = ['order_delivery_id', 'batch_id', 'shopper_id']

# Encode the target variable
target_encoder = LabelEncoder()
data[target_column] = target_encoder.fit_transform(data[target_column])

# Convert 'region_id' to 'category' dtype if it's in the data
if 'region_id' in data.columns:
    data['region_id'] = data['region_id'].astype('category')

# Identify categorical and numerical columns
categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_columns = data.select_dtypes(include=['number']).columns.tolist()

# Remove target and ID columns from feature lists
for col in [target_column] + id_columns:
    if col in categorical_columns:
        categorical_columns.remove(col)
    if col in numeric_columns:
        numeric_columns.remove(col)

# Ensure all categorical columns are of type 'category'
for col in categorical_columns:
    data[col] = data[col].astype('category')

# Split features and target
X = data.drop([target_column] + id_columns, axis=1)
y = data[target_column]

end_time = time.time()
print(f"Data Preparation Time: {end_time - start_time:.2f} seconds")

# ================== Data Splitting ==================

start_time = time.time()

# Calculate index for splitting the data
total_length = len(data)
train_end = int(0.7 * total_length)  # First 70% for training

# Time-based splits
X_train = X.iloc[:train_end].reset_index(drop=True)
y_train = y.iloc[:train_end].reset_index(drop=True)

X_test = X.iloc[train_end:].reset_index(drop=True)
y_test = y.iloc[train_end:].reset_index(drop=True)

end_time = time.time()
print(f"Data Splitting Time: {end_time - start_time:.2f} seconds")

# ================== Model Training ==================
# 1. the ebst hyperparameter for the model on SHOPPER_FRAUD_CLASSIFICATION_AGG_M1 (up to 30K)
best_params = {
    'objective': 'multiclass',
    'num_class': len(np.unique(y)),
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'device': 'cpu',
    'is_unbalance': True,
    'verbose': -1,
    'learning_rate': 0.014673672445495577,
    'num_leaves': 128,
    'max_depth': 12,
    'min_data_in_leaf': 100,
    'feature_fraction': 0.6002933605844709,
    'bagging_fraction': 0.9019153707902258,
    'bagging_freq': 4,
    'lambda_l1': 5.084063300383103e-06,
    'lambda_l2': 0.015662588748170638,
    'min_gain_to_split': 0.9095436389736504
}

# 2. the ebst hyperparameter for the model on SHOPPER_FRAUD_CLASSIFICATION_FULL_NONFRAUD_AGG_M1(up to 500K)
# best_params = {
#     'objective': 'multiclass',
#     'num_class': len(np.unique(y)),
#     'metric': 'multi_logloss',
#     'boosting_type': 'gbdt',
#     'device': 'cpu',
#     'is_unbalance': True,
#     'verbose': -1,
#     'learning_rate': 0.006732795304011321,
#     'num_leaves': 61,
#     'max_depth': 12,
#     'min_data_in_leaf': 30,
#     'feature_fraction': 0.7127518290986533,
#     'bagging_fraction': 0.7204684435174533,
#     'bagging_freq': 7,
#     'lambda_l1': 6.84680812174637e-06,
#     'lambda_l2': 0.5254075879024248,
#     'min_gain_to_split': 0.02618080955415214
# }



 # 3. the ebst hyperparameter for the model on SHOPPER_FRAUD_CLASSIFICATION_FULL_AGG_M1(up to 1.5M)
# best_params = {
#     'objective': 'multiclass',
#     'num_class': len(np.unique(y)),
#     'metric': 'multi_logloss',
#     'boosting_type': 'gbdt',
#     'device': 'cpu',
#     'is_unbalance': True,
#     'verbose': -1,
#     'learning_rate': 0.01529444436516868,
#     'num_leaves': 107,
#     'max_depth': 15,
#     'min_data_in_leaf': 43,
#     'feature_fraction': 0.636380794598689,
#     'bagging_fraction': 0.9513900502235751,
#     'bagging_freq': 2,
#     'lambda_l1': 4.5496892503442985e-07,
#     'lambda_l2': 2.6911600070939785e-08,
#     'min_gain_to_split': 0.32519314542307187
# }

# Prepare LightGBM dataset
lgb_train = lgb.Dataset(
    X_train,
    label=y_train,
    categorical_feature=categorical_columns,
    free_raw_data=False
)

print("\nTraining the model on the training data...")

final_model = lgb.train(
    best_params,
    lgb_train,
    num_boost_round=1000,
    callbacks=[
        log_evaluation(period=100)
    ]
)

print("\nModel training completed.")

# ================== Model Evaluation Function ==================


def evaluate_model(y_true, y_pred_proba, y_pred_labels, dataset_name):
    from sklearn.metrics import (
        classification_report, precision_score, recall_score, roc_auc_score,
        average_precision_score, confusion_matrix
    )
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Classification report with per-class precision and recall
    report = classification_report(
        y_true,
        y_pred_labels,
        target_names=target_encoder.classes_,
        output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()
    print(f"\nClassification Report on {dataset_name} Set:\n")
    print(report_df)

    # Extract per-class metrics
    precision_per_class = report_df.iloc[:-3]['precision']
    recall_per_class = report_df.iloc[:-3]['recall']
    f1_per_class = report_df.iloc[:-3]['f1-score']

    # Calculate micro and macro average precision and recall
    precision_micro = precision_score(y_true, y_pred_labels, average='micro')
    recall_micro = recall_score(y_true, y_pred_labels, average='micro')
    precision_macro = precision_score(y_true, y_pred_labels, average='macro')
    recall_macro = recall_score(y_true, y_pred_labels, average='macro')

    print(f"\nMicro-Average Precision on {dataset_name} Set: {precision_micro:.4f}")
    print(f"Micro-Average Recall on {dataset_name} Set: {recall_micro:.4f}")
    print(f"Macro-Average Precision on {dataset_name} Set: {precision_macro:.4f}")
    print(f"Macro-Average Recall on {dataset_name} Set: {recall_macro:.4f}")

    # Calculate ROC AUC scores
    try:
        roc_auc_micro = roc_auc_score(
            y_true,
            y_pred_proba,
            multi_class='ovr',
            average='micro'
        )
        roc_auc_macro = roc_auc_score(
            y_true,
            y_pred_proba,
            multi_class='ovr',
            average='macro'
        )
        roc_auc_weighted = roc_auc_score(
            y_true,
            y_pred_proba,
            multi_class='ovr',
            average='weighted'
        )
        print(f"\nMicro-Averaged ROC AUC on {dataset_name} Set: {roc_auc_micro:.4f}")
        print(f"Macro-Averaged ROC AUC on {dataset_name} Set: {roc_auc_macro:.4f}")
        print(f"Weighted ROC AUC on {dataset_name} Set: {roc_auc_weighted:.4f}")
    except ValueError as e:
        print(f"\nROC AUC Score could not be computed on {dataset_name} Set: {e}")

    # Calculate Precision-Recall AUC scores
    try:
        pr_auc_micro = average_precision_score(
            y_true,
            y_pred_proba,
            average='micro'
        )
        pr_auc_macro = average_precision_score(
            y_true,
            y_pred_proba,
            average='macro'
        )
        pr_auc_weighted = average_precision_score(
            y_true,
            y_pred_proba,
            average='weighted'
        )
        print(f"\nMicro-Averaged Precision-Recall AUC on {dataset_name} Set: {pr_auc_micro:.4f}")
        print(f"Macro-Averaged Precision-Recall AUC on {dataset_name} Set: {pr_auc_macro:.4f}")
        print(f"Weighted Precision-Recall AUC on {dataset_name} Set: {pr_auc_weighted:.4f}")
    except ValueError as e:
        print(f"\nPrecision-Recall AUC Score could not be computed on {dataset_name} Set: {e}")

    # Calculate per-class Precision-Recall AUC
    pr_auc_per_class = {}
    for i, class_label in enumerate(np.unique(y_true)):
        pr_auc = average_precision_score(
            (y_true == class_label).astype(int),
            y_pred_proba[:, i]
        )
        class_name = target_encoder.inverse_transform([class_label])[0]
        pr_auc_per_class[class_name] = pr_auc

    pr_auc_df = pd.DataFrame.from_dict(pr_auc_per_class, orient='index', columns=['PR AUC'])
    print(f"\nPer-Class Precision-Recall AUC on {dataset_name} Set:")
    print(pr_auc_df)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=target_encoder.classes_,
        yticklabels=target_encoder.classes_
    )
    plt.title(f"Confusion Matrix on {dataset_name} Set")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{dataset_name.lower()}.png')
    plt.close()

# def evaluate_model(y_true, y_pred_proba, y_pred_labels, dataset_name):
#     # Classification report with per-class precision and recall
#     report = classification_report(
#         y_true,
#         y_pred_labels,
#         target_names=target_encoder.classes_
#     )
#     print(f"\nClassification Report on {dataset_name} Set:\n", report)
    
#     # Extract precision, recall, and F1-score for each class
#     precision, recall, f1_score, support = precision_recall_fscore_support(
#         y_true,
#         y_pred_labels,
#         labels=np.unique(y_true)
#     )
    
#     # Create DataFrame for visualization
#     metrics_df = pd.DataFrame({
#         'Class': target_encoder.inverse_transform(np.unique(y_true)),
#         'Precision': precision,
#         'Recall': recall,
#         'F1-Score': f1_score,
#         'Support': support
#     })
    
#     print(f"\nPrecision, Recall, and F1-Score per Class on {dataset_name} Set:")
#     print(metrics_df)
    
#     # Calculate weighted and macro-averaged AUC
#     try:
#         auc_weighted = roc_auc_score(
#             y_true,
#             y_pred_proba,
#             multi_class='ovr',
#             average='weighted'
#         )
#         auc_macro = roc_auc_score(
#             y_true,
#             y_pred_proba,
#             multi_class='ovr',
#             average='macro'
#         )
#         print(f"\nWeighted AUC on {dataset_name} Set: {auc_weighted:.4f}")
#         print(f"Macro-Averaged AUC on {dataset_name} Set: {auc_macro:.4f}")
#     except ValueError as e:
#         print(f"\nAUC Score could not be computed on {dataset_name} Set: {e}")
    
#     # Calculate PR AUC for each class
#     pr_auc_per_class = {}
#     for i, class_label in enumerate(np.unique(y_true)):
#         precision_curve, recall_curve, _ = precision_recall_curve(
#             (y_true == class_label).astype(int),
#             y_pred_proba[:, i]
#         )
#         pr_auc = average_precision_score(
#             (y_true == class_label).astype(int),
#             y_pred_proba[:, i]
#         )
#         pr_auc_per_class[target_encoder.inverse_transform([class_label])[0]] = pr_auc
    
#     pr_auc_df = pd.DataFrame.from_dict(pr_auc_per_class, orient='index', columns=['PR AUC'])
#     print(f"\nPrecision-Recall AUC per Class on {dataset_name} Set:")
#     print(pr_auc_df)
    
#     # Calculate macro and weighted average PR AUC
#     pr_auc_weighted = average_precision_score(
#         y_true,
#         y_pred_proba,
#         average='weighted'
#     )
#     pr_auc_macro = average_precision_score(
#         y_true,
#         y_pred_proba,
#         average='macro'
#     )
#     print(f"\nWeighted PR AUC on {dataset_name} Set: {pr_auc_weighted:.4f}")
#     print(f"Macro-Averaged PR AUC on {dataset_name} Set: {pr_auc_macro:.4f}")
    
#     # Confusion Matrix
#     conf_matrix = confusion_matrix(y_true, y_pred_labels)
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(
#         conf_matrix,
#         annot=True,
#         fmt='d',
#         cmap='Blues',
#         xticklabels=target_encoder.classes_,
#         yticklabels=target_encoder.classes_
#     )
#     plt.title(f"Confusion Matrix on {dataset_name} Set")
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.tight_layout()
#     plt.savefig(f'confusion_matrix_{dataset_name.lower()}.png')
#     plt.close()

# ================== Model Evaluation on Training Set ==================

print("\nEvaluating the model on the training set...")

# Predictions on training set
y_pred_train = final_model.predict(X_train, num_iteration=final_model.best_iteration)
y_pred_labels_train = np.argmax(y_pred_train, axis=1)

evaluate_model(y_train, y_pred_train, y_pred_labels_train, "Training")

# ================== Model Evaluation on Test Set ==================

print("\nEvaluating the model on the test set...")

# Predictions on test set
y_pred_test = final_model.predict(X_test, num_iteration=final_model.best_iteration)
y_pred_labels_test = np.argmax(y_pred_test, axis=1)

evaluate_model(y_test, y_pred_test, y_pred_labels_test, "Test")

# ================== Feature Importance Analysis ==================

print("\nPerforming Feature Importance Analysis...")

def plot_feature_importance(model, feature_names, top_n=20):
    """Basic feature importance plot using gain"""
    # Get feature importance
    importance = model.feature_importance(importance_type='gain')
    
    # Create DataFrame with feature names and importance
    feature_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    # Plot top N features
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_imp.head(top_n))
    plt.title(f'Top {top_n} Feature Importance (Gain)')
    plt.xlabel('Importance (Gain)')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('feature_importance_gain.png')
    plt.close()
    
    return feature_imp

def analyze_feature_importance(model, feature_names):
    """Analyze different types of feature importance"""
    importance_types = ['gain', 'split'] 
    importance_dict = {}
    
    for imp_type in importance_types:
        importance = model.feature_importance(importance_type=imp_type)
        importance_dict[imp_type] = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)
    

    fig, axes = plt.subplots(1, 2, figsize=(15, 6)) 
    
    for ax, (imp_type, imp_df) in zip(axes, importance_dict.items()):
        sns.barplot(x='Importance', y='Feature', 
                   data=imp_df.head(10), ax=ax)
        ax.set_title(f'Top 10 Features ({imp_type})')
        ax.set_xlabel(f'Importance ({imp_type})')
        ax.set_ylabel('Features')
    
    plt.tight_layout()
    plt.savefig('feature_importance_comparison.png')
    plt.close()
    
    return importance_dict

def analyze_shap_values(model, X_data, feature_names, max_display=20):
    """Calculate and plot SHAP values"""
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    
    # Plot summary for each class
    if isinstance(shap_values, list):  # For multi-class
        for i, class_shap in enumerate(shap_values):
            plt.figure(figsize=(12, 8))
            shap.summary_plot(class_shap, X_data, feature_names=feature_names,
                            max_display=max_display, show=False)
            plt.title(f'SHAP Values for Class {target_encoder.inverse_transform([i])[0]}')
            plt.tight_layout()
            plt.savefig(f'shap_summary_class_{i}.png')
            plt.close()
    
    return shap_values

def analyze_feature_stability(model, X_train, X_test):
    """Analyze feature importance stability between train and test sets"""
    train_imp = pd.DataFrame({
        'Feature': X_train.columns,
        'Train_Importance': model.feature_importance(importance_type='gain')
    })
    
    test_imp = pd.DataFrame({
        'Feature': X_test.columns,
        'Test_Importance': model.feature_importance(importance_type='gain')
    })
    
    # Merge and calculate stability metrics
    stability = train_imp.merge(test_imp, on='Feature')
    stability['Importance_Diff'] = abs(
        stability['Train_Importance'] - stability['Test_Importance']
    )
    stability['Importance_Ratio'] = (
        stability['Train_Importance'] / 
        stability['Test_Importance'].replace(0, 1e-10)
    )
    
    return stability.sort_values('Train_Importance', ascending=False)

def comprehensive_feature_analysis(model, X_train, X_test):
    """Perform comprehensive feature importance analysis"""
    feature_names = X_train.columns
    results = {}
    
    # 1. Basic Feature Importance
    print("\nCalculating basic feature importance...")
    results['basic_importance'] = plot_feature_importance(model, feature_names)
    
    # 2. Multiple Importance Types
    print("Analyzing different importance metrics...")
    results['detailed_importance'] = analyze_feature_importance(model, feature_names)
    
    # 3. SHAP Analysis
    print("Calculating SHAP values...")
    # Use a sample of data for SHAP analysis if dataset is large
    sample_size = min(1000, len(X_test))
    X_sample = X_test.sample(sample_size, random_state=42)
    results['shap_values'] = analyze_shap_values(model, X_sample, feature_names)
    
    # 4. Feature Stability Analysis
    print("Analyzing feature stability...")
    results['stability_scores'] = analyze_feature_stability(model, X_train, X_test)
    
    return results

# Perform the comprehensive analysis
feature_analysis = comprehensive_feature_analysis(final_model, X_train, X_test)

# Print and save results
print("\nTop 20 Most Important Features (by Gain):")
print(feature_analysis['basic_importance'].head(20))

print("\nFeatures with Largest Train/Test Differences:")
stability_df = feature_analysis['stability_scores']
print(stability_df.sort_values('Importance_Diff', ascending=False).head(10))

# Save detailed results to CSV
feature_analysis['basic_importance'].to_csv('feature_importance_gain.csv')
feature_analysis['stability_scores'].to_csv('feature_stability.csv')

# Save importance results for each metric type
for imp_type, imp_df in feature_analysis['detailed_importance'].items():
    imp_df.to_csv(f'feature_importance_{imp_type}.csv')

print("\nFeature importance analysis completed. Results saved to CSV files and plots.")

# ================== Learning Curves ==================

# # Function to compute training loss over iterations
# def compute_training_loss_over_iterations(model, data, labels):
#     loss_list = []
#     num_iterations = model.current_iteration()
#     for i in range(1, num_iterations + 1):
#         y_pred = model.predict(data, num_iteration=i)
#         loss = log_loss(labels, y_pred)
#         loss_list.append(loss)
#     return loss_list

# # Compute training loss over iterations
# print("\nComputing training loss over iterations...")
# train_loss = compute_training_loss_over_iterations(final_model, X_train, y_train)
# epochs = range(1, len(train_loss) + 1)
# plt.figure(figsize=(8, 6))
# plt.plot(epochs, train_loss, label='Training Loss')
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
# plt.title('Loss over Iterations for Training Set')
# plt.legend()
# plt.tight_layout()
# plt.savefig('learning_curve_loss_training.png')
# plt.close()

# # Function to compute PR AUC over iterations
# def compute_pr_auc_over_iterations(model, data, labels):
#     pr_auc_list = []
#     num_iterations = model.current_iteration()
#     for i in range(1, num_iterations + 1):
#         y_pred = model.predict(data, num_iteration=i)
#         pr_auc = average_precision_score(labels, y_pred, average='macro')
#         pr_auc_list.append(pr_auc)
#     return pr_auc_list

# # Compute PR AUC over iterations for training set
# print("\nComputing PR AUC over iterations for training set...")
# train_pr_auc = compute_pr_auc_over_iterations(final_model, X_train, y_train)
# epochs = range(1, len(train_pr_auc) + 1)
# plt.figure(figsize=(8, 6))
# plt.plot(epochs, train_pr_auc, label='Training PR AUC')
# plt.xlabel('Iterations')
# plt.ylabel('PR AUC')
# plt.title('PR AUC over Iterations for Training Set')
# plt.legend()
# plt.tight_layout()
# plt.savefig('learning_curve_pr_auc_training.png')
# plt.close()

# # Compute PR AUC over iterations for test set
# print("\nComputing PR AUC over iterations for test set...")
# test_pr_auc = compute_pr_auc_over_iterations(final_model, X_test, y_test)
# epochs = range(1, len(test_pr_auc) + 1)
# plt.figure(figsize=(8, 6))
# plt.plot(epochs, test_pr_auc, label='Test PR AUC')
# plt.xlabel('Iterations')
# plt.ylabel('PR AUC')
# plt.title('PR AUC over Iterations for Test Set')
# plt.legend()
# plt.tight_layout()
# plt.savefig('learning_curve_pr_auc_test.png')
# plt.close()

# print("\nLearning curves generated and saved.")

# ================== Save the Final Model ==================

# Save the final model
final_model.save_model('lightgbm_fraud_classifier.txt')

print("\nModel evaluation and learning curves completed.")

# import os
# import time
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import lightgbm as lgb
# import snowflake.connector
# from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_fscore_support
# from sklearn.preprocessing import LabelEncoder
# from cryptography.hazmat.primitives import serialization
# from cryptography.hazmat.backends import default_backend
# from dotenv import load_dotenv
# from sqlalchemy import create_engine
# from lightgbm import early_stopping, log_evaluation
# import warnings

# warnings.filterwarnings('ignore')

# # ================== Data Loading ==================

# start_time = time.time()

# # Load environment variables from .env file
# load_dotenv()

# # Verify that essential environment variables are loaded
# required_vars = [
#     'SNOWFLAKE_USER',
#     'SNOWFLAKE_ACCOUNT',
#     'SNOWFLAKE_DATABASE',
#     'SNOWFLAKE_WAREHOUSE',
#     'SNOWFLAKE_ROLE',
#     'SNOWFLAKE_SCHEMA',
#     'SNOWFLAKE_PRIVATE_KEY_PATH'
# ]
# for var in required_vars:
#     if not os.getenv(var):
#         raise EnvironmentError(f"Required environment variable {var} is not set.")

# # Load the private key for Snowflake connection
# private_key_path = os.path.expanduser(os.getenv('SNOWFLAKE_PRIVATE_KEY_PATH'))
# with open(private_key_path, "rb") as key_file:
#     private_key = serialization.load_pem_private_key(
#         key_file.read(),
#         password=os.getenv('SNOWFLAKE_PRIVATE_KEY_PASSPHRASE').encode()
#         if os.getenv('SNOWFLAKE_PRIVATE_KEY_PASSPHRASE') else None,
#         backend=default_backend()
#     )
#     # Convert the private key to the format required by Snowflake
#     pkb = private_key.private_bytes(
#         encoding=serialization.Encoding.DER,
#         format=serialization.PrivateFormat.PKCS8,
#         encryption_algorithm=serialization.NoEncryption()
#     )

# # Create a SQLAlchemy engine using key pair authentication
# engine = create_engine(
#     f'snowflake://{os.getenv("SNOWFLAKE_USER")}'
#     f'@{os.getenv("SNOWFLAKE_ACCOUNT")}/{os.getenv("SNOWFLAKE_DATABASE")}',
#     connect_args={
#         'private_key': pkb,
#         'warehouse': os.getenv("SNOWFLAKE_WAREHOUSE"),
#         'role': os.getenv("SNOWFLAKE_ROLE"),
#         'schema': os.getenv("SNOWFLAKE_SCHEMA"),
#     }
# )

# # Define your SQL query
# query = """
# SELECT *
# FROM SANDBOX_DB.AHMADSAJEDI.SHOPPER_FRAUD_CLASSIFICATION_AGG_M1
# ORDER BY delivery_created_date_time_utc ASC
# """

# # Execute the query and fetch the data into a pandas DataFrame
# data = pd.read_sql(query, engine)

# end_time = time.time()
# print(f"Data Loading Time: {end_time - start_time:.2f} seconds")

# # ================== Data Preparation ==================

# start_time = time.time()

# # Remove 'delivery_created_date_time_utc' from the dataset
# if 'delivery_created_date_time_utc' in data.columns:
#     data.drop('delivery_created_date_time_utc', axis=1, inplace=True)

# # Identify the target column and ID columns
# target_column = 'fraud_vector'
# id_columns = ['order_delivery_id', 'batch_id', 'shopper_id']

# # Encode the target variable
# target_encoder = LabelEncoder()
# data[target_column] = target_encoder.fit_transform(data[target_column])

# # Convert 'region_id' to 'category' dtype if it's in the data
# if 'region_id' in data.columns:
#     data['region_id'] = data['region_id'].astype('category')

# # Identify categorical and numerical columns
# categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
# numeric_columns = data.select_dtypes(include=['number']).columns.tolist()

# # Remove target and ID columns from feature lists
# for col in [target_column] + id_columns:
#     if col in categorical_columns:
#         categorical_columns.remove(col)
#     if col in numeric_columns:
#         numeric_columns.remove(col)

# # Ensure all categorical columns are of type 'category'
# for col in categorical_columns:
#     data[col] = data[col].astype('category')

# # Split features and target
# X = data.drop([target_column] + id_columns, axis=1)
# y = data[target_column]

# end_time = time.time()
# print(f"Data Preparation Time: {end_time - start_time:.2f} seconds")

# # ================== Data Splitting ==================

# start_time = time.time()

# # Calculate indices for splitting the data
# total_length = len(data)
# train_end = int(0.7 * total_length)       # First 70% for training
# valid_end = int(0.85 * total_length)      # Next 15% for validation (70% + 15% = 85%)

# # Time-based splits
# X_train = X.iloc[:train_end].reset_index(drop=True)
# y_train = y.iloc[:train_end].reset_index(drop=True)

# X_valid = X.iloc[train_end:valid_end].reset_index(drop=True)
# y_valid = y.iloc[train_end:valid_end].reset_index(drop=True)

# X_test = X.iloc[valid_end:].reset_index(drop=True)
# y_test = y.iloc[valid_end:].reset_index(drop=True)

# end_time = time.time()
# print(f"Data Splitting Time: {end_time - start_time:.2f} seconds")

# # ================== Model Training ==================

# best_params = {
#     'objective': 'multiclass',
#     'num_class': len(np.unique(y)),
#     'metric': 'multi_logloss',
#     'boosting_type': 'gbdt',
#     'device': 'cpu',
#     'is_unbalance': True,
#     'verbose': -1,
#     'learning_rate': 0.014673672445495577,
#     'num_leaves': 128,
#     'max_depth': 12,
#     'min_data_in_leaf': 100,
#     'feature_fraction': 0.6002933605844709,
#     'bagging_fraction': 0.9019153707902258,
#     'bagging_freq': 4,
#     'lambda_l1': 5.084063300383103e-06,
#     'lambda_l2': 0.015662588748170638,
#     'min_gain_to_split': 0.9095436389736504
# }

# # Prepare LightGBM datasets
# lgb_train = lgb.Dataset(
#     X_train,
#     label=y_train,
#     categorical_feature=categorical_columns,
#     free_raw_data=False
# )
# lgb_valid = lgb.Dataset(
#     X_valid,
#     label=y_valid,
#     categorical_feature=categorical_columns,
#     free_raw_data=False
# )

# print("\nTraining the model on the training data...")

# final_model = lgb.train(
#     best_params,
#     lgb_train,
#     num_boost_round=1000,
#     valid_sets=[lgb_valid],
#     valid_names=['valid'],
#     callbacks=[
#         early_stopping(stopping_rounds=50),
#         log_evaluation(period=100)
#     ]
# )

# print("\nModel training completed.")

# # ================== Model Evaluation Function ==================

# def evaluate_model(y_true, y_pred_proba, y_pred_labels, dataset_name):
#     # Classification report
#     report = classification_report(
#         y_true,
#         y_pred_labels,
#         target_names=target_encoder.classes_
#     )
#     print(f"\nClassification Report on {dataset_name} Set:\n", report)
    
#     # Extract precision, recall, and F1-score for each class
#     precision, recall, f1_score, support = precision_recall_fscore_support(
#         y_true,
#         y_pred_labels,
#         labels=np.unique(y_true)
#     )
    
#     # Create DataFrame for visualization
#     metrics_df = pd.DataFrame({
#         'Class': target_encoder.inverse_transform(np.unique(y_true)),
#         'Precision': precision,
#         'Recall': recall,
#         'F1-Score': f1_score,
#         'Support': support
#     })
    
#     print(f"\nPrecision, Recall, and F1-Score per Class on {dataset_name} Set:")
#     print(metrics_df)
    
#     # Plot precision and recall
#     plt.figure(figsize=(10, 6))
#     x = np.arange(len(metrics_df['Class']))
#     width = 0.35  # the width of the bars
    
#     plt.bar(x - width/2, metrics_df['Precision'], width, label='Precision')
#     plt.bar(x + width/2, metrics_df['Recall'], width, label='Recall')
    
#     plt.xticks(x, metrics_df['Class'], rotation=45)
#     plt.ylabel('Score')
#     plt.title(f'Precision and Recall per Class on {dataset_name} Set')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f'precision_recall_per_class_{dataset_name.lower()}.png')
#     plt.close()
    
#     # Multi-class AUC score
#     try:
#         auc_score = roc_auc_score(
#             y_true,
#             y_pred_proba,
#             multi_class='ovr',
#             average='weighted'
#         )
#         print(f"\nWeighted AUC Score on {dataset_name} Set: {auc_score:.4f}")
#     except ValueError as e:
#         print(f"\nAUC Score could not be computed on {dataset_name} Set: {e}")
    
#     # Confusion Matrix
#     conf_matrix = confusion_matrix(y_true, y_pred_labels)
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(
#         conf_matrix,
#         annot=True,
#         fmt='d',
#         cmap='Blues',
#         xticklabels=target_encoder.classes_,
#         yticklabels=target_encoder.classes_
#     )
#     plt.title(f"Confusion Matrix on {dataset_name} Set")
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.tight_layout()
#     plt.savefig(f'confusion_matrix_{dataset_name.lower()}.png')
#     plt.close()

# # ================== Model Evaluation on Training Set ==================

# print("\nEvaluating the model on the training set...")

# # Predictions on training set
# y_pred_train = final_model.predict(X_train, num_iteration=final_model.best_iteration)
# y_pred_labels_train = np.argmax(y_pred_train, axis=1)

# evaluate_model(y_train, y_pred_train, y_pred_labels_train, "Training")

# # ================== Model Evaluation on Validation Set ==================

# print("\nEvaluating the model on the validation set...")

# # Predictions on validation set
# y_pred_valid = final_model.predict(X_valid, num_iteration=final_model.best_iteration)
# y_pred_labels_valid = np.argmax(y_pred_valid, axis=1)

# evaluate_model(y_valid, y_pred_valid, y_pred_labels_valid, "Validation")

# # ================== Model Evaluation on Test Set ==================

# print("\nEvaluating the model on the test set...")

# # Predictions on test set
# y_pred_test = final_model.predict(X_test, num_iteration=final_model.best_iteration)
# y_pred_labels_test = np.argmax(y_pred_test, axis=1)

# evaluate_model(y_test, y_pred_test, y_pred_labels_test, "Test")

# # ================== Save the Final Model ==================

# # Save the final model
# final_model.save_model('lightgbm_fraud_classifier.txt')

# print("\nModel evaluation on all datasets completed.")


# import os
# import time
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import lightgbm as lgb
# import optuna
# import snowflake.connector
# from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_fscore_support
# from sklearn.preprocessing import LabelEncoder
# from cryptography.hazmat.primitives import serialization
# from cryptography.hazmat.backends import default_backend
# from dotenv import load_dotenv
# from sqlalchemy import create_engine
# from lightgbm import early_stopping, log_evaluation
# import warnings

# warnings.filterwarnings('ignore')

# # ================== Data Loading ==================

# start_time = time.time()

# # Load environment variables from .env file
# load_dotenv()

# # Verify that essential environment variables are loaded
# required_vars = [
#     'SNOWFLAKE_USER',
#     'SNOWFLAKE_ACCOUNT',
#     'SNOWFLAKE_DATABASE',
#     'SNOWFLAKE_WAREHOUSE',
#     'SNOWFLAKE_ROLE',
#     'SNOWFLAKE_SCHEMA',
#     'SNOWFLAKE_PRIVATE_KEY_PATH'
# ]
# for var in required_vars:
#     if not os.getenv(var):
#         raise EnvironmentError(f"Required environment variable {var} is not set.")

# # Load the private key for Snowflake connection
# private_key_path = os.path.expanduser(os.getenv('SNOWFLAKE_PRIVATE_KEY_PATH'))
# with open(private_key_path, "rb") as key_file:
#     private_key = serialization.load_pem_private_key(
#         key_file.read(),
#         password=os.getenv('SNOWFLAKE_PRIVATE_KEY_PASSPHRASE').encode()
#         if os.getenv('SNOWFLAKE_PRIVATE_KEY_PASSPHRASE') else None,
#         backend=default_backend()
#     )
#     # Convert the private key to the format required by Snowflake
#     pkb = private_key.private_bytes(
#         encoding=serialization.Encoding.DER,
#         format=serialization.PrivateFormat.PKCS8,
#         encryption_algorithm=serialization.NoEncryption()
#     )

# # Create a SQLAlchemy engine using key pair authentication
# engine = create_engine(
#     f'snowflake://{os.getenv("SNOWFLAKE_USER")}'
#     f'@{os.getenv("SNOWFLAKE_ACCOUNT")}/{os.getenv("SNOWFLAKE_DATABASE")}',
#     connect_args={
#         'private_key': pkb,
#         'warehouse': os.getenv("SNOWFLAKE_WAREHOUSE"),
#         'role': os.getenv("SNOWFLAKE_ROLE"),
#         'schema': os.getenv("SNOWFLAKE_SCHEMA"),
#     }
# )

# # Define your SQL query
# query = """
# SELECT *
# FROM SANDBOX_DB.AHMADSAJEDI.SHOPPER_FRAUD_CLASSIFICATION_AGG_M1
# ORDER BY delivery_created_date_time_utc ASC
# """

# # Execute the query and fetch the data into a pandas DataFrame
# data = pd.read_sql(query, engine)

# end_time = time.time()
# print(f"Data Loading Time: {end_time - start_time:.2f} seconds")

# # ================== Data Preparation ==================

# start_time = time.time()

# # Remove 'delivery_created_date_time_utc' from the dataset
# if 'delivery_created_date_time_utc' in data.columns:
#     data.drop('delivery_created_date_time_utc', axis=1, inplace=True)

# # Identify the target column and ID columns
# target_column = 'fraud_vector'
# id_columns = ['order_delivery_id', 'batch_id', 'shopper_id']

# # Encode the target variable
# target_encoder = LabelEncoder()
# data[target_column] = target_encoder.fit_transform(data[target_column])

# # Convert 'region_id' to 'category' dtype if it's in the data
# if 'region_id' in data.columns:
#     data['region_id'] = data['region_id'].astype('category')

# # Identify categorical and numerical columns
# categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
# numeric_columns = data.select_dtypes(include=['number']).columns.tolist()

# # Remove target and ID columns from feature lists
# for col in [target_column] + id_columns:
#     if col in categorical_columns:
#         categorical_columns.remove(col)
#     if col in numeric_columns:
#         numeric_columns.remove(col)

# # Ensure all categorical columns are of type 'category'
# for col in categorical_columns:
#     data[col] = data[col].astype('category')

# # Split features and target
# X = data.drop([target_column] + id_columns, axis=1)
# y = data[target_column]

# end_time = time.time()
# print(f"Data Preparation Time: {end_time - start_time:.2f} seconds")

# # ================== Data Splitting ==================

# start_time = time.time()

# # Calculate indices for splitting the data
# total_length = len(data)
# train_end = int(0.7 * total_length)       # First 70% for training
# valid_end = int(0.85 * total_length)      # Next 15% for validation (70% + 15% = 85%)

# # Time-based splits
# X_train = X.iloc[:train_end].reset_index(drop=True)
# y_train = y.iloc[:train_end].reset_index(drop=True)

# X_valid = X.iloc[train_end:valid_end].reset_index(drop=True)
# y_valid = y.iloc[train_end:valid_end].reset_index(drop=True)

# X_test = X.iloc[valid_end:].reset_index(drop=True)
# y_test = y.iloc[valid_end:].reset_index(drop=True)

# end_time = time.time()
# print(f"Data Splitting Time: {end_time - start_time:.2f} seconds")

# # ================== Hyperparameter Tuning with Optuna ==================

# def objective(trial):
#     # Hyperparameter search space
#     param = {
#         'objective': 'multiclass',
#         'num_class': len(np.unique(y)),
#         'metric': 'multi_logloss',
#         'boosting_type': 'gbdt',
#         'device': 'cpu',
#         'is_unbalance': True,
#         'verbose': -1,
#         'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.05),
#         'num_leaves': trial.suggest_int('num_leaves', 31, 128),
#         'max_depth': trial.suggest_int('max_depth', 5, 15),
#         'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),
#         'feature_fraction': trial.suggest_uniform('feature_fraction', 0.6, 1.0),
#         'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.6, 1.0),
#         'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
#         'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
#         'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
#         'min_gain_to_split': trial.suggest_uniform('min_gain_to_split', 0.0, 1.0)
#     }

#     # Prepare LightGBM datasets
#     lgb_train = lgb.Dataset(
#         X_train,
#         label=y_train,
#         categorical_feature=categorical_columns,
#         free_raw_data=False
#     )
#     lgb_valid = lgb.Dataset(
#         X_valid,
#         label=y_valid,
#         categorical_feature=categorical_columns,
#         free_raw_data=False
#     )

#     # Train the model with callbacks and specify the validation dataset name
#     gbm = lgb.train(
#         param,
#         lgb_train,
#         num_boost_round=1000,
#         valid_sets=[lgb_valid],
#         valid_names=['valid'],  # Specify the validation set name
#         callbacks=[
#             early_stopping(stopping_rounds=50),
#             log_evaluation(period=100)
#         ]
#     )

#     # Predict on validation set
#     y_pred_valid = gbm.predict(X_valid, num_iteration=gbm.best_iteration)
#     y_pred_labels_valid = np.argmax(y_pred_valid, axis=1)

#     # Compute validation metrics
#     precision, recall, f1_score, _ = precision_recall_fscore_support(
#         y_valid,
#         y_pred_labels_valid,
#         average='weighted'
#     )

#     # Store metrics in trial for later analysis
#     trial.set_user_attr('precision', precision)
#     trial.set_user_attr('recall', recall)
#     trial.set_user_attr('f1_score', f1_score)

#     # For per-class metrics
#     per_class_metrics = classification_report(
#         y_valid,
#         y_pred_labels_valid,
#         target_names=target_encoder.classes_,
#         output_dict=True
#     )
#     trial.set_user_attr('per_class_metrics', per_class_metrics)

#     # Return validation multi-logloss
#     return gbm.best_score['valid']['multi_logloss']

# print("Starting hyperparameter tuning...")

# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=100)

# print("Hyperparameter tuning completed.")

# # Retrieve best trial
# best_trial = study.best_trial
# print(f"Best trial number: {best_trial.number}")
# print("Best hyperparameters:")
# for key, value in best_trial.params.items():
#     print(f"  {key}: {value}")

# print("\nValidation metrics for the best trial:")
# print(f"Precision: {best_trial.user_attrs['precision']:.4f}")
# print(f"Recall: {best_trial.user_attrs['recall']:.4f}")
# print(f"F1-Score: {best_trial.user_attrs['f1_score']:.4f}")

# # Convert per-class metrics to DataFrame
# per_class_metrics = best_trial.user_attrs['per_class_metrics']
# per_class_df = pd.DataFrame.from_dict(per_class_metrics).T
# per_class_df = per_class_df.iloc[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
# print("\nPer-class metrics on validation set for the best trial:")
# print(per_class_df[['precision', 'recall', 'f1-score', 'support']])

# # ================== Final Model Training ==================

# # Combine training and validation data
# X_train_full = pd.concat([X_train, X_valid]).reset_index(drop=True)
# y_train_full = pd.concat([y_train, y_valid]).reset_index(drop=True)

# # Prepare dataset
# lgb_train_full = lgb.Dataset(
#     X_train_full,
#     label=y_train_full,
#     categorical_feature=categorical_columns,
#     free_raw_data=False
# )

# print("\nTraining final model on the combined training and validation data...")

# final_model = lgb.train(
#     {**best_trial.params, 'objective': 'multiclass', 'num_class': len(np.unique(y)), 'metric': 'multi_logloss'},
#     lgb_train_full,
#     num_boost_round=1000,
#     callbacks=[
#         early_stopping(stopping_rounds=50),
#         log_evaluation(period=100)
#     ]
# )

# print("\nFinal model training completed.")

# # ================== Model Evaluation on Test Set ==================

# print("\nEvaluating the final model on the test set...")

# # Predict on test set
# y_pred_test = final_model.predict(X_test, num_iteration=final_model.best_iteration)
# y_pred_labels_test = np.argmax(y_pred_test, axis=1)

# # Classification report with precision and recall
# report = classification_report(
#     y_test,
#     y_pred_labels_test,
#     target_names=target_encoder.classes_
# )
# print("\nClassification Report on Test Set:\n", report)

# # Extract precision, recall, and f1-score for each class
# precision, recall, f1_score, support = precision_recall_fscore_support(
#     y_test,
#     y_pred_labels_test,
#     labels=np.unique(y_test)
# )

# # Create a DataFrame for better visualization
# metrics_df = pd.DataFrame({
#     'Class': target_encoder.inverse_transform(np.unique(y_test)),
#     'Precision': precision,
#     'Recall': recall,
#     'F1-Score': f1_score,
#     'Support': support
# })

# print("\nPrecision, Recall, and F1-Score per Class on Test Set:")
# print(metrics_df)

# # Plot precision and recall
# plt.figure(figsize=(10, 6))
# x = np.arange(len(metrics_df['Class']))
# width = 0.35  # the width of the bars

# plt.bar(x - width/2, metrics_df['Precision'], width, label='Precision')
# plt.bar(x + width/2, metrics_df['Recall'], width, label='Recall')

# plt.xticks(x, metrics_df['Class'])
# plt.ylabel('Score')
# plt.title('Precision and Recall per Class on Test Set')
# plt.legend()
# plt.tight_layout()
# plt.savefig('precision_recall_per_class_test.png')
# plt.close()

# # Multi-class AUC score
# auc_score = roc_auc_score(
#     y_test,
#     y_pred_test,
#     multi_class='ovr',
#     average='weighted'
# )
# print("\nWeighted AUC Score on Test Set:", auc_score)

# # Confusion Matrix
# conf_matrix = confusion_matrix(y_test, y_pred_labels_test)
# plt.figure(figsize=(10, 8))
# sns.heatmap(
#     conf_matrix,
#     annot=True,
#     fmt='d',
#     cmap='Blues',
#     xticklabels=target_encoder.classes_,
#     yticklabels=target_encoder.classes_
# )
# plt.title("Confusion Matrix on Test Set")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.tight_layout()
# plt.savefig('confusion_matrix_test.png')
# plt.close()

# # Save the final model
# final_model.save_model('lightgbm_fraud_classifier.txt')

# print("\nModel evaluation on test set completed.")

# # import os
# # import time
# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # import lightgbm as lgb
# # import snowflake.connector
# # from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_fscore_support
# # from sklearn.preprocessing import LabelEncoder
# # from cryptography.hazmat.primitives import serialization
# # from cryptography.hazmat.backends import default_backend
# # from dotenv import load_dotenv
# # from sqlalchemy import create_engine
# # from lightgbm import early_stopping, log_evaluation
# # import warnings

# # warnings.filterwarnings('ignore')

# # # ================== Data Loading ==================

# # start_time = time.time()

# # # Load environment variables from .env file
# # load_dotenv()

# # # Verify that essential environment variables are loaded
# # required_vars = [
# #     'SNOWFLAKE_USER',
# #     'SNOWFLAKE_ACCOUNT',
# #     'SNOWFLAKE_DATABASE',
# #     'SNOWFLAKE_WAREHOUSE',
# #     'SNOWFLAKE_ROLE',
# #     'SNOWFLAKE_SCHEMA',
# #     'SNOWFLAKE_PRIVATE_KEY_PATH'
# # ]
# # for var in required_vars:
# #     if not os.getenv(var):
# #         raise EnvironmentError(f"Required environment variable {var} is not set.")

# # # Load the private key for Snowflake connection
# # private_key_path = os.path.expanduser(os.getenv('SNOWFLAKE_PRIVATE_KEY_PATH'))
# # with open(private_key_path, "rb") as key_file:
# #     private_key = serialization.load_pem_private_key(
# #         key_file.read(),
# #         password=os.getenv('SNOWFLAKE_PRIVATE_KEY_PASSPHRASE').encode()
# #         if os.getenv('SNOWFLAKE_PRIVATE_KEY_PASSPHRASE') else None,
# #         backend=default_backend()
# #     )
# #     # Convert the private key to the format required by Snowflake
# #     pkb = private_key.private_bytes(
# #         encoding=serialization.Encoding.DER,
# #         format=serialization.PrivateFormat.PKCS8,
# #         encryption_algorithm=serialization.NoEncryption()
# #     )

# # # Create a SQLAlchemy engine using key pair authentication
# # engine = create_engine(
# #     f'snowflake://{os.getenv("SNOWFLAKE_USER")}'
# #     f'@{os.getenv("SNOWFLAKE_ACCOUNT")}/{os.getenv("SNOWFLAKE_DATABASE")}',
# #     connect_args={
# #         'private_key': pkb,
# #         'warehouse': os.getenv("SNOWFLAKE_WAREHOUSE"),
# #         'role': os.getenv("SNOWFLAKE_ROLE"),
# #         'schema': os.getenv("SNOWFLAKE_SCHEMA"),
# #     }
# # )

# # # Define your SQL query
# # query = """
# # SELECT *
# # FROM SANDBOX_DB.AHMADSAJEDI.SHOPPER_FRAUD_CLASSIFICATION_AGG_M1
# # ORDER BY delivery_created_date_time_utc ASC
# # """

# # # Execute the query and fetch the data into a pandas DataFrame
# # data = pd.read_sql(query, engine)

# # end_time = time.time()
# # print(f"Data Loading Time: {end_time - start_time:.2f} seconds")

# # # ================== Data Preparation ==================

# # start_time = time.time()

# # # Remove 'DELIVERY_CREATED_DATE_TIME_UTC' from the dataset
# # if 'delivery_created_date_time_utc' in data.columns:
# #     data.drop('delivery_created_date_time_utc', axis=1, inplace=True)

# # # Identify the target column and ID columns
# # target_column = 'fraud_vector'
# # id_columns = ['order_delivery_id', 'batch_id', 'shopper_id']

# # # Encode the target variable
# # target_encoder = LabelEncoder()
# # data[target_column] = target_encoder.fit_transform(data[target_column])

# # # Convert 'region_id' to 'category' dtype if it's in the data
# # if 'region_id' in data.columns:
# #     data['region_id'] = data['region_id'].astype('category')

# # # Identify categorical and numerical columns
# # categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
# # numeric_columns = data.select_dtypes(include=['number']).columns.tolist()

# # # Remove target and ID columns from feature lists
# # for col in [target_column] + id_columns:
# #     if col in categorical_columns:
# #         categorical_columns.remove(col)
# #     if col in numeric_columns:
# #         numeric_columns.remove(col)

# # # Ensure all categorical columns are of type 'category'
# # for col in categorical_columns:
# #     data[col] = data[col].astype('category')

# # # Split features and target
# # X = data.drop([target_column] + id_columns, axis=1)
# # y = data[target_column]

# # end_time = time.time()
# # print(f"Data Preparation Time: {end_time - start_time:.2f} seconds")

# # # ================== Data Splitting ==================

# # start_time = time.time()

# # # Calculate indices for splitting the data
# # total_length = len(data)
# # train_end = int(0.7 * total_length)       # First 70% for training
# # valid_end = int(0.85 * total_length)      # Next 15% for validation (70% + 15% = 85%)

# # # Time-based splits
# # X_train = X.iloc[:train_end].reset_index(drop=True)
# # y_train = y.iloc[:train_end].reset_index(drop=True)

# # X_valid = X.iloc[train_end:valid_end].reset_index(drop=True)
# # y_valid = y.iloc[train_end:valid_end].reset_index(drop=True)

# # X_test = X.iloc[valid_end:].reset_index(drop=True)
# # y_test = y.iloc[valid_end:].reset_index(drop=True)

# # # Prepare LightGBM datasets
# # lgb_train = lgb.Dataset(
# #     X_train,
# #     label=y_train,
# #     categorical_feature=categorical_columns,
# #     free_raw_data=False
# # )
# # lgb_valid = lgb.Dataset(
# #     X_valid,
# #     label=y_valid,
# #     categorical_feature=categorical_columns,
# #     free_raw_data=False
# # )

# # end_time = time.time()
# # print(f"Data Splitting Time: {end_time - start_time:.2f} seconds")

# # # ================== Model Training ==================

# # start_time = time.time()

# # # Define the best parameters
# # best_params = {
# #     'objective': 'multiclass',
# #     'num_class': len(np.unique(y)),
# #     'metric': 'multi_logloss',
# #     'boosting_type': 'gbdt',
# #     'device': 'cpu',  
# #     'is_unbalance': True,
# #     'verbose': -1,
# #     'learning_rate': 0.01004545415644607,
# #     'num_leaves': 114,
# #     'max_depth': 9,
# #     'min_data_in_leaf': 20,
# #     'feature_fraction': 0.7159067625423929,
# #     'bagging_fraction': 0.7010161036473805,
# #     'bagging_freq': 3,
# #     'lambda_l1': 3.700115625625493e-08,
# #     'lambda_l2': 2.154382646585372e-07,
# #     'min_gain_to_split': 0.5982134430353536
# # }

# # # Train the final model with early stopping
# # final_model = lgb.train(
# #     best_params,
# #     lgb_train,
# #     num_boost_round=1000,
# #     valid_sets=[lgb_valid],
# #     valid_names=['valid'],
# #     callbacks=[
# #         early_stopping(stopping_rounds=50),
# #         log_evaluation(period=100)
# #     ]
# # )

# # end_time = time.time()
# # print(f"Model Training Time: {end_time - start_time:.2f} seconds")

# # # ================== Model Evaluation ==================

# # start_time = time.time()

# # # Predict on test set
# # y_pred = final_model.predict(X_test, num_iteration=final_model.best_iteration)
# # y_pred_labels = np.argmax(y_pred, axis=1)

# # # Classification report with precision and recall
# # report = classification_report(
# #     y_test,
# #     y_pred_labels,
# #     target_names=target_encoder.classes_
# # )
# # print("Classification Report:\n", report)

# # # Extract precision, recall, and F1-score for each class
# # precision, recall, f1_score, support = precision_recall_fscore_support(
# #     y_test,
# #     y_pred_labels,
# #     labels=np.unique(y_test)
# # )

# # # Create a DataFrame for better visualization
# # metrics_df = pd.DataFrame({
# #     'Class': target_encoder.inverse_transform(np.unique(y_test)),
# #     'Precision': precision,
# #     'Recall': recall,
# #     'F1-Score': f1_score,
# #     'Support': support
# # })

# # print("\nPrecision, Recall, and F1-Score per Class:")
# # print(metrics_df)

# # # Optionally, plot precision and recall
# # plt.figure(figsize=(10, 6))
# # x = np.arange(len(metrics_df['Class']))
# # width = 0.35  # the width of the bars

# # plt.bar(x - width/2, metrics_df['Precision'], width, label='Precision')
# # plt.bar(x + width/2, metrics_df['Recall'], width, label='Recall')

# # plt.xticks(x, metrics_df['Class'])
# # plt.ylabel('Score')
# # plt.title('Precision and Recall per Class')
# # plt.legend()
# # plt.tight_layout()
# # plt.savefig('precision_recall_per_class.png')
# # plt.close()

# # # Multi-class AUC score
# # auc_score = roc_auc_score(
# #     y_test,
# #     y_pred,
# #     multi_class='ovr',
# #     average='weighted'
# # )
# # print("Weighted AUC Score:", auc_score)

# # # Confusion Matrix
# # conf_matrix = confusion_matrix(y_test, y_pred_labels)
# # plt.figure(figsize=(10, 8))
# # sns.heatmap(
# #     conf_matrix,
# #     annot=True,
# #     fmt='d',
# #     cmap='Blues',
# #     xticklabels=target_encoder.classes_,
# #     yticklabels=target_encoder.classes_
# # )
# # plt.title("Confusion Matrix")
# # plt.xlabel("Predicted")
# # plt.ylabel("Actual")
# # plt.tight_layout()
# # plt.savefig('confusion_matrix.png')
# # plt.close()

# # # Save the final model
# # final_model.save_model('lightgbm_fraud_classifier.txt')

# # end_time = time.time()
# # print(f"Model Evaluation Time: {end_time - start_time:.2f} seconds")
