"""
Credit Rating Classification using Multinomial Logistic Regression
===================================================================
This script implements a complete ML pipeline for classifying corporate credit ratings
into 6 grouped categories using financial metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             roc_curve, auc, precision_recall_curve, average_precision_score)
from sklearn.preprocessing import label_binarize
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Configure plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_and_explore_data(filepath):
    """Load dataset and perform initial exploration."""
    print("="*80)
    print("STEP 1: LOADING AND EXPLORING DATA")
    print("="*80)

    df = pd.read_csv(filepath)
    print(f"\nDataset shape: {df.shape}")
    print(f"Number of features: {df.shape[1]}")
    print(f"Number of samples: {df.shape[0]}")

    print("\nColumn names and types:")
    print(df.dtypes)

    print("\nMissing values:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values found!")

    print("\nTarget variable (Rating) distribution:")
    print(df['Rating'].value_counts().sort_index())

    return df


def select_features(df):
    """Select relevant features: Sector + 16 financial metrics."""
    print("\n" + "="*80)
    print("STEP 2: FEATURE SELECTION")
    print("="*80)

    # Features to keep
    feature_columns = [
        'Sector',
        'Current Ratio',
        'Long-term Debt / Capital',
        'Debt/Equity Ratio',
        'Gross Margin',
        'Operating Margin',
        'EBIT Margin',
        'EBITDA Margin',
        'Pre-Tax Profit Margin',
        'Net Profit Margin',
        'Asset Turnover',
        'ROE - Return On Equity',
        'Return On Tangible Equity',
        'ROA - Return On Assets',
        'ROI - Return On Investment',
        'Operating Cash Flow Per Share',
        'Free Cash Flow Per Share'
    ]

    # Keep only selected features + target
    df_selected = df[feature_columns + ['Rating']].copy()

    print(f"\nSelected features: {len(feature_columns)}")
    print(f"- Categorical: 1 (Sector)")
    print(f"- Numerical: 16 (financial metrics)")
    print(f"\nDropped columns: Rating Agency, Corporation")

    return df_selected


def handle_missing_values(df):
    """Handle missing values in the dataset."""
    print("\n" + "="*80)
    print("STEP 3: HANDLING MISSING VALUES")
    print("="*80)

    missing_before = df.isnull().sum().sum()
    print(f"\nTotal missing values: {missing_before}")

    if missing_before > 0:
        print("\nMissing values by column:")
        missing_cols = df.isnull().sum()
        print(missing_cols[missing_cols > 0])

        # Impute numerical columns with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"Imputed {col} with median: {median_val:.4f}")

        # Impute categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'Rating' and df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                print(f"Imputed {col} with mode: {mode_val}")
    else:
        print("No missing values to handle!")

    print(f"\nMissing values after handling: {df.isnull().sum().sum()}")
    return df


def group_ratings(df):
    """Group 23 rating classes into 6 meaningful categories."""
    print("\n" + "="*80)
    print("STEP 4: GROUPING RATING CLASSES")
    print("="*80)

    # Define rating groups based on credit quality
    rating_groups = {
        'Excellent': ['AAA', 'AA+', 'AA', 'AA-'],
        'Good': ['A+', 'A', 'A-'],
        'Moderate': ['BBB+', 'BBB', 'BBB-'],
        'Speculative': ['BB+', 'BB', 'BB-'],
        'Highly Speculative': ['B+', 'B', 'B-'],
        'High Risk': ['CCC+', 'CCC', 'CCC-', 'CC+', 'CC', 'C', 'D']
    }

    # Create mapping dictionary
    rating_to_group = {}
    for group, ratings in rating_groups.items():
        for rating in ratings:
            rating_to_group[rating] = group

    # Apply grouping
    df['Rating_Grouped'] = df['Rating'].map(rating_to_group)

    print("\nRating grouping scheme:")
    for group, ratings in rating_groups.items():
        print(f"  {group}: {', '.join(ratings)}")

    print("\nOriginal distribution (23 classes):")
    print(df['Rating'].value_counts().sort_index())

    print("\nGrouped distribution (6 classes):")
    grouped_dist = df['Rating_Grouped'].value_counts()
    print(grouped_dist)

    # Check for any unmapped ratings
    unmapped = df[df['Rating_Grouped'].isnull()]
    if len(unmapped) > 0:
        print(f"\nWarning: {len(unmapped)} ratings could not be mapped!")
        print(unmapped['Rating'].unique())

    return df, rating_groups


def encode_and_scale_features(df):
    """Encode categorical features and scale numerical features."""
    print("\n" + "="*80)
    print("STEP 5: ENCODING AND SCALING FEATURES")
    print("="*80)

    # Separate features and target
    X = df.drop(['Rating', 'Rating_Grouped'], axis=1)
    y = df['Rating_Grouped']

    print(f"\nFeature matrix shape before encoding: {X.shape}")

    # One-hot encode Sector
    X_encoded = pd.get_dummies(X, columns=['Sector'], prefix='Sector', drop_first=False)

    print(f"Feature matrix shape after encoding: {X_encoded.shape}")
    print(f"Added {X_encoded.shape[1] - X.shape[1]} dummy variables for Sector")

    # Get feature names
    feature_names = X_encoded.columns.tolist()

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    print("\nScaling applied: StandardScaler")
    print("  - Mean scaled to 0")
    print("  - Standard deviation scaled to 1")
    print("  - Prevents bias toward features with larger values")

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print(f"\nTarget classes: {list(label_encoder.classes_)}")
    print(f"Encoded as: {list(range(len(label_encoder.classes_)))}")

    return X_scaled, y_encoded, scaler, label_encoder, feature_names


def split_data(X, y):
    """Split data into train and test sets with stratification."""
    print("\n" + "="*80)
    print("STEP 6: SPLITTING DATA")
    print("="*80)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\nTrain set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

    print("\nTrain set class distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls}: {cnt} samples ({cnt/len(y_train)*100:.1f}%)")

    print("\nTest set class distribution:")
    unique, counts = np.unique(y_test, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls}: {cnt} samples ({cnt/len(y_test)*100:.1f}%)")

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Train multinomial logistic regression model."""
    print("\n" + "="*80)
    print("STEP 7: TRAINING MULTINOMIAL LOGISTIC REGRESSION MODEL")
    print("="*80)

    # Calculate class weights to handle remaining imbalance
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))

    print("\nClass weights (to handle imbalance):")
    for cls, weight in class_weight_dict.items():
        print(f"  Class {cls}: {weight:.4f}")

    # Train model
    print("\nModel configuration:")
    print("  - Solver: lbfgs (suitable for multinomial)")
    print("  - Max iterations: 1000")
    print("  - Multi-class: multinomial")
    print("  - Class weights: balanced")

    model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight='balanced'
    )

    print("\nTraining model...")
    model.fit(X_train, y_train)

    print("Training completed!")
    print(f"Number of iterations: {model.n_iter_[0]}")

    return model


def cross_validate_model(model, X_train, y_train):
    """Perform cross-validation to assess model stability."""
    print("\n" + "="*80)
    print("STEP 8: CROSS-VALIDATION")
    print("="*80)

    print("\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"Min CV accuracy: {cv_scores.min():.4f}")
    print(f"Max CV accuracy: {cv_scores.max():.4f}")

    return cv_scores


def evaluate_model(model, X_train, X_test, y_train, y_test, label_encoder):
    """Generate predictions and calculate evaluation metrics."""
    print("\n" + "="*80)
    print("STEP 9: MODEL EVALUATION")
    print("="*80)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Prediction probabilities
    y_train_proba = model.predict_proba(X_train)
    y_test_proba = model.predict_proba(X_test)

    # Training accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"\nTraining Accuracy: {train_accuracy:.4f}")

    # Test accuracy
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Classification report
    print("\n" + "-"*80)
    print("CLASSIFICATION REPORT (Test Set)")
    print("-"*80)
    print(classification_report(y_test, y_test_pred,
                                target_names=label_encoder.classes_,
                                digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)

    return y_train_pred, y_test_pred, y_train_proba, y_test_proba, cm, train_accuracy, test_accuracy


def plot_confusion_matrix(cm, label_encoder, save_path='confusion_matrix.png'):
    """Create and save confusion matrix heatmap."""
    print("\n" + "="*80)
    print("STEP 10: CONFUSION MATRIX VISUALIZATION")
    print("="*80)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                cbar_kws={'label': 'Number of Samples'})
    plt.title('Confusion Matrix - Credit Rating Classification', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {save_path}")
    plt.close()


def plot_roc_curves(y_test, y_test_proba, label_encoder, save_path='roc_curves.png'):
    """Generate ROC curves for each class (one-vs-rest)."""
    print("\n" + "="*80)
    print("STEP 11: ROC CURVES")
    print("="*80)

    n_classes = len(label_encoder.classes_)

    # Binarize the output
    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_test_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{label_encoder.classes_[i]} (AUC = {roc_auc[i]:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - One-vs-Rest', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nROC curves saved to: {save_path}")

    print("\nAUC Scores by class:")
    for i in range(n_classes):
        print(f"  {label_encoder.classes_[i]}: {roc_auc[i]:.4f}")

    plt.close()
    return roc_auc


def plot_feature_importance(model, feature_names, label_encoder, save_path='feature_importance.png'):
    """Visualize feature importance using model coefficients."""
    print("\n" + "="*80)
    print("STEP 12: FEATURE IMPORTANCE")
    print("="*80)

    # Get coefficients for each class
    coefficients = model.coef_

    # Calculate average absolute coefficient across all classes
    avg_coef = np.abs(coefficients).mean(axis=0)

    # Get top 20 features
    top_indices = np.argsort(avg_coef)[-20:]
    top_features = [feature_names[i] for i in top_indices]
    top_values = avg_coef[top_indices]

    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_values, color='steelblue')
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Average Absolute Coefficient', fontsize=12)
    plt.title('Top 20 Most Important Features', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nFeature importance plot saved to: {save_path}")

    print("\nTop 10 most important features:")
    for i in range(min(10, len(top_features))):
        print(f"  {i+1}. {top_features[-(i+1)]}: {top_values[-(i+1)]:.4f}")

    plt.close()


def plot_class_distribution(df, rating_groups, save_path='class_distribution.png'):
    """Create class distribution plots before and after grouping."""
    print("\n" + "="*80)
    print("STEP 13: CLASS DISTRIBUTION VISUALIZATION")
    print("="*80)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Original distribution
    rating_order = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-',
                   'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-',
                   'B+', 'B', 'B-', 'CCC+', 'CCC', 'CCC-',
                   'CC+', 'CC', 'C', 'D']

    original_counts = df['Rating'].value_counts()
    original_counts = original_counts.reindex(rating_order, fill_value=0)

    axes[0].bar(range(len(original_counts)), original_counts.values, color='coral')
    axes[0].set_xticks(range(len(original_counts)))
    axes[0].set_xticklabels(original_counts.index, rotation=45, ha='right')
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Original Rating Distribution (23 Classes)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Grouped distribution
    group_order = ['Excellent', 'Good', 'Moderate', 'Speculative', 'Highly Speculative', 'High Risk']
    grouped_counts = df['Rating_Grouped'].value_counts()
    grouped_counts = grouped_counts.reindex(group_order)

    colors = ['darkgreen', 'green', 'yellow', 'orange', 'red', 'darkred']
    axes[1].bar(range(len(grouped_counts)), grouped_counts.values, color=colors)
    axes[1].set_xticks(range(len(grouped_counts)))
    axes[1].set_xticklabels(grouped_counts.index, rotation=45, ha='right')
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Grouped Rating Distribution (6 Classes)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nClass distribution plot saved to: {save_path}")
    plt.close()


def plot_precision_recall_curves(y_test, y_test_proba, label_encoder, save_path='precision_recall_curves.png'):
    """Generate Precision-Recall curves for each class."""
    print("\n" + "="*80)
    print("STEP 14: PRECISION-RECALL CURVES")
    print("="*80)

    n_classes = len(label_encoder.classes_)
    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    # Compute Precision-Recall curve and average precision for each class
    precision = dict()
    recall = dict()
    avg_precision = dict()

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_test_proba[:, i])
        avg_precision[i] = average_precision_score(y_test_bin[:, i], y_test_proba[:, i])

    # Plot
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))

    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                label=f'{label_encoder.classes_[i]} (AP = {avg_precision[i]:.3f})')

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves', fontsize=16, fontweight='bold')
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPrecision-Recall curves saved to: {save_path}")

    print("\nAverage Precision scores by class:")
    for i in range(n_classes):
        print(f"  {label_encoder.classes_[i]}: {avg_precision[i]:.4f}")

    plt.close()
    return avg_precision


def plot_learning_curves(model, X_train, y_train, save_path='learning_curves.png'):
    """Generate learning curves to assess model performance."""
    print("\n" + "="*80)
    print("STEP 15: LEARNING CURVES")
    print("="*80)

    print("\nComputing learning curves (this may take a moment)...")

    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    plt.plot(train_sizes, val_mean, 'o-', color='green', label='Cross-validation score')

    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                     alpha=0.1, color='blue')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                     alpha=0.1, color='green')

    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('Accuracy Score', fontsize=12)
    plt.title('Learning Curves', fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nLearning curves saved to: {save_path}")
    plt.close()


def save_model(model, scaler, label_encoder, feature_names, save_path='credit_rating_model.pkl'):
    """Save trained model and preprocessing objects."""
    print("\n" + "="*80)
    print("STEP 16: SAVING MODEL")
    print("="*80)

    model_package = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_names': feature_names
    }

    with open(save_path, 'wb') as f:
        pickle.dump(model_package, f)

    print(f"\nModel saved to: {save_path}")
    print("Package includes:")
    print("  - Trained LogisticRegression model")
    print("  - StandardScaler for feature scaling")
    print("  - LabelEncoder for target decoding")
    print("  - Feature names for reference")


def generate_report(train_accuracy, test_accuracy, cv_scores, roc_auc, avg_precision,
                   label_encoder, save_path='model_report.txt'):
    """Generate comprehensive model performance report."""
    print("\n" + "="*80)
    print("STEP 17: GENERATING COMPREHENSIVE REPORT")
    print("="*80)

    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CREDIT RATING CLASSIFICATION - MODEL PERFORMANCE REPORT\n")
        f.write("="*80 + "\n\n")

        f.write("MODEL CONFIGURATION\n")
        f.write("-"*80 + "\n")
        f.write("Algorithm: Multinomial Logistic Regression\n")
        f.write("Solver: lbfgs\n")
        f.write("Class Weights: balanced\n")
        f.write("Random State: 42\n\n")

        f.write("DATASET SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write("Total Samples: 7,805\n")
        f.write("Training Set: 5,464 samples (70%)\n")
        f.write("Test Set: 2,341 samples (30%)\n")
        f.write("Number of Features: 27 (after one-hot encoding)\n")
        f.write("Number of Classes: 6 (grouped from 23 original ratings)\n\n")

        f.write("CLASS GROUPS\n")
        f.write("-"*80 + "\n")
        f.write("1. Excellent: AAA, AA+, AA, AA-\n")
        f.write("2. Good: A+, A, A-\n")
        f.write("3. Moderate: BBB+, BBB, BBB-\n")
        f.write("4. Speculative: BB+, BB, BB-\n")
        f.write("5. Highly Speculative: B+, B, B-\n")
        f.write("6. High Risk: CCC+, CCC, CCC-, CC+, CC, C, D\n\n")

        f.write("ACCURACY METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Training Accuracy: {train_accuracy:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Cross-Validation Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n\n")

        f.write("AREA UNDER ROC CURVE (AUC) - One-vs-Rest\n")
        f.write("-"*80 + "\n")
        for i, cls in enumerate(label_encoder.classes_):
            f.write(f"{cls}: {roc_auc[i]:.4f}\n")
        f.write(f"\nAverage AUC: {np.mean(list(roc_auc.values())):.4f}\n\n")

        f.write("AVERAGE PRECISION SCORES\n")
        f.write("-"*80 + "\n")
        for i, cls in enumerate(label_encoder.classes_):
            f.write(f"{cls}: {avg_precision[i]:.4f}\n")
        f.write(f"\nMean Average Precision: {np.mean(list(avg_precision.values())):.4f}\n\n")

        f.write("MODEL INTERPRETATION\n")
        f.write("-"*80 + "\n")
        f.write("The model uses financial metrics to classify credit ratings into 6 groups.\n")
        f.write("Key points:\n")
        f.write("- Higher AUC indicates better discrimination ability for each class\n")
        f.write("- Learning curves show model convergence and generalization\n")
        f.write("- Feature importance reveals which financial metrics are most predictive\n")
        f.write("- Balanced class weights help handle remaining class imbalance\n\n")

        f.write("VISUALIZATIONS GENERATED\n")
        f.write("-"*80 + "\n")
        f.write("1. confusion_matrix.png - Classification confusion matrix\n")
        f.write("2. roc_curves.png - ROC curves for all classes\n")
        f.write("3. feature_importance.png - Top 20 influential features\n")
        f.write("4. class_distribution.png - Original vs grouped distributions\n")
        f.write("5. precision_recall_curves.png - PR curves for all classes\n")
        f.write("6. learning_curves.png - Training and validation performance\n\n")

        f.write("="*80 + "\n")
        f.write("End of Report\n")
        f.write("="*80 + "\n")

    print(f"\nComprehensive report saved to: {save_path}")


def main():
    """Main execution pipeline."""
    print("\n")
    print("="*80)
    print(" "*20 + "CREDIT RATING CLASSIFICATION")
    print(" "*15 + "Multinomial Logistic Regression Model")
    print("="*80)
    print("\n")

    # 1. Load and explore data
    df = load_and_explore_data('company_rating_data.csv')

    # 2. Select features
    df = select_features(df)

    # 3. Handle missing values
    df = handle_missing_values(df)

    # 4. Group ratings
    df, rating_groups = group_ratings(df)

    # 5. Encode and scale features
    X, y, scaler, label_encoder, feature_names = encode_and_scale_features(df)

    # 6. Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 7. Train model
    model = train_model(X_train, y_train)

    # 8. Cross-validation
    cv_scores = cross_validate_model(model, X_train, y_train)

    # 9. Evaluate model
    y_train_pred, y_test_pred, y_train_proba, y_test_proba, cm, train_acc, test_acc = \
        evaluate_model(model, X_train, X_test, y_train, y_test, label_encoder)

    # 10. Confusion matrix
    plot_confusion_matrix(cm, label_encoder)

    # 11. ROC curves
    roc_auc = plot_roc_curves(y_test, y_test_proba, label_encoder)

    # 12. Feature importance
    plot_feature_importance(model, feature_names, label_encoder)

    # 13. Class distribution
    plot_class_distribution(df, rating_groups)

    # 14. Precision-Recall curves
    avg_precision = plot_precision_recall_curves(y_test, y_test_proba, label_encoder)

    # 15. Learning curves
    plot_learning_curves(model, X_train, y_train)

    # 16. Save model
    save_model(model, scaler, label_encoder, feature_names)

    # 17. Generate report
    generate_report(train_acc, test_acc, cv_scores, roc_auc, avg_precision, label_encoder)

    print("\n")
    print("="*80)
    print(" "*25 + "PIPELINE COMPLETED!")
    print("="*80)
    print("\nAll outputs saved successfully:")
    print("  - credit_rating_model.pkl (trained model)")
    print("  - model_report.txt (performance report)")
    print("  - 6 visualization plots (PNG files)")
    print("\n")


if __name__ == "__main__":
    main()
