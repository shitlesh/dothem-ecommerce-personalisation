import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             roc_curve, precision_recall_curve, f1_score, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import pickle
import os
import warnings

warnings.filterwarnings('ignore')


class ChurnPredictor:
    """
    Advanced churn prediction system with ensemble methods and comprehensive evaluation.

    Features:
    - Multiple ML algorithms with hyperparameter tuning
    - Feature importance analysis
    - Model interpretability tools
    - Comprehensive performance evaluation
    - Production-ready prediction capabilities
    """

    def __init__(self, model_type='random_forest', random_state=42):
        """
        Initialize the churn prediction system.

        Parameters:
        -----------
        model_type : str, default='random_forest'
            Type of model to use ('random_forest', 'gradient_boosting', 'logistic_regression', 'ensemble')
        random_state : int, default=42
            Random state for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()

        # Model performance tracking
        self.feature_importance = None
        self.performance_metrics = {}
        self.feature_names = []
        self.is_fitted = False

        # Cross-validation results
        self.cv_results = {}
        self.best_params = {}

        # Prediction thresholds
        self.optimal_threshold = 0.5
        self.risk_thresholds = {'low': 0.3, 'medium': 0.6, 'high': 0.8}

    def prepare_features(self, data):
        """
        Prepare features for churn prediction model.

        Parameters:
        -----------
        data : pd.DataFrame
            Processed customer data

        Returns:
        --------
        tuple
            (X, y, feature_names) for model training
        """
        print("🔍 Preparing features for churn prediction...")

        # Define comprehensive feature set
        base_features = [
            'Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp',
            'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress',
            'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount',
            'DaySinceLastOrder', 'CashbackAmount'
        ]

        # Engineered features
        engineered_features = [
            'CustomerValueScore', 'EngagementScore', 'HighRiskCategory',
            'HighRiskPayment', 'LowSatisfaction', 'OrderFrequency',
            'CashbackEfficiency', 'ServiceQualityScore', 'LogisticsConvenience'
        ]

        # Encoded categorical features
        categorical_encoded = [
            'PreferredLoginDevice_encoded', 'PreferredPaymentMode_encoded',
            'Gender_encoded', 'PreferedOrderCat_encoded', 'MaritalStatus_encoded'
        ]

        # Segment information (if available)
        segment_features = ['CustomerSegment'] if 'CustomerSegment' in data.columns else []

        # Combine all features
        all_features = base_features + engineered_features + categorical_encoded + segment_features

        # Filter available features
        available_features = [f for f in all_features if f in data.columns]

        if len(available_features) < 10:
            raise ValueError(f"Insufficient features for modeling. Found: {len(available_features)}")

        # Prepare feature matrix and target
        X = data[available_features].copy()

        # Handle missing values
        X = X.fillna(X.median())

        # Get target variable
        if 'Churn' not in data.columns:
            raise ValueError("Target variable 'Churn' not found in data")

        y = data['Churn'].copy()

        self.feature_names = available_features

        print(f"Prepared {len(available_features)} features for modeling:")
        print(f"Base features: {len([f for f in available_features if f in base_features])}")
        print(f"Engineered features: {len([f for f in available_features if f in engineered_features])}")
        print(f"Categorical features: {len([f for f in available_features if f in categorical_encoded])}")
        print(f"Target distribution: {y.value_counts().to_dict()}")

        return X, y, available_features

    def _get_model_with_params(self):
        """Get model instance with hyperparameter search space."""

        if self.model_type == 'random_forest':
            model = RandomForestClassifier(random_state=self.random_state)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', 'balanced_subsample']
            }

        elif self.model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(random_state=self.random_state)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'min_samples_split': [2, 5, 10]
            }

        elif self.model_type == 'logistic_regression':
            model = LogisticRegression(random_state=self.random_state, max_iter=1000)
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'lbfgs'],
                'class_weight': [None, 'balanced']
            }

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return model, param_grid

    def train_model(self, data, test_size=0.2, cv_folds=5, scoring='roc_auc'):
        """
        Train churn prediction model with comprehensive evaluation.

        Parameters:
        -----------
        data : pd.DataFrame
            Training data
        test_size : float, default=0.2
            Fraction of data to use for testing
        cv_folds : int, default=5
            Number of cross-validation folds
        scoring : str, default='roc_auc'
            Scoring metric for hyperparameter tuning

        Returns:
        --------
        dict
            Model performance metrics
        """
        print(f"\nTRAINING CHURN PREDICTION MODEL ({self.model_type.upper()})")
        print("=" * 60)

        # Prepare features
        X, y, feature_names = self.prepare_features(data)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=y
        )

        print(f"\n Data Split:")
        print(f"   Training set: {len(X_train):,} samples")
        print(f"   Test set: {len(X_test):,} samples")
        print(f"   Features: {len(feature_names)}")
        print(f"   Train churn rate: {y_train.mean():.3f}")
        print(f"   Test churn rate: {y_test.mean():.3f}")

        # Scale features if using logistic regression
        if self.model_type == 'logistic_regression':
            print(f"\n Scaling features for {self.model_type}...")
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            X_train, X_test = X_train_scaled, X_test_scaled

        # Get model and parameter grid
        base_model, param_grid = self._get_model_with_params()

        # Perform hyperparameter tuning
        print(f"\n Hyperparameter tuning with {cv_folds}-fold cross-validation...")

        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        grid_search = GridSearchCV(
            base_model, param_grid, cv=cv_strategy, scoring=scoring,
            n_jobs=-1, verbose=0, return_train_score=True
        )

        grid_search.fit(X_train, y_train)

        # Store best model and parameters
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

        print(f" Best parameters found:")
        for param, value in self.best_params.items():
            print(f"   {param}: {value}")
        print(f"   Best CV score: {grid_search.best_score_:.4f}")

        # Store cross-validation results
        self.cv_results = {
            'best_score': grid_search.best_score_,
            'best_params': self.best_params,
            'cv_scores': grid_search.cv_results_
        }

        # Evaluate model on test set
        print(f"\n Evaluating model on test set...")
        performance_metrics = self._evaluate_model(X_test, y_test, X_train, y_train)

        # Calculate feature importance
        self._calculate_feature_importance()

        # Find optimal threshold
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        self.optimal_threshold = self._find_optimal_threshold(y_test, y_pred_proba)

        print(f" Optimal prediction threshold: {self.optimal_threshold:.3f}")

        # Mark as fitted
        self.is_fitted = True

        return performance_metrics

    def _evaluate_model(self, X_test, y_test, X_train, y_train):
        """Comprehensive model evaluation."""

        # Predictions
        y_pred_test = self.model.predict(X_test)
        y_pred_proba_test = self.model.predict_proba(X_test)[:, 1]

        y_pred_train = self.model.predict(X_train)
        y_pred_proba_train = self.model.predict_proba(X_train)[:, 1]

        # Calculate metrics
        metrics = {
            # Test set metrics
            'test_auc': roc_auc_score(y_test, y_pred_proba_test),
            'test_f1': f1_score(y_test, y_pred_test),
            'test_precision': precision_score(y_test, y_pred_test),
            'test_recall': recall_score(y_test, y_pred_test),

            # Training set metrics (for overfitting detection)
            'train_auc': roc_auc_score(y_train, y_pred_proba_train),
            'train_f1': f1_score(y_train, y_pred_train),

            # Additional metrics
            'classification_report': classification_report(y_test, y_pred_test, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test),

            # ROC and PR curves data
            'roc_curve': roc_curve(y_test, y_pred_proba_test),
            'pr_curve': precision_recall_curve(y_test, y_pred_proba_test)
        }

        # Store performance metrics
        self.performance_metrics = metrics

        # Print key metrics
        print(f"   Model Performance Summary:")
        print(f"   AUC Score: {metrics['test_auc']:.4f}")
        print(f"   F1 Score: {metrics['test_f1']:.4f}")
        print(f"   Precision: {metrics['test_precision']:.4f}")
        print(f"   Recall: {metrics['test_recall']:.4f}")

        # Check for overfitting
        auc_diff = metrics['train_auc'] - metrics['test_auc']
        if auc_diff > 0.05:
            print(
                f"  Potential overfitting detected (Train AUC: {metrics['train_auc']:.4f}, Test AUC: {metrics['test_auc']:.4f})")
        else:
            print(f" Good generalization (AUC difference: {auc_diff:.4f})")

        return metrics

    def _calculate_feature_importance(self):
        """Calculate and store feature importance."""

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            print("Feature importance not available for this model type")
            return

        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print(f"\n Top 10 Most Important Features:")
        for i, (_, row) in enumerate(self.feature_importance.head(10).iterrows(), 1):
            print(f"   {i:2d}. {row['feature']}: {row['importance']:.4f}")

    def _find_optimal_threshold(self, y_true, y_pred_proba):
        """Find optimal prediction threshold using F1 score."""

        thresholds = np.arange(0.1, 0.9, 0.02)
        f1_scores = []

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred)
            f1_scores.append(f1)

        optimal_idx = np.argmax(f1_scores)
        return thresholds[optimal_idx]

    def predict_churn_risk(self, customer_data, return_probabilities=True):
        """
        Predict churn risk for customers.

        Parameters:
        -----------
        customer_data : pd.DataFrame
            Customer data for prediction
        return_probabilities : bool, default=True
            Whether to return probabilities or just risk categories

        Returns:
        --------
        tuple or np.array
            Churn probabilities and risk categories, or just risk categories
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first. Call train_model().")

        # Prepare features (same way as training)
        X = customer_data[self.feature_names].copy()
        X = X.fillna(X.median())

        # Scale if needed
        if self.model_type == 'logistic_regression':
            X = self.scaler.transform(X)

        # Get probabilities
        churn_probabilities = self.model.predict_proba(X)[:, 1]

        # Create risk categories
        risk_categories = np.where(
            churn_probabilities >= self.risk_thresholds['high'], 'High Risk',
            np.where(churn_probabilities >= self.risk_thresholds['medium'], 'Medium Risk',
                     np.where(churn_probabilities >= self.risk_thresholds['low'], 'Low-Medium Risk', 'Low Risk'))
        )

        if return_probabilities:
            return churn_probabilities, risk_categories
        else:
            return risk_categories

    def plot_model_performance(self, figsize=(15, 10)):
        """
        Create comprehensive model performance visualizations.

        Parameters:
        -----------
        figsize : tuple, default=(15, 10)
            Figure size for the plots
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first. Call train_model().")

        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # Plot 1: Confusion Matrix
        cm = self.performance_metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')

        # Plot 2: ROC Curve
        fpr, tpr, _ = self.performance_metrics['roc_curve']
        auc = self.performance_metrics['test_auc']

        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend(loc="lower right")

        # Plot 3: Precision-Recall Curve
        precision, recall, _ = self.performance_metrics['pr_curve']

        axes[0, 2].plot(recall, precision, color='blue', lw=2)
        axes[0, 2].axhline(y=self.performance_metrics['test_precision'], color='red',
                           linestyle='--', label=f'Precision = {self.performance_metrics["test_precision"]:.3f}')
        axes[0, 2].set_xlabel('Recall')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].set_title('Precision-Recall Curve')
        axes[0, 2].legend()

        # Plot 4: Feature Importance (Top 15)
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(15)

            axes[1, 0].barh(range(len(top_features)), top_features['importance'])
            axes[1, 0].set_yticks(range(len(top_features)))
            axes[1, 0].set_yticklabels(top_features['feature'])
            axes[1, 0].set_xlabel('Feature Importance')
            axes[1, 0].set_title('Top 15 Feature Importances')
            axes[1, 0].invert_yaxis()

        # Plot 5: Model Metrics Comparison
        metrics_names = ['AUC', 'F1', 'Precision', 'Recall']
        test_scores = [
            self.performance_metrics['test_auc'],
            self.performance_metrics['test_f1'],
            self.performance_metrics['test_precision'],
            self.performance_metrics['test_recall']
        ]

        axes[1, 1].bar(metrics_names, test_scores, color=['skyblue', 'lightgreen', 'coral', 'gold'])
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Model Performance Metrics')

        # Add value labels on bars
        for i, v in enumerate(test_scores):
            axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

        # Plot 6: Class Distribution in Predictions
        if hasattr(self, '_last_predictions'):
            pred_dist = pd.Series(self._last_predictions).value_counts()
            axes[1, 2].pie(pred_dist.values, labels=pred_dist.index, autopct='%1.1f%%')
            axes[1, 2].set_title('Prediction Distribution')
        else:
            axes[1, 2].text(0.5, 0.5, 'Prediction\nDistribution\n(Run predictions first)',
                            ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Prediction Distribution')

        plt.tight_layout()
        return fig

    def get_feature_importance(self, top_n=20):
        """
        Get feature importance dataframe.

        Parameters:
        -----------
        top_n : int, default=20
            Number of top features to return

        Returns:
        --------
        pd.DataFrame
            Feature importance rankings
        """
        if self.feature_importance is None:
            raise ValueError("Feature importance not available. Train model first.")

        return self.feature_importance.head(top_n)

    def generate_model_report(self):
        """
        Generate comprehensive model performance report.

        Returns:
        --------
        dict
            Detailed model performance report
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first. Call train_model().")

        report = {
            'model_info': {
                'model_type': self.model_type,
                'best_parameters': self.best_params,
                'optimal_threshold': self.optimal_threshold,
                'feature_count': len(self.feature_names)
            },
            'performance_metrics': {
                'auc_score': self.performance_metrics['test_auc'],
                'f1_score': self.performance_metrics['test_f1'],
                'precision': self.performance_metrics['test_precision'],
                'recall': self.performance_metrics['test_recall']
            },
            'cross_validation': {
                'cv_auc_mean': self.cv_results['best_score'],
                'cv_auc_std': np.std([score for score in self.cv_results['cv_scores']['mean_test_score']])
            },
            'feature_importance': self.feature_importance.head(10).to_dict(
                'records') if self.feature_importance is not None else [],
            'business_impact': self._calculate_business_impact()
        }

        return report

    def _calculate_business_impact(self):
        """Calculate potential business impact of the model."""

        if not self.performance_metrics:
            return {}

        # Assumptions for business impact calculation
        total_customers = 10000  # Example customer base
        avg_customer_value = 1200  # Average customer lifetime value
        churn_rate = 0.168  # Overall churn rate (from analysis)

        # Model performance metrics
        precision = self.performance_metrics['test_precision']
        recall = self.performance_metrics['test_recall']

        # Calculate potential impact
        customers_at_risk = int(total_customers * churn_rate)
        customers_identified = int(customers_at_risk * recall)
        true_positives = int(customers_identified * precision)

        # Assume 30% of identified at-risk customers can be saved
        retention_rate = 0.30
        customers_saved = int(true_positives * retention_rate)

        revenue_saved = customers_saved * avg_customer_value

        business_impact = {
            'customers_at_risk': customers_at_risk,
            'customers_identified_by_model': customers_identified,
            'true_at_risk_identified': true_positives,
            'estimated_customers_saved': customers_saved,
            'estimated_revenue_saved': revenue_saved,
            'model_precision': precision,
            'model_recall': recall
        }

        return business_impact

    def save_model(self, file_path='models/churn_prediction_model.pkl'):
        """
        Save the trained churn prediction model.

        Parameters:
        -----------
        file_path : str
            Path where to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first. Call train_model().")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Prepare model data
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'performance_metrics': self.performance_metrics,
            'best_params': self.best_params,
            'optimal_threshold': self.optimal_threshold,
            'risk_thresholds': self.risk_thresholds,
            'cv_results': self.cv_results
        }

        # Save model
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f" Churn prediction model saved to: {file_path}")

    def load_model(self, file_path):
        """
        Load a trained churn prediction model.

        Parameters:
        -----------
        file_path : str
            Path to the saved model
        """
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)

            # Restore model components
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.model_type = model_data['model_type']
            self.feature_names = model_data['feature_names']
            self.feature_importance = model_data['feature_importance']
            self.performance_metrics = model_data['performance_metrics']
            self.best_params = model_data['best_params']
            self.optimal_threshold = model_data['optimal_threshold']
            self.risk_thresholds = model_data['risk_thresholds']
            self.cv_results = model_data['cv_results']

            self.is_fitted = True
            print(f" Churn prediction model loaded from: {file_path}")

        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")


def demonstrate_churn_prediction():
    """Demonstration function for the churn prediction module."""
    print(" Churn Prediction Module Demo")
    print("=" * 50)

    # Create sample data for demonstration
    np.random.seed(42)
    n_customers = 2000

    # Generate realistic sample data
    sample_data = pd.DataFrame({
        'CustomerID': range(1, n_customers + 1),
        'Tenure': np.random.exponential(12, n_customers),
        'CityTier': np.random.choice([1, 2, 3], n_customers, p=[0.6, 0.2, 0.2]),
        'WarehouseToHome': np.random.gamma(2, 8, n_customers),
        'HourSpendOnApp': np.random.gamma(2, 1.5, n_customers),
        'NumberOfDeviceRegistered': np.random.poisson(3, n_customers) + 1,
        'SatisfactionScore': np.random.randint(1, 6, n_customers),
        'NumberOfAddress': np.random.poisson(5, n_customers) + 1,
        'Complain': np.random.binomial(1, 0.25, n_customers),
        'OrderAmountHikeFromlastYear': np.random.normal(15, 8, n_customers),
        'CouponUsed': np.random.poisson(2, n_customers),
        'OrderCount': np.random.poisson(3, n_customers) + 1,
        'DaySinceLastOrder': np.random.exponential(8, n_customers),
        'CashbackAmount': np.random.normal(120, 30, n_customers),

        # Engineered features
        'CustomerValueScore': np.random.normal(1.5, 0.8, n_customers),
        'EngagementScore': np.random.normal(2.0, 1.0, n_customers),
        'HighRiskCategory': np.random.binomial(1, 0.3, n_customers),
        'HighRiskPayment': np.random.binomial(1, 0.2, n_customers),
        'LowSatisfaction': np.random.binomial(1, 0.25, n_customers),
        'OrderFrequency': np.random.gamma(1, 0.5, n_customers),
        'CashbackEfficiency': np.random.gamma(2, 15, n_customers),
        'ServiceQualityScore': np.random.normal(3.5, 1.2, n_customers),
        'LogisticsConvenience': np.random.binomial(1, 0.6, n_customers),

        # Encoded categorical features
        'PreferredLoginDevice_encoded': np.random.randint(0, 3, n_customers),
        'PreferredPaymentMode_encoded': np.random.randint(0, 7, n_customers),
        'Gender_encoded': np.random.randint(0, 2, n_customers),
        'PreferedOrderCat_encoded': np.random.randint(0, 6, n_customers),
        'MaritalStatus_encoded': np.random.randint(0, 3, n_customers),

        # Customer segment
        'CustomerSegment': np.random.randint(0, 5, n_customers)
    })

    # Generate realistic churn target based on features
    churn_probability = (
            0.3 * sample_data['Complain'] +
            0.2 * (sample_data['SatisfactionScore'] <= 2) +
            0.2 * sample_data['HighRiskCategory'] +
            0.15 * (sample_data['DaySinceLastOrder'] > 15) +
            0.1 * sample_data['HighRiskPayment'] +
            0.05 * np.random.random(n_customers)  # Random noise
    )

    sample_data['Churn'] = np.random.binomial(1, np.clip(churn_probability, 0, 1), n_customers)

    print(f" Generated {n_customers} customers with {sample_data['Churn'].mean():.1%} churn rate")

    try:
        # Test different model types
        model_types = ['random_forest', 'gradient_boosting', 'logistic_regression']
        results = {}

        for model_type in model_types:
            print(f"\n Testing {model_type.replace('_', ' ').title()} model...")

            # Initialize predictor
            predictor = ChurnPredictor(model_type=model_type)

            # Train model
            performance = predictor.train_model(sample_data, cv_folds=3)  # Reduced CV for demo

            results[model_type] = {
                'auc': performance['test_auc'],
                'f1': performance['test_f1'],
                'predictor': predictor
            }

            print(f"   AUC: {performance['test_auc']:.3f}, F1: {performance['test_f1']:.3f}")

        # Find best model
        best_model_type = max(results.keys(), key=lambda k: results[k]['auc'])
        best_predictor = results[best_model_type]['predictor']

        print(f"\n Best model: {best_model_type.replace('_', ' ').title()}")
        print(f"   Best AUC: {results[best_model_type]['auc']:.3f}")

        # Test predictions on sample customers
        sample_customers = sample_data.head(10)
        probabilities, risk_categories = best_predictor.predict_churn_risk(sample_customers)

        print(f"\n Sample Predictions:")
        for i in range(min(5, len(sample_customers))):
            print(f"   Customer {i + 1}: {probabilities[i]:.3f} probability ({risk_categories[i]})")

        return best_predictor, sample_data, results

    except Exception as e:
        print(f" Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    # Run demonstration
    predictor, data, results = demonstrate_churn_prediction()

    if predictor is not None:
        print(f"\n Churn prediction module ready for production!")
        print(f" Best model trained and validated")
        print(f" {len(data)} customers analyzed")

        # Generate model report
        report = predictor.generate_model_report()
        print(f" Estimated revenue impact: ${report['business_impact']['estimated_revenue_saved']:,.0f}")
