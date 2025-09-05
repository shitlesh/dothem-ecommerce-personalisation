
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings

warnings.filterwarnings('ignore')


class DohtemDataProcessor:
    """
    Comprehensive data processor for Dohtem e-commerce customer data.

    This class handles:
    - Data loading and validation
    - Missing value treatment
    - Feature engineering
    - Categorical encoding
    - Data quality assessment
    """

    def __init__(self, data_path):
        """
        Initialize the data processor.

        Parameters:
        -----------
        data_path : str
            Path to the CSV file containing customer data
        """
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.data_quality_report = {}

    def load_data(self):
        """
        Load raw data from CSV file and perform initial validation.

        Returns:
        --------
        pd.DataFrame
            Raw customer data
        """
        try:
            self.raw_data = pd.read_csv(self.data_path)
            print(f"Data loaded successfully!")
            print(f"Dataset shape: {self.raw_data.shape}")
            print(f"Columns: {list(self.raw_data.columns)}")

            # Basic validation
            if 'CustomerID' not in self.raw_data.columns:
                raise ValueError("CustomerID column not found in dataset")
            if 'Churn' not in self.raw_data.columns:
                raise ValueError("Churn column not found in dataset")

            return self.raw_data

        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def explore_data(self):
        """
        Perform comprehensive data exploration and quality assessment.

        Returns:
        --------
        pd.DataFrame
            Missing data analysis report
        """
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print("\n" + "=" * 60)
        print("COMPREHENSIVE DATA EXPLORATION")
        print("=" * 60)

        # Dataset overview
        print(f"\n1. DATASET OVERVIEW:")
        print(f"Rows: {self.raw_data.shape[0]:,}")
        print(f"Columns: {self.raw_data.shape[1]}")
        print(f"Memory usage: {self.raw_data.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

        # Data types
        print(f"\n2. DATA TYPES:")
        dtype_counts = self.raw_data.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"   {dtype}: {count} columns")

        # Missing values analysis
        print(f"\n3. MISSING VALUES ANALYSIS:")
        missing_data = self.raw_data.isnull().sum()
        missing_percent = (missing_data / len(self.raw_data)) * 100

        missing_df = pd.DataFrame({
            'Missing_Count': missing_data,
            'Missing_Percent': missing_percent
        })

        missing_features = missing_df[missing_df.Missing_Count > 0].sort_values('Missing_Percent', ascending=False)

        if not missing_features.empty:
            print(f"Features with missing values:")
            for feature, row in missing_features.iterrows():
                print(f"      {feature}: {row.Missing_Count:,} ({row.Missing_Percent:.1f}%)")
        else:
            print(f"No missing values found!")

        # Target variable analysis
        print(f"\n4. TARGET VARIABLE ANALYSIS (CHURN):")
        churn_dist = self.raw_data['Churn'].value_counts()
        churn_percent = self.raw_data['Churn'].value_counts(normalize=True) * 100

        print(f"   No Churn (0): {churn_dist[0]:,} customers ({churn_percent[0]:.1f}%)")
        print(f"   Churned (1): {churn_dist[1]:,} customers ({churn_percent[1]:.1f}%)")

        # Categorical variables analysis
        print(f"\n5. CATEGORICAL VARIABLES:")
        categorical_cols = self.raw_data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            unique_values = self.raw_data[col].nunique()
            print(f"   {col}: {unique_values} unique values")
            if unique_values <= 10:
                value_counts = self.raw_data[col].value_counts()
                print(f"      Top values: {dict(value_counts.head(3))}")

        # Numerical variables analysis
        print(f"\n6. NUMERICAL VARIABLES SUMMARY:")
        numerical_cols = self.raw_data.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != 'CustomerID']

        if len(numerical_cols) > 0:
            print(f"Statistical summary:")
            stats_summary = self.raw_data[numerical_cols].describe()
            print(stats_summary.round(2))

        # Store data quality report
        self.data_quality_report = {
            'total_rows': len(self.raw_data),
            'total_columns': len(self.raw_data.columns),
            'missing_data_summary': missing_df,
            'churn_distribution': churn_dist.to_dict(),
            'categorical_summary': {col: self.raw_data[col].nunique() for col in categorical_cols},
            'numerical_summary': stats_summary.to_dict() if len(numerical_cols) > 0 else {}
        }

        return missing_df

    def preprocess_data(self):
        """
        Clean and preprocess the data for machine learning.

        Returns:
        --------
        pd.DataFrame
            Preprocessed and feature-engineered dataset
        """
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print("\n" + "=" * 60)
        print("DATA PREPROCESSING & FEATURE ENGINEERING")
        print("=" * 60)

        # Create a copy for processing
        self.processed_data = self.raw_data.copy()

        # Step 1: Handle missing values
        print(f"\n1. HANDLING MISSING VALUES:")
        self._handle_missing_values()

        # Step 2: Feature engineering
        print(f"\n2. FEATURE ENGINEERING:")
        self._create_engineered_features()

        # Step 3: Encode categorical variables
        print(f"\n3. ENCODING CATEGORICAL VARIABLES:")
        self._encode_categorical_variables()

        # Step 4: Data validation
        print(f"\n4. DATA VALIDATION:")
        self._validate_processed_data()

        print(f"\nData preprocessing completed!")
        print(f"Final dataset shape: {self.processed_data.shape}")
        print(f"New features created: {self.processed_data.shape[1] - self.raw_data.shape[1]}")

        return self.processed_data

    def _handle_missing_values(self):
        """Handle missing values using appropriate strategies."""

        # Numerical columns - use median imputation
        numerical_cols = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp',
                          'OrderAmountHikeFromlastYear', 'CouponUsed',
                          'OrderCount', 'DaySinceLastOrder']

        for col in numerical_cols:
            if col in self.processed_data.columns:
                missing_count = self.processed_data[col].isnull().sum()
                if missing_count > 0:
                    median_value = self.processed_data[col].median()
                    self.processed_data[col].fillna(median_value, inplace=True)
                    print(f"{col}: Filled {missing_count} missing values with median ({median_value:.2f})")

        # Create missing value indicators for high-missing features
        high_missing_threshold = 0.15  # 15% threshold
        for col in numerical_cols:
            if col in self.processed_data.columns:
                missing_rate = self.raw_data[col].isnull().mean()
                if missing_rate > high_missing_threshold:
                    indicator_name = f'{col}_WasMissing'
                    self.processed_data[indicator_name] = self.raw_data[col].isnull().astype(int)
                    print(f"Created missing indicator: {indicator_name}")

    def _create_engineered_features(self):
        """Create advanced engineered features."""

        # Customer Value Score
        # Combines order frequency, cashback, app usage, and satisfaction
        self.processed_data['CustomerValueScore'] = (
                self.processed_data['OrderCount'].fillna(0) * 0.3 +
                self.processed_data['CashbackAmount'] * 0.001 +  # Scale down cashback
                self.processed_data['HourSpendOnApp'].fillna(0) * 0.2 +
                (6 - self.processed_data['SatisfactionScore']) * -0.1  # Inverse relationship
        )
        print(f"CustomerValueScore created (mean: {self.processed_data['CustomerValueScore'].mean():.2f})")

        # Engagement Score
        # Measures customer engagement across multiple dimensions
        self.processed_data['EngagementScore'] = (
                self.processed_data['HourSpendOnApp'].fillna(0) * 0.4 +
                self.processed_data['OrderCount'].fillna(0) * 0.3 +
                self.processed_data['NumberOfDeviceRegistered'] * 0.2 +
                np.maximum(0, 30 - self.processed_data['DaySinceLastOrder'].fillna(30)) * 0.1  # Recent activity bonus
        )
        print(f"EngagementScore created (mean: {self.processed_data['EngagementScore'].mean():.2f})")

        # Risk Indicator Features
        self.processed_data['HighRiskCategory'] = self.processed_data['PreferedOrderCat'].isin(
            ['Mobile', 'Mobile Phone']
        ).astype(int)

        self.processed_data['HighRiskPayment'] = self.processed_data['PreferredPaymentMode'].isin(
            ['COD', 'Cash on Delivery']
        ).astype(int)

        self.processed_data['LowSatisfaction'] = (self.processed_data['SatisfactionScore'] <= 2).astype(int)

        print(f"Risk indicators created: HighRiskCategory, HighRiskPayment, LowSatisfaction")

        # Behavioral Features
        self.processed_data['OrderFrequency'] = np.where(
            self.processed_data['Tenure'].fillna(1) > 0,
            self.processed_data['OrderCount'].fillna(0) / self.processed_data['Tenure'].fillna(1),
            0
        )

        self.processed_data['CashbackEfficiency'] = np.where(
            self.processed_data['OrderCount'].fillna(1) > 0,
            self.processed_data['CashbackAmount'] / self.processed_data['OrderCount'].fillna(1),
            0
        )

        # Loyalty Indicators
        self.processed_data['IsLoyalCustomer'] = (
                (self.processed_data['Tenure'].fillna(0) > 12) &
                (self.processed_data['OrderCount'].fillna(0) > 5)
        ).astype(int)

        self.processed_data['HighValueCustomer'] = (
                self.processed_data['CustomerValueScore'] > self.processed_data['CustomerValueScore'].quantile(0.75)
        ).astype(int)

        print(f"Behavioral & loyalty features created")

        # Geographic & Service Quality Features
        self.processed_data['ServiceQualityScore'] = (
                self.processed_data['SatisfactionScore'] * 0.6 +
                (1 - self.processed_data['Complain']) * 2 * 0.4  # No complaints = better service
        )

        self.processed_data['LogisticsConvenience'] = np.where(
            self.processed_data['WarehouseToHome'].fillna(50) <= 15, 1, 0
        )

        print(f"Service quality & logistics features created")

    def _encode_categorical_variables(self):
        """Encode categorical variables for machine learning."""

        categorical_cols = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender',
                            'PreferedOrderCat', 'MaritalStatus']

        for col in categorical_cols:
            if col in self.processed_data.columns:
                # Handle any missing values in categorical columns
                mode_value = self.processed_data[col].mode()[0] if not self.processed_data[
                    col].mode().empty else 'Unknown'
                self.processed_data[col].fillna(mode_value, inplace=True)

                # Label encode
                le = LabelEncoder()
                self.processed_data[f'{col}_encoded'] = le.fit_transform(self.processed_data[col])
                self.label_encoders[col] = le

                print(f"{col}: {len(le.classes_)} unique values encoded")

        # Create interaction features between important categorical variables
        if all(col in self.processed_data.columns for col in ['Gender', 'PreferedOrderCat']):
            self.processed_data['Gender_Category_Interaction'] = (
                    self.processed_data['Gender_encoded'].astype(str) + '_' +
                    self.processed_data['PreferedOrderCat_encoded'].astype(str)
            )

            # Label encode the interaction
            le_interaction = LabelEncoder()
            self.processed_data['Gender_Category_Interaction_encoded'] = le_interaction.fit_transform(
                self.processed_data['Gender_Category_Interaction']
            )
            self.label_encoders['Gender_Category_Interaction'] = le_interaction
            print(f"Gender-Category interaction feature created")

    def _validate_processed_data(self):
        """Validate the processed data quality."""

        # Check for remaining missing values
        missing_count = self.processed_data.isnull().sum().sum()
        if missing_count > 0:
            print(f"Warning: {missing_count} missing values remain")
            missing_cols = self.processed_data.columns[self.processed_data.isnull().any()].tolist()
            print(f"Columns with missing values: {missing_cols}")
        else:
            print(f"No missing values remaining")

        # Check for infinite values
        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns
        inf_count = np.isinf(self.processed_data[numeric_cols]).sum().sum()
        if inf_count > 0:
            print(f"Warning: {inf_count} infinite values found")
        else:
            print(f"No infinite values found")

        # Check data types
        print(f"Final data types:")
        dtype_summary = self.processed_data.dtypes.value_counts()
        for dtype, count in dtype_summary.items():
            print(f"{dtype}: {count} columns")

        # Feature summary
        engineered_features = [col for col in self.processed_data.columns
                               if col not in self.raw_data.columns and col != 'CustomerSegment']
        print(f"Engineered features: {len(engineered_features)}")
        for feature in engineered_features[:10]:  # Show first 10
            print(f"{feature}")
        if len(engineered_features) > 10:
            print(f"... and {len(engineered_features) - 10} more")

    def get_feature_importance_data(self):
        """
        Get data suitable for feature importance analysis.

        Returns:
        --------
        tuple
            (feature_matrix, target_vector, feature_names)
        """
        if self.processed_data is None:
            raise ValueError("Data not processed. Call preprocess_data() first.")

        # Select features for modeling (exclude ID and string columns)
        exclude_cols = ['CustomerID'] + [col for col in self.processed_data.columns
                                         if self.processed_data[col].dtype == 'object' and not col.endswith('_encoded')]

        feature_cols = [col for col in self.processed_data.columns
                        if col not in exclude_cols and col != 'Churn']

        X = self.processed_data[feature_cols].fillna(0)
        y = self.processed_data['Churn']

        return X, y, feature_cols

    def save_processed_data(self, file_path='data/processed/processed_customer_data.csv'):
        """
        Save processed data to CSV file.

        Parameters:
        -----------
        file_path : str
            Path where to save the processed data
        """
        if self.processed_data is None:
            raise ValueError("No processed data to save. Call preprocess_data() first.")

        import os
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        self.processed_data.to_csv(file_path, index=False)
        print(f"Processed data saved to: {file_path}")

    def create_data_quality_report(self):
        """
        Create a comprehensive data quality report.

        Returns:
        --------
        dict
            Data quality metrics and insights
        """
        if self.processed_data is None:
            raise ValueError("Data not processed. Call preprocess_data() first.")

        report = {
            'dataset_overview': {
                'total_customers': len(self.processed_data),
                'total_features': len(self.processed_data.columns),
                'churn_rate': self.processed_data['Churn'].mean(),
                'data_completeness': (1 - self.processed_data.isnull().sum().sum() /
                                      (len(self.processed_data) * len(self.processed_data.columns)))
            },
            'feature_summary': {
                'numerical_features': len(self.processed_data.select_dtypes(include=[np.number]).columns),
                'categorical_features': len(self.processed_data.select_dtypes(include=['object']).columns),
                'engineered_features': len([col for col in self.processed_data.columns
                                            if col not in self.raw_data.columns])
            },
            'data_quality_scores': {
                'completeness': 1 - (self.processed_data.isnull().sum().sum() /
                                     (len(self.processed_data) * len(self.processed_data.columns))),
                'consistency': 1.0,  # Assume consistent after processing
                'validity': 1.0 if not np.isinf(
                    self.processed_data.select_dtypes(include=[np.number])).any().any() else 0.9
            }
        }

        return report


if __name__ == "__main__":
    """Example usage of the data processor."""
    print("🔍 Testing Dohtem Data Processor...")

    # Initialize processor
    processor = DohtemDataProcessor('dohtem_ecommerce_customers.csv')

    try:
        # Load and explore data
        raw_data = processor.load_data()
        missing_analysis = processor.explore_data()

        # Process data
        processed_data = processor.preprocess_data()

        # Generate quality report
        quality_report = processor.create_data_quality_report()
        print(f"\nData Quality Summary:")
        print(f"   Completeness: {quality_report['data_quality_scores']['completeness']:.2%}")
        print(
            f"   Total Features: {quality_report['feature_summary']['numerical_features'] + quality_report['feature_summary']['categorical_features']}")
        print(f"   Engineered Features: {quality_report['feature_summary']['engineered_features']}")

        print("\nData processor test completed successfully!")

    except Exception as e:
        print(f"Error testing data processor: {str(e)}")
