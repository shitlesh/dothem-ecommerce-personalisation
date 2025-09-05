import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import pickle
import os
import warnings

warnings.filterwarnings('ignore')


class CustomerSegmentation:
    """
    Advanced customer segmentation using K-Means clustering.

    This class provides:
    - Optimal cluster number determination
    - Customer segmentation with detailed profiles
    - Comprehensive visualization capabilities
    - Model persistence and loading
    """

    def __init__(self, n_clusters=5, random_state=42):
        """
        Initialize the customer segmentation model.

        Parameters:
        -----------
        n_clusters : int, default=5
            Number of customer segments to create
        random_state : int, default=42
            Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)

        # Model attributes
        self.segment_features = None
        self.segment_profiles = {}
        self.feature_names = []
        self.is_fitted = False
        self.silhouette_scores = {}
        self.cluster_validation_metrics = {}

    def prepare_features(self, data):
        """
        Prepare and select features for customer segmentation.

        Parameters:
        -----------
        data : pd.DataFrame
            Processed customer data

        Returns:
        --------
        pd.DataFrame
            Selected and prepared features for clustering
        """
        print("Preparing features for customer segmentation...")

        # Select relevant features for segmentation
        segmentation_features = [
            'Tenure', 'HourSpendOnApp', 'NumberOfDeviceRegistered',
            'SatisfactionScore', 'NumberOfAddress', 'OrderCount',
            'CashbackAmount', 'CustomerValueScore', 'EngagementScore',
            'WarehouseToHome', 'OrderAmountHikeFromlastYear', 'CouponUsed',
            'OrderFrequency', 'CashbackEfficiency', 'ServiceQualityScore'
        ]

        # Filter features that exist in the dataset
        available_features = [f for f in segmentation_features if f in data.columns]

        if len(available_features) < 5:
            raise ValueError(f"Insufficient features for segmentation. Found: {available_features}")

        self.feature_names = available_features
        self.segment_features = data[available_features].copy()

        # Handle any remaining missing values
        self.segment_features = self.segment_features.fillna(self.segment_features.median())

        print(f"Selected {len(available_features)} features for segmentation:")
        for i, feature in enumerate(available_features, 1):
            print(f"{i:2d}. {feature}")

        return self.segment_features

    def find_optimal_clusters(self, data, max_clusters=10):
        """
        Find optimal number of clusters using multiple metrics.

        Parameters:
        -----------
        data : pd.DataFrame
            Data for clustering
        max_clusters : int, default=10
            Maximum number of clusters to test

        Returns:
        --------
        dict
            Clustering evaluation metrics
        """
        print(f"\nFinding optimal number of clusters (testing 2-{max_clusters})...")

        features = self.prepare_features(data)
        features_scaled = self.scaler.fit_transform(features)

        inertias = []
        silhouette_scores = []
        ch_scores = []
        cluster_range = range(2, min(max_clusters + 1, len(features) // 2))

        for k in cluster_range:
            # Fit K-means
            kmeans_temp = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans_temp.fit_predict(features_scaled)

            # Calculate metrics
            inertia = kmeans_temp.inertia_
            sil_score = silhouette_score(features_scaled, cluster_labels)
            ch_score = calinski_harabasz_score(features_scaled, cluster_labels)

            inertias.append(inertia)
            silhouette_scores.append(sil_score)
            ch_scores.append(ch_score)

            print(f"   K={k}: Inertia={inertia:.0f}, Silhouette={sil_score:.3f}, CH={ch_score:.0f}")

        # Store results
        self.cluster_validation_metrics = {
            'cluster_range': list(cluster_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'calinski_harabasz_scores': ch_scores
        }

        # Find optimal k using silhouette score
        optimal_k_idx = np.argmax(silhouette_scores)
        optimal_k = cluster_range[optimal_k_idx]

        print(f"\n Recommended number of clusters: {optimal_k}")
        print(f"   Best Silhouette Score: {silhouette_scores[optimal_k_idx]:.3f}")

        return self.cluster_validation_metrics

    def fit_segments(self, data, find_optimal=False):
        """
        Fit customer segmentation model.

        Parameters:
        -----------
        data : pd.DataFrame
            Customer data for segmentation
        find_optimal : bool, default=False
            Whether to find optimal number of clusters first

        Returns:
        --------
        pd.DataFrame
            Data with customer segments assigned
        """
        print(f"\n👥 FITTING CUSTOMER SEGMENTATION MODEL")
        print("=" * 50)

        # Optionally find optimal clusters
        if find_optimal:
            self.find_optimal_clusters(data)

        # Prepare features
        features = self.prepare_features(data)

        # Standardize features
        features_scaled = self.scaler.fit_transform(features)

        # Fit clustering model
        print(f"\n Training K-Means with {self.n_clusters} clusters...")
        self.kmeans.fit(features_scaled)

        # Get cluster labels
        cluster_labels = self.kmeans.predict(features_scaled)
        data_segmented = data.copy()
        data_segmented['CustomerSegment'] = cluster_labels

        # Calculate clustering quality metrics
        silhouette_avg = silhouette_score(features_scaled, cluster_labels)
        ch_score = calinski_harabasz_score(features_scaled, cluster_labels)

        print(f" Segmentation completed!")
        print(f"   Silhouette Score: {silhouette_avg:.3f}")
        print(f"   Calinski-Harabasz Score: {ch_score:.0f}")
        print(f"   Inertia: {self.kmeans.inertia_:.0f}")

        # Create detailed segment profiles
        self.create_segment_profiles(data_segmented)

        # Mark as fitted
        self.is_fitted = True

        return data_segmented

    def create_segment_profiles(self, data):
        """
        Create detailed profiles for each customer segment.

        Parameters:
        -----------
        data : pd.DataFrame
            Data with segment assignments
        """
        print(f"\n CREATING DETAILED SEGMENT PROFILES")
        print("=" * 50)

        self.segment_profiles = {}

        for segment_id in range(self.n_clusters):
            segment_data = data[data['CustomerSegment'] == segment_id]

            if len(segment_data) == 0:
                continue

            # Basic segment metrics
            profile = {
                'segment_id': segment_id,
                'size': len(segment_data),
                'percentage': len(segment_data) / len(data) * 100,

                # Churn and risk metrics
                'churn_rate': segment_data['Churn'].mean(),
                'avg_satisfaction': segment_data['SatisfactionScore'].mean(),
                'complaint_rate': segment_data['Complain'].mean(),

                # Behavioral metrics
                'avg_tenure': segment_data['Tenure'].mean(),
                'avg_orders': segment_data['OrderCount'].mean(),
                'avg_hours_app': segment_data['HourSpendOnApp'].mean(),
                'avg_cashback': segment_data['CashbackAmount'].mean(),

                # Value and engagement
                'avg_value_score': segment_data['CustomerValueScore'].mean(),
                'avg_engagement': segment_data['EngagementScore'].mean(),

                # Demographic and preference info
                'top_category': segment_data['PreferedOrderCat'].mode().iloc[0] if len(segment_data) > 0 else 'Unknown',
                'dominant_gender': segment_data['Gender'].mode().iloc[0] if len(segment_data) > 0 else 'Unknown',
                'primary_device': segment_data['PreferredLoginDevice'].mode().iloc[0] if len(
                    segment_data) > 0 else 'Unknown',
                'top_payment': segment_data['PreferredPaymentMode'].mode().iloc[0] if len(
                    segment_data) > 0 else 'Unknown',

                # Geographic and logistics
                'avg_distance': segment_data['WarehouseToHome'].mean(),
                'avg_addresses': segment_data['NumberOfAddress'].mean(),
                'avg_devices': segment_data['NumberOfDeviceRegistered'].mean(),

                # Advanced metrics
                'high_value_customers': (
                            segment_data['CustomerValueScore'] > segment_data['CustomerValueScore'].quantile(
                        0.75)).sum(),
                'loyal_customers': (segment_data.get('IsLoyalCustomer', 0) == 1).sum(),
                'at_risk_customers': (segment_data['Churn'] == 1).sum()
            }

            # Calculate segment characteristics
            profile['risk_level'] = self._determine_risk_level(profile['churn_rate'])
            profile['value_tier'] = self._determine_value_tier(profile['avg_value_score'])
            profile['engagement_level'] = self._determine_engagement_level(profile['avg_engagement'])

            # Generate segment description
            profile['description'] = self._generate_segment_description(profile)

            self.segment_profiles[segment_id] = profile

            # Print segment summary
            print(f"\n  Segment {segment_id} - {profile['description']}")
            print(f"    Size: {profile['size']:,} customers ({profile['percentage']:.1f}%)")
            print(f"    Churn Rate: {profile['churn_rate']:.1%}")
            print(f"    Satisfaction: {profile['avg_satisfaction']:.1f}/5")
            print(f"    Avg Orders: {profile['avg_orders']:.1f}")
            print(f"    Value Score: {profile['avg_value_score']:.2f}")
            print(f"    Top Category: {profile['top_category']}")
            print(f"    Risk Level: {profile['risk_level']}")

    def _determine_risk_level(self, churn_rate):
        """Determine risk level based on churn rate."""
        if churn_rate >= 0.25:
            return "High"
        elif churn_rate >= 0.15:
            return "Medium"
        else:
            return "Low"

    def _determine_value_tier(self, value_score):
        """Determine value tier based on value score."""
        if value_score >= 2.0:
            return "Premium"
        elif value_score >= 1.0:
            return "Standard"
        else:
            return "Basic"

    def _determine_engagement_level(self, engagement_score):
        """Determine engagement level based on engagement score."""
        if engagement_score >= 3.0:
            return "High"
        elif engagement_score >= 1.5:
            return "Medium"
        else:
            return "Low"

    def _generate_segment_description(self, profile):
        """Generate human-readable segment description."""
        risk = profile['risk_level']
        value = profile['value_tier']
        engagement = profile['engagement_level']
        category = profile['top_category']

        if risk == "High" and value == "Premium":
            return f"High-Value At-Risk {category} Customers"
        elif risk == "Low" and value == "Premium":
            return f"Premium Loyal {category} Customers"
        elif risk == "High" and engagement == "Low":
            return f"Disengaged At-Risk {category} Customers"
        elif engagement == "High" and value == "Standard":
            return f"Engaged {category} Customers"
        else:
            return f"{value} {category} Customers"

    def predict_segment(self, new_data):
        """
        Predict segment for new customers.

        Parameters:
        -----------
        new_data : pd.DataFrame
            New customer data

        Returns:
        --------
        np.array
            Predicted segment labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first. Call fit_segments().")

        # Prepare features same way as training
        features = new_data[self.feature_names].fillna(new_data[self.feature_names].median())
        features_scaled = self.scaler.transform(features)

        # Predict segments
        segments = self.kmeans.predict(features_scaled)
        return segments

    def plot_segments(self, figsize=(15, 10)):
        """
        Create comprehensive visualization of customer segments.

        Parameters:
        -----------
        figsize : tuple, default=(15, 10)
            Figure size for the plots
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first. Call fit_segments().")

        # Create subplot layout
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Customer Segmentation Analysis', fontsize=16, y=1.02)

        # Plot 1: Segment sizes
        segment_sizes = [profile['size'] for profile in self.segment_profiles.values()]
        segment_labels = [f"Segment {i}\n({profile['size']} customers)"
                          for i, profile in self.segment_profiles.items()]

        axes[0, 0].pie(segment_sizes, labels=segment_labels, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Segment Distribution')

        # Plot 2: Churn rates by segment
        segments = list(self.segment_profiles.keys())
        churn_rates = [self.segment_profiles[seg]['churn_rate'] * 100 for seg in segments]
        colors = ['red' if rate > 20 else 'orange' if rate > 15 else 'green' for rate in churn_rates]

        axes[0, 1].bar(segments, churn_rates, color=colors, alpha=0.7)
        axes[0, 1].set_title('Churn Rate by Segment')
        axes[0, 1].set_xlabel('Segment')
        axes[0, 1].set_ylabel('Churn Rate (%)')
        axes[0, 1].set_ylim(0, max(churn_rates) * 1.1)

        # Plot 3: Value scores by segment
        value_scores = [self.segment_profiles[seg]['avg_value_score'] for seg in segments]

        axes[0, 2].bar(segments, value_scores, color='skyblue', alpha=0.7)
        axes[0, 2].set_title('Average Value Score by Segment')
        axes[0, 2].set_xlabel('Segment')
        axes[0, 2].set_ylabel('Value Score')

        # Plot 4: Engagement vs Value scatter (if we have the data)
        if len(segments) > 0:
            engagement_scores = [self.segment_profiles[seg]['avg_engagement'] for seg in segments]
            scatter_colors = ['red' if self.segment_profiles[seg]['churn_rate'] > 0.2 else 'green'
                              for seg in segments]

            axes[1, 0].scatter(value_scores, engagement_scores, c=scatter_colors, s=segment_sizes, alpha=0.7)
            axes[1, 0].set_title('Engagement vs Value (Size = Segment Size)')
            axes[1, 0].set_xlabel('Value Score')
            axes[1, 0].set_ylabel('Engagement Score')

            # Add segment labels
            for i, seg in enumerate(segments):
                axes[1, 0].annotate(f'S{seg}', (value_scores[i], engagement_scores[i]))

        # Plot 5: Satisfaction vs Orders
        satisfaction_scores = [self.segment_profiles[seg]['avg_satisfaction'] for seg in segments]
        order_counts = [self.segment_profiles[seg]['avg_orders'] for seg in segments]

        axes[1, 1].scatter(satisfaction_scores, order_counts, c=churn_rates, cmap='RdYlGn_r', s=100)
        axes[1, 1].set_title('Satisfaction vs Order Count')
        axes[1, 1].set_xlabel('Average Satisfaction Score')
        axes[1, 1].set_ylabel('Average Order Count')

        # Add colorbar
        cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
        cbar.set_label('Churn Rate (%)')

        # Plot 6: Feature importance for segmentation (PCA visualization)
        if self.segment_features is not None and len(self.segment_features) > 2:
            # Perform PCA for visualization
            features_scaled = self.scaler.transform(self.segment_features)
            pca_features = self.pca.fit_transform(features_scaled)

            # Color by cluster
            cluster_labels = self.kmeans.labels_
            scatter = axes[1, 2].scatter(pca_features[:, 0], pca_features[:, 1],
                                         c=cluster_labels, cmap='tab10', alpha=0.6, s=10)

            # Add cluster centers
            centers_pca = self.pca.transform(self.kmeans.cluster_centers_)
            axes[1, 2].scatter(centers_pca[:, 0], centers_pca[:, 1],
                               c='red', marker='x', s=200, linewidth=3)

            axes[1, 2].set_title('Customer Segments (PCA Projection)')
            axes[1, 2].set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%} variance)')
            axes[1, 2].set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%} variance)')

            plt.colorbar(scatter, ax=axes[1, 2])
        else:
            axes[1, 2].text(0.5, 0.5, 'PCA visualization\nnot available',
                            ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('PCA Visualization')

        plt.tight_layout()
        return fig

    def plot_cluster_validation(self, figsize=(12, 4)):
        """
        Plot cluster validation metrics.

        Parameters:
        -----------
        figsize : tuple, default=(12, 4)
            Figure size for the plots
        """
        if not self.cluster_validation_metrics:
            print("No cluster validation metrics available. Run find_optimal_clusters() first.")
            return

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        cluster_range = self.cluster_validation_metrics['cluster_range']

        # Elbow method plot
        axes[0].plot(cluster_range, self.cluster_validation_metrics['inertias'], 'bo-')
        axes[0].set_title('Elbow Method')
        axes[0].set_xlabel('Number of Clusters')
        axes[0].set_ylabel('Inertia')
        axes[0].grid(True, alpha=0.3)

        # Silhouette score plot
        axes[1].plot(cluster_range, self.cluster_validation_metrics['silhouette_scores'], 'ro-')
        axes[1].set_title('Silhouette Score')
        axes[1].set_xlabel('Number of Clusters')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].grid(True, alpha=0.3)

        # Mark optimal point
        max_idx = np.argmax(self.cluster_validation_metrics['silhouette_scores'])
        optimal_k = cluster_range[max_idx]
        axes[1].axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7)
        axes[1].annotate(f'Optimal: {optimal_k}',
                         xy=(optimal_k, self.cluster_validation_metrics['silhouette_scores'][max_idx]),
                         xytext=(optimal_k + 0.5, self.cluster_validation_metrics['silhouette_scores'][max_idx]),
                         arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))

        # Calinski-Harabasz score plot
        axes[2].plot(cluster_range, self.cluster_validation_metrics['calinski_harabasz_scores'], 'go-')
        axes[2].set_title('Calinski-Harabasz Score')
        axes[2].set_xlabel('Number of Clusters')
        axes[2].set_ylabel('CH Score')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def get_segment_summary(self):
        """
        Get segment summary as a DataFrame.

        Returns:
        --------
        pd.DataFrame
            Summary of all customer segments
        """
        if not self.segment_profiles:
            raise ValueError("Segment profiles not available. Call fit_segments() first.")

        summary_data = []
        for seg_id, profile in self.segment_profiles.items():
            summary_data.append({
                'Segment': seg_id,
                'Description': profile['description'],
                'Size': profile['size'],
                'Percentage': f"{profile['percentage']:.1f}%",
                'Churn_Rate': f"{profile['churn_rate']:.1%}",
                'Risk_Level': profile['risk_level'],
                'Value_Tier': profile['value_tier'],
                'Engagement': profile['engagement_level'],
                'Avg_Orders': f"{profile['avg_orders']:.1f}",
                'Avg_Satisfaction': f"{profile['avg_satisfaction']:.1f}",
                'Top_Category': profile['top_category'],
                'Value_Score': f"{profile['avg_value_score']:.2f}"
            })

        return pd.DataFrame(summary_data)

    def save_model(self, file_path='models/customer_segmentation_model.pkl'):
        """
        Save the trained segmentation model.

        Parameters:
        -----------
        file_path : str
            Path where to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first. Call fit_segments().")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Prepare model data
        model_data = {
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'pca': self.pca,
            'segment_profiles': self.segment_profiles,
            'feature_names': self.feature_names,
            'n_clusters': self.n_clusters,
            'cluster_validation_metrics': self.cluster_validation_metrics
        }

        # Save model
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Segmentation model saved to: {file_path}")

    def load_model(self, file_path):
        """
        Load a trained segmentation model.

        Parameters:
        -----------
        file_path : str
            Path to the saved model
        """
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)

            # Restore model components
            self.kmeans = model_data['kmeans']
            self.scaler = model_data['scaler']
            self.pca = model_data['pca']
            self.segment_profiles = model_data['segment_profiles']
            self.feature_names = model_data['feature_names']
            self.n_clusters = model_data['n_clusters']
            self.cluster_validation_metrics = model_data.get('cluster_validation_metrics', {})

            self.is_fitted = True
            print(f"Segmentation model loaded from: {file_path}")

        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

    def generate_segment_insights(self):
        """
        Generate business insights for each segment.

        Returns:
        --------
        dict
            Business insights and recommendations for each segment
        """
        if not self.segment_profiles:
            raise ValueError("Segment profiles not available. Call fit_segments() first.")

        insights = {}

        for seg_id, profile in self.segment_profiles.items():
            segment_insights = {
                'segment_id': seg_id,
                'description': profile['description'],
                'key_characteristics': [],
                'business_opportunities': [],
                'retention_strategies': [],
                'growth_potential': 'Low'
            }

            # Analyze characteristics
            if profile['churn_rate'] > 0.25:
                segment_insights['key_characteristics'].append(f"High churn risk ({profile['churn_rate']:.1%})")
                segment_insights['retention_strategies'].extend([
                    "Implement immediate retention campaigns",
                    "Provide personalized customer support",
                    "Offer targeted discounts and incentives"
                ])

            if profile['avg_value_score'] > 2.0:
                segment_insights['key_characteristics'].append("High-value customers")
                segment_insights['business_opportunities'].extend([
                    "Premium service offerings",
                    "Exclusive product access",
                    "VIP loyalty program"
                ])
                segment_insights['growth_potential'] = 'High'

            if profile['avg_engagement'] > 3.0:
                segment_insights['key_characteristics'].append("Highly engaged")
                segment_insights['business_opportunities'].extend([
                    "Beta product testing",
                    "Community building initiatives",
                    "Referral program opportunities"
                ])

            if profile['complaint_rate'] > 0.3:
                segment_insights['key_characteristics'].append("High complaint rate")
                segment_insights['retention_strategies'].extend([
                    "Proactive customer service",
                    "Process improvement focus",
                    "Satisfaction monitoring"
                ])

            # Category-specific insights
            if profile['top_category'] in ['Mobile', 'Mobile Phone']:
                segment_insights['business_opportunities'].extend([
                    "Technology-focused content marketing",
                    "Trade-in programs",
                    "Extended warranty offerings"
                ])
            elif profile['top_category'] == 'Grocery':
                segment_insights['business_opportunities'].extend([
                    "Subscription services",
                    "Bulk purchase discounts",
                    "Fresh product guarantees"
                ])

            insights[seg_id] = segment_insights

        return insights


def demonstrate_segmentation():
    """Demonstration function for the segmentation module."""
    print("Customer Segmentation Module Demo")
    print("=" * 50)

    # Create sample data for demonstration
    np.random.seed(42)
    n_customers = 1000

    sample_data = pd.DataFrame({
        'CustomerID': range(1, n_customers + 1),
        'Tenure': np.random.exponential(12, n_customers),
        'HourSpendOnApp': np.random.gamma(2, 2, n_customers),
        'OrderCount': np.random.poisson(3, n_customers),
        'SatisfactionScore': np.random.randint(1, 6, n_customers),
        'CashbackAmount': np.random.normal(120, 30, n_customers),
        'CustomerValueScore': np.random.normal(1.5, 0.8, n_customers),
        'EngagementScore': np.random.normal(2.0, 1.0, n_customers),
        'Churn': np.random.binomial(1, 0.17, n_customers)
    })

    # Add remaining required columns with reasonable defaults
    required_cols = ['NumberOfDeviceRegistered', 'NumberOfAddress', 'WarehouseToHome',
                     'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderFrequency',
                     'CashbackEfficiency', 'ServiceQualityScore', 'PreferedOrderCat',
                     'Gender', 'PreferredLoginDevice', 'PreferredPaymentMode', 'Complain']

    for col in required_cols:
        if col not in sample_data.columns:
            if col == 'PreferedOrderCat':
                sample_data[col] = np.random.choice(['Mobile', 'Laptop & Accessory', 'Fashion', 'Grocery'], n_customers)
            elif col == 'Gender':
                sample_data[col] = np.random.choice(['Male', 'Female'], n_customers)
            elif col == 'PreferredLoginDevice':
                sample_data[col] = np.random.choice(['Mobile Phone', 'Computer', 'Phone'], n_customers)
            elif col == 'PreferredPaymentMode':
                sample_data[col] = np.random.choice(['Credit Card', 'Debit Card', 'UPI'], n_customers)
            elif col == 'Complain':
                sample_data[col] = np.random.binomial(1, 0.2, n_customers)
            else:
                sample_data[col] = np.random.normal(10, 3, n_customers)

    try:
        # Initialize and fit segmentation
        segmentation = CustomerSegmentation(n_clusters=4)

        # Find optimal clusters (optional)
        metrics = segmentation.find_optimal_clusters(sample_data, max_clusters=8)

        # Fit the model
        segmented_data = segmentation.fit_segments(sample_data)

        # Display results
        print(f"\n Segmentation completed successfully!")
        print(f" Created {segmentation.n_clusters} customer segments")

        # Show segment summary
        summary = segmentation.get_segment_summary()
        print("\n Segment Summary:")
        print(summary.to_string(index=False))

        # Generate insights
        insights = segmentation.generate_segment_insights()
        print(f"\n Generated insights for {len(insights)} segments")

        return segmentation, segmented_data

    except Exception as e:
        print(f" Demo failed: {str(e)}")
        return None, None


if __name__ == "__main__":
    # Run demonstration
    model, data = demonstrate_segmentation()

    if model is not None:
        print(f"\n Customer segmentation module ready for use!")
        print(f" Model fitted with {model.n_clusters} segments")
        print(f" {len(data)} customers segmented")
