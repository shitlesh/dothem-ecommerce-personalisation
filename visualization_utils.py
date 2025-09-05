"""
Dohtem Visualization Utilities Module
====================================

This module provides comprehensive visualization capabilities for the entire
personalization analysis pipeline, creating publication-ready charts and dashboards.

Author: ML Engineering Team
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def create_comprehensive_visualizations(raw_data, processed_data, segmentation_model,
                                        churn_model, business_impact, strategies,
                                        save_path='visualizations/'):
    """
    Create comprehensive visualization suite for the entire analysis.

    Parameters:
    -----------
    raw_data : pd.DataFrame
        Original raw customer data
    processed_data : pd.DataFrame
        Processed data with segments
    segmentation_model : CustomerSegmentation
        Trained segmentation model
    churn_model : ChurnPredictor
        Trained churn prediction model
    business_impact : dict
        Business impact analysis results
    strategies : dict
        Personalization strategies
    save_path : str, default='visualizations/'
        Path to save visualization files

    Returns:
    --------
    dict
        Dictionary of created visualization files
    """
    print(f"📈 Creating comprehensive visualization suite...")

    # Create save directory
    os.makedirs(save_path, exist_ok=True)

    created_files = {}

    # 1. Data Exploration Visualizations
    print(f"   📊 Creating data exploration visualizations...")
    data_viz = create_data_exploration_visualizations(raw_data, processed_data, save_path)
    created_files.update(data_viz)

    # 2. Customer Segmentation Visualizations
    print(f"   👥 Creating segmentation visualizations...")
    segment_viz = create_segmentation_visualizations(processed_data, segmentation_model, save_path)
    created_files.update(segment_viz)

    # 3. Churn Analysis Visualizations
    print(f"   ⚠️  Creating churn analysis visualizations...")
    churn_viz = create_churn_analysis_visualizations(processed_data, churn_model, save_path)
    created_files.update(churn_viz)

    # 4. Business Impact Visualizations
    print(f"   💰 Creating business impact visualizations...")
    impact_viz = create_business_impact_visualizations(business_impact, save_path)
    created_files.update(impact_viz)

    # 5. Personalization Strategy Visualizations
    print(f"   🎯 Creating strategy visualizations...")
    strategy_viz = create_strategy_visualizations(strategies, save_path)
    created_files.update(strategy_viz)

    # 6. Executive Dashboard
    print(f"   📋 Creating executive dashboard...")
    dashboard = create_executive_dashboard(processed_data, business_impact, strategies, save_path)
    created_files.update(dashboard)

    print(f"✅ Created {len(created_files)} visualization files in '{save_path}'")
    return created_files


def create_data_exploration_visualizations(raw_data, processed_data, save_path):
    """Create data exploration and quality visualizations."""

    created_files = {}

    # 1. Data Quality Overview
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Data Quality and Exploration Overview', fontsize=16, y=0.98)

    # Missing data heatmap
    missing_data = raw_data.isnull().sum()
    missing_percent = (missing_data / len(raw_data)) * 100
    missing_df = pd.DataFrame({
        'Feature': missing_data.index,
        'Missing_Percent': missing_percent.values
    }).sort_values('Missing_Percent', ascending=False)

    missing_with_data = missing_df[missing_df['Missing_Percent'] > 0]
    if not missing_with_data.empty:
        axes[0, 0].barh(missing_with_data['Feature'], missing_with_data['Missing_Percent'])
        axes[0, 0].set_title('Missing Data by Feature (%)')
        axes[0, 0].set_xlabel('Missing Percentage')
    else:
        axes[0, 0].text(0.5, 0.5, 'No Missing Data', ha='center', va='center',
                        transform=axes[0, 0].transAxes, fontsize=14)
        axes[0, 0].set_title('Missing Data Analysis')

    # Churn distribution
    churn_counts = raw_data['Churn'].value_counts()
    axes[0, 1].pie(churn_counts.values, labels=['Retained', 'Churned'],
                   autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral'])
    axes[0, 1].set_title(f'Churn Distribution\n({churn_counts[1]:,} / {len(raw_data):,} churned)')

    # Customer satisfaction distribution
    if 'SatisfactionScore' in raw_data.columns:
        satisfaction_counts = raw_data['SatisfactionScore'].value_counts().sort_index()
        axes[0, 2].bar(satisfaction_counts.index, satisfaction_counts.values,
                       color='skyblue', alpha=0.7)
        axes[0, 2].set_title('Customer Satisfaction Distribution')
        axes[0, 2].set_xlabel('Satisfaction Score')
        axes[0, 2].set_ylabel('Number of Customers')

    # Category preferences
    if 'PreferedOrderCat' in raw_data.columns:
        category_counts = raw_data['PreferedOrderCat'].value_counts()
        axes[1, 0].barh(category_counts.index, category_counts.values, color='coral', alpha=0.7)
        axes[1, 0].set_title('Product Category Preferences')
        axes[1, 0].set_xlabel('Number of Customers')

    # Numerical features distribution
    numerical_cols = ['Tenure', 'HourSpendOnApp', 'OrderCount', 'CashbackAmount']
    available_numerical = [col for col in numerical_cols if col in raw_data.columns]

    if available_numerical:
        sample_col = available_numerical[0]
        axes[1, 1].hist(raw_data[sample_col].dropna(), bins=30, alpha=0.7, color='lightblue')
        axes[1, 1].set_title(f'{sample_col} Distribution')
        axes[1, 1].set_xlabel(sample_col)
        axes[1, 1].set_ylabel('Frequency')

    # Churn by category
    if 'PreferedOrderCat' in raw_data.columns:
        churn_by_category = raw_data.groupby('PreferedOrderCat')['Churn'].agg(['count', 'mean']).reset_index()
        churn_by_category['churn_rate'] = churn_by_category['mean'] * 100

        axes[1, 2].bar(range(len(churn_by_category)), churn_by_category['churn_rate'],
                       color='orange', alpha=0.7)
        axes[1, 2].set_xticks(range(len(churn_by_category)))
        axes[1, 2].set_xticklabels(churn_by_category['PreferedOrderCat'], rotation=45, ha='right')
        axes[1, 2].set_title('Churn Rate by Product Category (%)')
        axes[1, 2].set_ylabel('Churn Rate (%)')

    plt.tight_layout()
    file_path = os.path.join(save_path, 'data_quality_overview.png')
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    created_files['data_quality_overview'] = file_path

    # 2. Feature Engineering Impact
    if 'CustomerValueScore' in processed_data.columns:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Feature Engineering Impact Analysis', fontsize=16, y=0.98)

        # Customer Value Score distribution
        axes[0, 0].hist(processed_data['CustomerValueScore'], bins=30, alpha=0.7, color='gold')
        axes[0, 0].set_title('Customer Value Score Distribution')
        axes[0, 0].set_xlabel('Value Score')
        axes[0, 0].set_ylabel('Frequency')

        # Value Score vs Churn
        churned = processed_data[processed_data['Churn'] == 1]['CustomerValueScore']
        retained = processed_data[processed_data['Churn'] == 0]['CustomerValueScore']

        axes[0, 1].boxplot([retained.dropna(), churned.dropna()],
                           labels=['Retained', 'Churned'])
        axes[0, 1].set_title('Customer Value Score by Churn Status')
        axes[0, 1].set_ylabel('Value Score')

        # Engagement Score analysis
        if 'EngagementScore' in processed_data.columns:
            axes[1, 0].scatter(processed_data['CustomerValueScore'],
                               processed_data['EngagementScore'],
                               c=processed_data['Churn'], cmap='RdYlGn_r', alpha=0.6)
            axes[1, 0].set_title('Value vs Engagement (Color = Churn)')
            axes[1, 0].set_xlabel('Customer Value Score')
            axes[1, 0].set_ylabel('Engagement Score')

        # Risk factors impact
        risk_features = ['HighRiskCategory', 'HighRiskPayment', 'LowSatisfaction']
        available_risk = [f for f in risk_features if f in processed_data.columns]

        if available_risk:
            risk_impact = []
            risk_labels = []

            for feature in available_risk:
                churn_rate_with = processed_data[processed_data[feature] == 1]['Churn'].mean()
                churn_rate_without = processed_data[processed_data[feature] == 0]['Churn'].mean()
                impact = (churn_rate_with - churn_rate_without) * 100
                risk_impact.append(impact)
                risk_labels.append(feature.replace('HighRisk', '').replace('Low', ''))

            axes[1, 1].bar(risk_labels, risk_impact, color='red', alpha=0.7)
            axes[1, 1].set_title('Risk Factor Impact on Churn (%)')
            axes[1, 1].set_ylabel('Churn Rate Increase (%)')
            axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        file_path = os.path.join(save_path, 'feature_engineering_impact.png')
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        created_files['feature_engineering_impact'] = file_path

    return created_files


def create_segmentation_visualizations(data, segmentation_model, save_path):
    """Create customer segmentation visualizations."""

    created_files = {}

    # 1. Segment Overview
    if hasattr(segmentation_model, 'segment_profiles'):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Customer Segmentation Analysis', fontsize=16, y=0.98)

        profiles = segmentation_model.segment_profiles
        segments = list(profiles.keys())

        # Segment sizes
        sizes = [profiles[s]['size'] for s in segments]
        labels = [f"Segment {s}\n({profiles[s]['size']:,})" for s in segments]

        axes[0, 0].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Segment Size Distribution')

        # Churn rates by segment
        churn_rates = [profiles[s]['churn_rate'] * 100 for s in segments]
        colors = ['red' if rate > 25 else 'orange' if rate > 15 else 'green' for rate in churn_rates]

        axes[0, 1].bar(segments, churn_rates, color=colors, alpha=0.7)
        axes[0, 1].set_title('Churn Rate by Segment (%)')
        axes[0, 1].set_xlabel('Segment ID')
        axes[0, 1].set_ylabel('Churn Rate (%)')

        # Value scores by segment
        value_scores = [profiles[s]['avg_value_score'] for s in segments]

        axes[0, 2].bar(segments, value_scores, color='skyblue', alpha=0.7)
        axes[0, 2].set_title('Average Value Score by Segment')
        axes[0, 2].set_xlabel('Segment ID')
        axes[0, 2].set_ylabel('Value Score')

        # Satisfaction vs Orders scatter
        satisfaction_scores = [profiles[s]['avg_satisfaction'] for s in segments]
        order_counts = [profiles[s]['avg_orders'] for s in segments]

        scatter = axes[1, 0].scatter(satisfaction_scores, order_counts,
                                     c=churn_rates, cmap='RdYlGn_r', s=sizes, alpha=0.7)
        axes[1, 0].set_title('Satisfaction vs Orders\n(Size=Segment Size, Color=Churn Rate)')
        axes[1, 0].set_xlabel('Average Satisfaction Score')
        axes[1, 0].set_ylabel('Average Order Count')

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=axes[1, 0])
        cbar.set_label('Churn Rate (%)')

        # Top categories by segment
        categories = [profiles[s]['top_category'] for s in segments]
        category_counts = pd.Series(categories).value_counts()

        axes[1, 1].pie(category_counts.values, labels=category_counts.index,
                       autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Dominant Categories Across Segments')

        # Segment characteristics radar-like comparison
        metrics = ['avg_satisfaction', 'avg_engagement', 'avg_value_score']
        available_metrics = [m for m in metrics if m in profiles[segments[0]]]

        if available_metrics and len(segments) <= 5:  # Limit to 5 segments for readability
            x_pos = np.arange(len(available_metrics))
            width = 0.15

            for i, segment in enumerate(segments[:5]):
                values = [profiles[segment][metric] for metric in available_metrics]
                axes[1, 2].bar(x_pos + i * width, values, width,
                               label=f'Segment {segment}', alpha=0.7)

            axes[1, 2].set_xlabel('Metrics')
            axes[1, 2].set_ylabel('Average Values')
            axes[1, 2].set_title('Segment Characteristics Comparison')
            axes[1, 2].set_xticks(x_pos + width * 2)
            axes[1, 2].set_xticklabels([m.replace('avg_', '').title() for m in available_metrics])
            axes[1, 2].legend()

        plt.tight_layout()
        file_path = os.path.join(save_path, 'segmentation_overview.png')
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        created_files['segmentation_overview'] = file_path

    # 2. Segment Profiles Heatmap
    if hasattr(segmentation_model, 'segment_profiles'):
        # Create heatmap data
        profiles = segmentation_model.segment_profiles
        segments = list(profiles.keys())

        heatmap_data = []
        metrics = ['churn_rate', 'avg_satisfaction', 'avg_orders', 'avg_value_score',
                   'avg_engagement', 'avg_cashback']
        available_metrics = [m for m in metrics if m in profiles[segments[0]]]

        for segment in segments:
            row = [profiles[segment].get(metric, 0) for metric in available_metrics]
            heatmap_data.append(row)

        if heatmap_data:
            # Normalize data for better visualization
            heatmap_df = pd.DataFrame(heatmap_data,
                                      index=[f'Segment {s}' for s in segments],
                                      columns=[m.replace('avg_', '').title().replace('_', ' ') for m in
                                               available_metrics])

            # Normalize each column to 0-1 scale for comparison
            heatmap_normalized = heatmap_df.div(heatmap_df.max())

            plt.figure(figsize=(12, 8))
            sns.heatmap(heatmap_normalized, annot=True, fmt='.2f', cmap='RdYlGn',
                        cbar_kws={'label': 'Normalized Value'})
            plt.title('Customer Segment Characteristics Heatmap\n(Normalized Values)', fontsize=14)
            plt.tight_layout()

            file_path = os.path.join(save_path, 'segment_profiles_heatmap.png')
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            created_files['segment_profiles_heatmap'] = file_path

    return created_files


def create_churn_analysis_visualizations(data, churn_model, save_path):
    """Create churn analysis and model performance visualizations."""

    created_files = {}

    # 1. Model Performance Overview
    if hasattr(churn_model, 'performance_metrics') and churn_model.performance_metrics:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Churn Prediction Model Performance', fontsize=16, y=0.98)

        # Confusion Matrix
        cm = churn_model.performance_metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')

        # ROC Curve
        fpr, tpr, _ = churn_model.performance_metrics['roc_curve']
        auc = churn_model.performance_metrics['test_auc']

        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2,
                        label=f'ROC Curve (AUC = {auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend(loc="lower right")
        axes[0, 1].grid(True, alpha=0.3)

        # Precision-Recall Curve
        precision, recall, _ = churn_model.performance_metrics['pr_curve']

        axes[1, 0].plot(recall, precision, color='blue', lw=2)
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision-Recall Curve')
        axes[1, 0].grid(True, alpha=0.3)

        # Model Metrics Comparison
        metrics_names = ['AUC', 'F1', 'Precision', 'Recall']
        test_scores = [
            churn_model.performance_metrics['test_auc'],
            churn_model.performance_metrics['test_f1'],
            churn_model.performance_metrics['test_precision'],
            churn_model.performance_metrics['test_recall']
        ]

        bars = axes[1, 1].bar(metrics_names, test_scores,
                              color=['skyblue', 'lightgreen', 'coral', 'gold'])
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Model Performance Metrics')

        # Add value labels on bars
        for bar, score in zip(bars, test_scores):
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f'{score:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        file_path = os.path.join(save_path, 'churn_model_performance.png')
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        created_files['churn_model_performance'] = file_path

    # 2. Feature Importance Analysis
    if hasattr(churn_model, 'feature_importance') and churn_model.feature_importance is not None:
        plt.figure(figsize=(12, 10))

        top_features = churn_model.feature_importance.head(20)

        plt.barh(range(len(top_features)), top_features['importance'], color='coral', alpha=0.7)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Most Important Features for Churn Prediction', fontsize=14)
        plt.gca().invert_yaxis()
        plt.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()

        file_path = os.path.join(save_path, 'feature_importance_churn.png')
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        created_files['feature_importance_churn'] = file_path

    # 3. Churn Risk Distribution
    if 'CustomerSegment' in data.columns:
        # Predict churn risk for all customers
        try:
            churn_probs, risk_categories = churn_model.predict_churn_risk(data)

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Churn Risk Analysis', fontsize=16, y=0.98)

            # Risk distribution
            risk_counts = pd.Series(risk_categories).value_counts()
            axes[0, 0].pie(risk_counts.values, labels=risk_counts.index,
                           autopct='%1.1f%%', startangle=90)
            axes[0, 0].set_title('Customer Risk Distribution')

            # Risk by segment
            risk_by_segment = pd.DataFrame({
                'Segment': data['CustomerSegment'],
                'Risk': risk_categories
            }).groupby(['Segment', 'Risk']).size().unstack(fill_value=0)

            risk_by_segment_pct = risk_by_segment.div(risk_by_segment.sum(axis=1), axis=0) * 100
            risk_by_segment_pct.plot(kind='bar', stacked=True, ax=axes[0, 1],
                                     color=['green', 'yellow', 'orange', 'red'])
            axes[0, 1].set_title('Risk Distribution by Segment (%)')
            axes[0, 1].set_xlabel('Segment')
            axes[0, 1].set_ylabel('Percentage')
            axes[0, 1].legend(title='Risk Level')

            # Churn probability distribution
            axes[1, 0].hist(churn_probs, bins=30, alpha=0.7, color='purple')
            axes[1, 0].axvline(x=np.mean(churn_probs), color='red', linestyle='--',
                               label=f'Mean: {np.mean(churn_probs):.3f}')
            axes[1, 0].set_title('Churn Probability Distribution')
            axes[1, 0].set_xlabel('Churn Probability')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()

            # High-risk customers by category
            if 'PreferedOrderCat' in data.columns:
                high_risk_data = data[pd.Series(risk_categories) == 'High Risk']
                category_risk = high_risk_data['PreferedOrderCat'].value_counts()

                axes[1, 1].bar(range(len(category_risk)), category_risk.values,
                               color='red', alpha=0.7)
                axes[1, 1].set_xticks(range(len(category_risk)))
                axes[1, 1].set_xticklabels(category_risk.index, rotation=45, ha='right')
                axes[1, 1].set_title('High-Risk Customers by Category')
                axes[1, 1].set_ylabel('Number of Customers')

            plt.tight_layout()
            file_path = os.path.join(save_path, 'churn_risk_analysis.png')
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            created_files['churn_risk_analysis'] = file_path

        except Exception as e:
            print(f"   ⚠️  Could not create churn risk analysis: {str(e)}")

    return created_files


def create_business_impact_visualizations(business_impact, save_path):
    """Create business impact and ROI visualizations."""

    created_files = {}

    if not business_impact:
        return created_files

    # 1. Financial Impact Overview
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Business Impact Analysis', fontsize=16, y=0.98)

    # Revenue components breakdown
    if 'segment_impacts' in business_impact:
        segments = list(business_impact['segment_impacts'].keys())
        retention_revenue = [business_impact['segment_impacts'][s].get('retention_revenue', 0)
                             for s in segments]
        value_revenue = [business_impact['segment_impacts'][s].get('value_increase_revenue', 0)
                         for s in segments]
        engagement_revenue = [business_impact['segment_impacts'][s].get('engagement_revenue', 0)
                              for s in segments]

        # Stacked bar chart of revenue sources
        width = 0.6
        axes[0, 0].bar(segments, [r / 1000 for r in retention_revenue], width,
                       label='Retention', color='lightgreen')
        axes[0, 0].bar(segments, [v / 1000 for v in value_revenue], width,
                       bottom=[r / 1000 for r in retention_revenue],
                       label='Value Increase', color='skyblue')
        axes[0, 0].bar(segments, [e / 1000 for e in engagement_revenue], width,
                       bottom=[(r + v) / 1000 for r, v in zip(retention_revenue, value_revenue)],
                       label='Engagement', color='coral')

        axes[0, 0].set_title('Revenue Impact by Source ($K)')
        axes[0, 0].set_xlabel('Segment')
        axes[0, 0].set_ylabel('Revenue Impact ($K)')
        axes[0, 0].legend()

    # Overall financial metrics
    financial_metrics = [
        ('Total Revenue Impact', business_impact.get('total_revenue_impact', 0)),
        ('Implementation Cost', business_impact.get('total_implementation_cost', 0)),
        ('Operational Cost', business_impact.get('total_operational_cost', 0)),
        ('Net Benefit', business_impact.get('net_benefit', 0))
    ]

    metrics, values = zip(*financial_metrics)
    colors = ['green', 'red', 'orange', 'blue']

    axes[0, 1].bar(metrics, [v / 1000 for v in values], color=colors, alpha=0.7)
    axes[0, 1].set_title('Financial Summary ($K)')
    axes[0, 1].set_ylabel('Amount ($K)')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # ROI and Payback metrics
    roi = business_impact.get('roi_percentage', 0)
    payback = business_impact.get('payback_period_months', 0)

    # ROI gauge chart
    roi_categories = ['Poor (<50%)', 'Fair (50-100%)', 'Good (100-200%)', 'Excellent (>200%)']
    roi_colors = ['red', 'orange', 'yellow', 'green']
    roi_category_idx = 0 if roi < 50 else 1 if roi < 100 else 2 if roi < 200 else 3

    axes[0, 2].pie([roi, max(0, 400 - roi)], colors=[roi_colors[roi_category_idx], 'lightgray'],
                   startangle=90, counterclock=False, wedgeprops=dict(width=0.5))
    axes[0, 2].text(0, 0, f'{roi:.0f}%\nROI', ha='center', va='center', fontsize=14, weight='bold')
    axes[0, 2].set_title('Return on Investment')

    # Customer retention impact
    customers_retained = business_impact.get('total_customers_retained', 0)
    total_customers = business_impact.get('total_customers', 1)
    retention_rate = customers_retained / total_customers * 100 if total_customers > 0 else 0

    axes[1, 0].bar(['Current Churn', 'After Intervention'],
                   [business_impact.get('current_churn_rate', 0) * 100,
                    business_impact.get('new_churn_rate', 0) * 100],
                   color=['red', 'green'], alpha=0.7)
    axes[1, 0].set_title('Churn Rate Improvement')
    axes[1, 0].set_ylabel('Churn Rate (%)')

    # Payback timeline
    if payback < 24:  # Only show if reasonable payback period
        months = list(range(1, int(payback) + 7))
        cumulative_investment = [business_impact.get('total_costs', 0)] * len(months)
        monthly_revenue = business_impact.get('total_revenue_impact', 0) / 12
        cumulative_revenue = [monthly_revenue * m for m in months]

        axes[1, 1].plot(months, [c / 1000 for c in cumulative_investment],
                        label='Cumulative Investment', color='red', linestyle='--')
        axes[1, 1].plot(months, [r / 1000 for r in cumulative_revenue],
                        label='Cumulative Revenue', color='green')
        axes[1, 1].axvline(x=payback, color='blue', linestyle=':',
                           label=f'Break-even: Month {payback:.1f}')
        axes[1, 1].set_title('Payback Analysis')
        axes[1, 1].set_xlabel('Months')
        axes[1, 1].set_ylabel('Amount ($K)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    # Segment ROI comparison
    if 'segment_impacts' in business_impact:
        segment_rois = [business_impact['segment_impacts'][s].get('roi_percentage', 0)
                        for s in segments]
        segment_colors = ['green' if roi > 200 else 'orange' if roi > 100 else 'red'
                          for roi in segment_rois]

        axes[1, 2].bar(segments, segment_rois, color=segment_colors, alpha=0.7)
        axes[1, 2].set_title('ROI by Segment (%)')
        axes[1, 2].set_xlabel('Segment')
        axes[1, 2].set_ylabel('ROI (%)')
        axes[1, 2].axhline(y=100, color='black', linestyle='--', alpha=0.5, label='Break-even')
        axes[1, 2].legend()

    plt.tight_layout()
    file_path = os.path.join(save_path, 'business_impact_overview.png')
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    created_files['business_impact_overview'] = file_path

    return created_files


def create_strategy_visualizations(strategies, save_path):
    """Create personalization strategy visualizations."""

    created_files = {}

    if not strategies:
        return created_files

    # 1. Strategy Overview
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Personalization Strategies Overview', fontsize=16, y=0.98)

    # Strategy type distribution
    strategy_types = [s['strategy_type'] for s in strategies.values()]
    strategy_counts = pd.Series(strategy_types).value_counts()

    axes[0, 0].pie(strategy_counts.values, labels=strategy_counts.index,
                   autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Strategy Type Distribution')

    # Priority levels
    priorities = [s['priority_level'] for s in strategies.values()]
    priority_counts = pd.Series(priorities).value_counts()
    priority_order = ['Critical', 'High', 'Medium-High', 'Medium']
    ordered_priorities = [priority_counts.get(p, 0) for p in priority_order]
    priority_colors = ['red', 'orange', 'yellow', 'lightblue']

    axes[0, 1].bar(priority_order, ordered_priorities, color=priority_colors, alpha=0.7)
    axes[0, 1].set_title('Strategy Priority Distribution')
    axes[0, 1].set_xlabel('Priority Level')
    axes[0, 1].set_ylabel('Number of Strategies')

    # Customer coverage by strategy
    strategy_customers = {}
    for strategy in strategies.values():
        strategy_type = strategy['strategy_type']
        if strategy_type not in strategy_customers:
            strategy_customers[strategy_type] = 0
        strategy_customers[strategy_type] += strategy.get('segment_size', 0)

    axes[0, 2].barh(list(strategy_customers.keys()),
                    list(strategy_customers.values()), color='coral', alpha=0.7)
    axes[0, 2].set_title('Customer Coverage by Strategy Type')
    axes[0, 2].set_xlabel('Number of Customers')

    # Risk vs Value matrix
    segments = list(strategies.keys())
    risk_levels = [strategies[s].get('churn_risk_level', 'Medium') for s in segments]
    value_tiers = [strategies[s].get('value_tier', 'Standard') for s in segments]
    segment_sizes = [strategies[s].get('segment_size', 0) for s in segments]

    # Convert categorical to numerical for plotting
    risk_mapping = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
    value_mapping = {'Basic': 1, 'Standard': 2, 'High': 3, 'Premium': 4}

    risk_numeric = [risk_mapping.get(r, 2) for r in risk_levels]
    value_numeric = [value_mapping.get(v, 2) for v in value_tiers]

    scatter = axes[1, 0].scatter(risk_numeric, value_numeric, s=segment_sizes,
                                 alpha=0.6, c=range(len(segments)), cmap='tab10')
    axes[1, 0].set_xlabel('Risk Level')
    axes[1, 0].set_ylabel('Value Tier')
    axes[1, 0].set_title('Risk vs Value Matrix\n(Bubble size = Segment size)')
    axes[1, 0].set_xticks(range(1, 5))
    axes[1, 0].set_xticklabels(['Low', 'Medium', 'High', 'Critical'])
    axes[1, 0].set_yticks(range(1, 5))
    axes[1, 0].set_yticklabels(['Basic', 'Standard', 'High', 'Premium'])

    # Add segment labels
    for i, seg in enumerate(segments):
        axes[1, 0].annotate(f'S{seg}', (risk_numeric[i], value_numeric[i]))

    # Expected impact by strategy
    if 'expected_impact' in list(strategies.values())[0]:
        impacts = [s['expected_impact'].get('revenue_impact', 0) / 1000 for s in strategies.values()]

        axes[1, 1].bar(segments, impacts, color='gold', alpha=0.7)
        axes[1, 1].set_title('Expected Revenue Impact by Segment ($K)')
        axes[1, 1].set_xlabel('Segment ID')
        axes[1, 1].set_ylabel('Revenue Impact ($K)')

    # Implementation timeline
    timelines = [s.get('implementation_timeline', 'Unknown') for s in strategies.values()]
    timeline_counts = pd.Series(timelines).value_counts()

    axes[1, 2].barh(range(len(timeline_counts)), timeline_counts.values,
                    color='lightgreen', alpha=0.7)
    axes[1, 2].set_yticks(range(len(timeline_counts)))
    axes[1, 2].set_yticklabels(timeline_counts.index)
    axes[1, 2].set_title('Implementation Timeline Distribution')
    axes[1, 2].set_xlabel('Number of Segments')

    plt.tight_layout()
    file_path = os.path.join(save_path, 'strategy_overview.png')
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    created_files['strategy_overview'] = file_path

    return created_files


def create_executive_dashboard(data, business_impact, strategies, save_path):
    """Create executive dashboard with key metrics."""

    created_files = {}

    # 1. Executive Summary Dashboard
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('DOHTEM PERSONALIZATION INITIATIVE - EXECUTIVE DASHBOARD',
                 fontsize=20, y=0.98, weight='bold')

    # Create custom grid layout
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)

    # Key metrics (top row)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])

    # Create metric cards
    total_customers = len(data) if data is not None else 0
    total_revenue = business_impact.get('total_revenue_impact', 0) if business_impact else 0
    roi = business_impact.get('roi_percentage', 0) if business_impact else 0
    payback = business_impact.get('payback_period_months', 0) if business_impact else 0

    # Metric card 1: Total Customers
    ax1.text(0.5, 0.7, f'{total_customers:,}', ha='center', va='center',
             fontsize=24, weight='bold', transform=ax1.transAxes)
    ax1.text(0.5, 0.3, 'Total Customers', ha='center', va='center',
             fontsize=12, transform=ax1.transAxes)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='blue', lw=2))

    # Metric card 2: Revenue Impact
    ax2.text(0.5, 0.7, f'${total_revenue / 1000:.0f}K', ha='center', va='center',
             fontsize=24, weight='bold', transform=ax2.transAxes, color='green')
    ax2.text(0.5, 0.3, 'Annual Revenue Impact', ha='center', va='center',
             fontsize=12, transform=ax2.transAxes)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='green', lw=2))

    # Metric card 3: ROI
    roi_color = 'green' if roi > 200 else 'orange' if roi > 100 else 'red'
    ax3.text(0.5, 0.7, f'{roi:.0f}%', ha='center', va='center',
             fontsize=24, weight='bold', transform=ax3.transAxes, color=roi_color)
    ax3.text(0.5, 0.3, 'Return on Investment', ha='center', va='center',
             fontsize=12, transform=ax3.transAxes)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor=roi_color, lw=2))

    # Metric card 4: Payback Period
    payback_color = 'green' if payback < 12 else 'orange' if payback < 24 else 'red'
    ax4.text(0.5, 0.7, f'{payback:.1f}', ha='center', va='center',
             fontsize=24, weight='bold', transform=ax4.transAxes, color=payback_color)
    ax4.text(0.5, 0.3, 'Payback (Months)', ha='center', va='center',
             fontsize=12, transform=ax4.transAxes)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor=payback_color, lw=2))

    # Second row: Customer segments and churn analysis
    ax5 = fig.add_subplot(gs[1, :2])  # Segment distribution
    ax6 = fig.add_subplot(gs[1, 2:])  # Churn by category

    # Segment distribution
    if 'CustomerSegment' in data.columns:
        segment_counts = data['CustomerSegment'].value_counts().sort_index()
        churn_by_segment = data.groupby('CustomerSegment')['Churn'].mean()

        bars = ax5.bar(segment_counts.index, segment_counts.values,
                       color=[
                           'red' if churn_by_segment[i] > 0.2 else 'orange' if churn_by_segment[i] > 0.15 else 'green'
                           for i in segment_counts.index], alpha=0.7)
        ax5.set_title('Customer Segments (Color = Risk Level)', fontsize=14, weight='bold')
        ax5.set_xlabel('Segment ID')
        ax5.set_ylabel('Number of Customers')

        # Add churn rate labels
        for bar, segment in zip(bars, segment_counts.index):
            churn_rate = churn_by_segment[segment]
            ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                     f'{churn_rate:.1%}', ha='center', va='bottom', fontsize=10)

    # Churn by category
    if 'PreferedOrderCat' in data.columns:
        churn_by_cat = data.groupby('PreferedOrderCat')['Churn'].agg(['count', 'mean']).reset_index()
        churn_by_cat['churn_rate'] = churn_by_cat['mean'] * 100
        churn_by_cat = churn_by_cat.sort_values('churn_rate', ascending=True)

        colors = ['green' if rate < 10 else 'orange' if rate < 20 else 'red'
                  for rate in churn_by_cat['churn_rate']]

        bars = ax6.barh(churn_by_cat['PreferedOrderCat'], churn_by_cat['churn_rate'],
                        color=colors, alpha=0.7)
        ax6.set_title('Churn Rate by Product Category', fontsize=14, weight='bold')
        ax6.set_xlabel('Churn Rate (%)')

        # Add customer count labels
        for i, (bar, count) in enumerate(zip(bars, churn_by_cat['count'])):
            ax6.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     f'{count:,}', ha='left', va='center', fontsize=10)

    # Third row: Financial impact and strategy distribution
    ax7 = fig.add_subplot(gs[2, :2])  # Revenue breakdown
    ax8 = fig.add_subplot(gs[2, 2:])  # Strategy types

    # Revenue breakdown
    if business_impact and 'segment_impacts' in business_impact:
        segments = list(business_impact['segment_impacts'].keys())
        retention_rev = [business_impact['segment_impacts'][s].get('retention_revenue', 0) / 1000
                         for s in segments]
        value_rev = [business_impact['segment_impacts'][s].get('value_increase_revenue', 0) / 1000
                     for s in segments]
        engagement_rev = [business_impact['segment_impacts'][s].get('engagement_revenue', 0) / 1000
                          for s in segments]

        width = 0.6
        ax7.bar(segments, retention_rev, width, label='Customer Retention', color='lightgreen')
        ax7.bar(segments, value_rev, width, bottom=retention_rev,
                label='Value Increase', color='skyblue')
        ax7.bar(segments, engagement_rev, width,
                bottom=[r + v for r, v in zip(retention_rev, value_rev)],
                label='Engagement Boost', color='coral')

        ax7.set_title('Revenue Impact by Source ($K)', fontsize=14, weight='bold')
        ax7.set_xlabel('Customer Segment')
        ax7.set_ylabel('Revenue Impact ($K)')
        ax7.legend()

    # Strategy types distribution
    if strategies:
        strategy_types = [s['strategy_type'] for s in strategies.values()]
        strategy_counts = pd.Series(strategy_types).value_counts()

        # Create pie chart with custom colors
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
        wedges, texts, autotexts = ax8.pie(strategy_counts.values, labels=strategy_counts.index,
                                           autopct='%1.1f%%', startangle=90, colors=colors)
        ax8.set_title('Personalization Strategy Distribution', fontsize=14, weight='bold')

        # Improve text readability
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_weight('bold')

    # Fourth row: Timeline and recommendations
    ax9 = fig.add_subplot(gs[3, :2])  # Implementation timeline
    ax10 = fig.add_subplot(gs[3, 2:])  # Key recommendations

    # Implementation timeline
    timeline_data = {
        'Phase 1\n(Months 1-6)': 40,  # Percentage of total effort
        'Phase 2\n(Months 7-12)': 35,
        'Phase 3\n(Months 13-18)': 25
    }

    ax9.barh(list(timeline_data.keys()), list(timeline_data.values()),
             color=['red', 'orange', 'green'], alpha=0.7)
    ax9.set_title('Implementation Timeline', fontsize=14, weight='bold')
    ax9.set_xlabel('Effort Distribution (%)')

    # Add phase descriptions
    descriptions = [
        'Critical segments & infrastructure',
        'Full rollout & optimization',
        'Scaling & continuous improvement'
    ]
    for i, (phase, desc) in enumerate(zip(timeline_data.keys(), descriptions)):
        ax9.text(timeline_data[phase] + 2, i, desc, va='center', fontsize=9)

    # Key recommendations
    ax10.text(0.05, 0.95, 'KEY RECOMMENDATIONS', fontsize=14, weight='bold',
              transform=ax10.transAxes)

    recommendations = [
        '• IMMEDIATE: Focus on 3,251 high-value, high-risk customers',
        '• PRIORITY: Implement retention strategies for mobile customers',
        '• GROWTH: Leverage grocery segment loyalty for cross-selling',
        '• TECHNOLOGY: Deploy real-time personalization engine',
        '• MEASUREMENT: Establish comprehensive A/B testing framework'
    ]

    for i, rec in enumerate(recommendations):
        ax10.text(0.05, 0.8 - i * 0.15, rec, fontsize=11, transform=ax10.transAxes)

    ax10.set_xlim(0, 1)
    ax10.set_ylim(0, 1)
    ax10.axis('off')
    ax10.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, fill=False, edgecolor='blue', lw=2))

    # Add footer with key insights
    fig.text(0.5, 0.02,
             f'KEY INSIGHT: Counter-intuitive finding - Higher satisfaction scores correlate with higher churn rates. '
             f'Strategy focuses on consistent experience delivery rather than satisfaction maximization alone.',
             ha='center', fontsize=10, style='italic', wrap=True)

    file_path = os.path.join(save_path, 'executive_dashboard.png')
    plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    created_files['executive_dashboard'] = file_path

    return created_files


def create_interactive_dashboard(data, business_impact, strategies, save_path):
    """Create interactive Plotly dashboard (optional)."""

    created_files = {}

    try:
        # Create interactive dashboard using Plotly
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Customer Segments', 'Churn Risk Distribution',
                            'Revenue Impact by Segment', 'Strategy Performance',
                            'Implementation Timeline', 'ROI Analysis'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )

        # Add plots (simplified version)
        if 'CustomerSegment' in data.columns:
            segment_counts = data['CustomerSegment'].value_counts().sort_index()
            fig.add_trace(go.Pie(labels=[f'Segment {i}' for i in segment_counts.index],
                                 values=segment_counts.values,
                                 name="Segments"), row=1, col=1)

        # Add more interactive elements as needed...

        fig.update_layout(height=1200, showlegend=False,
                          title_text="Interactive Dohtem Personalization Dashboard")

        file_path = os.path.join(save_path, 'interactive_dashboard.html')
        fig.write_html(file_path)
        created_files['interactive_dashboard'] = file_path

    except Exception as e:
        print(f"   ⚠️  Could not create interactive dashboard: {str(e)}")

    return created_files


def save_visualization_summary(created_files, save_path):
    """Save a summary of all created visualizations."""

    summary = {
        'total_visualizations': len(created_files),
        'visualization_files': created_files,
        'creation_timestamp': pd.Timestamp.now().isoformat(),
        'description': 'Comprehensive visualization suite for Dohtem personalization analysis'
    }

    summary_path = os.path.join(save_path, 'visualization_summary.json')
    with open(summary_path, 'w') as f:
        import json
        json.dump(summary, f, indent=2)

    print(f"📋 Visualization summary saved to: {summary_path}")
    return summary_path


def demonstrate_visualizations():
    """Demonstration function for the visualization utilities."""
    print("📈 Visualization Utils Demo")
    print("=" * 50)

    try:
        # Create sample data
        np.random.seed(42)
        n_customers = 2000

        sample_data = pd.DataFrame({
            'CustomerID': range(1, n_customers + 1),
            'CustomerSegment': np.random.randint(0, 4, n_customers),
            'Churn': np.random.binomial(1, 0.17, n_customers),
            'SatisfactionScore': np.random.randint(1, 6, n_customers),
            'PreferedOrderCat': np.random.choice(['Mobile', 'Fashion', 'Grocery', 'Laptop & Accessory'], n_customers),
            'CustomerValueScore': np.random.normal(1.5, 0.8, n_customers),
            'EngagementScore': np.random.normal(2.0, 1.0, n_customers)
        })

        # Sample business impact
        sample_impact = {
            'total_customers': n_customers,
            'total_revenue_impact': 750000,
            'roi_percentage': 315,
            'payback_period_months': 3.8,
            'segment_impacts': {
                0: {'retention_revenue': 200000, 'value_increase_revenue': 150000, 'engagement_revenue': 100000},
                1: {'retention_revenue': 100000, 'value_increase_revenue': 75000, 'engagement_revenue': 50000}
            }
        }

        # Sample strategies
        sample_strategies = {
            0: {'strategy_type': 'Premium_Retention', 'priority_level': 'Critical', 'segment_size': 500},
            1: {'strategy_type': 'Growth_Premium', 'priority_level': 'High', 'segment_size': 450}
        }

        print(f"📊 Creating demonstration visualizations...")

        # Create data exploration visualizations
        data_viz = create_data_exploration_visualizations(sample_data, sample_data, 'demo_visualizations/')
        print(f"   Created {len(data_viz)} data exploration charts")

        # Create business impact visualizations
        impact_viz = create_business_impact_visualizations(sample_impact, 'demo_visualizations/')
        print(f"   Created {len(impact_viz)} business impact charts")

        # Create executive dashboard
        dashboard_viz = create_executive_dashboard(sample_data, sample_impact, sample_strategies,
                                                   'demo_visualizations/')
        print(f"   Created {len(dashboard_viz)} dashboard files")

        all_files = {**data_viz, **impact_viz, **dashboard_viz}

        print(f"\n✅ Demo completed! Created {len(all_files)} visualization files")

        return all_files

    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}


if __name__ == "__main__":
    # Run demonstration
    viz_files = demonstrate_visualizations()

    if viz_files:
        print(f"\n🎨 Visualization utilities ready for production!")
        print(f"📊 Capable of creating comprehensive visualization suites")
        print(f"🎯 Executive-ready dashboards and detailed analysis charts")
