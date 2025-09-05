import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from typing import Dict, List, Tuple, Any
import warnings

warnings.filterwarnings('ignore')


class BusinessImpactCalculator:
    """
    Comprehensive business impact calculator for personalization strategies.

    Features:
    - ROI calculation and projections
    - Cost-benefit analysis
    - Segment-specific impact assessment
    - Scenario analysis and sensitivity testing
    - Financial reporting and visualization
    """

    def __init__(self, avg_customer_value=1200, implementation_cost=225000,
                 monthly_operational_cost=18750):
        """
        Initialize the business impact calculator.

        Parameters:
        -----------
        avg_customer_value : float, default=1200
            Average customer lifetime value in dollars
        implementation_cost : float, default=225000
            Total upfront implementation cost
        monthly_operational_cost : float, default=18750
            Monthly operational cost (staff, technology, maintenance)
        """
        self.avg_customer_value = avg_customer_value
        self.implementation_cost = implementation_cost
        self.monthly_operational_cost = monthly_operational_cost
        self.annual_operational_cost = monthly_operational_cost * 12

        # Business metrics and assumptions
        self.business_assumptions = {
            'customer_acquisition_cost': 150,  # Cost to acquire new customer
            'average_order_value': 85,  # Average order value
            'orders_per_year': 8,  # Average orders per customer per year
            'profit_margin': 0.25,  # Profit margin on sales
            'retention_cost_multiplier': 0.2,  # Cost of retention vs acquisition
            'churn_replacement_rate': 0.8,  # Rate at which churned customers are replaced
            'implementation_timeline_months': 6,  # Implementation timeline
            'ramp_up_period_months': 3  # Time to reach full impact
        }

        # Impact calculations storage
        self.segment_impacts = {}
        self.overall_impact = {}
        self.scenario_analyses = {}
        self.sensitivity_results = {}

    def calculate_comprehensive_impact(self, customer_data, personalization_strategies):
        """
        Calculate comprehensive business impact across all segments and strategies.

        Parameters:
        -----------
        customer_data : pd.DataFrame
            Customer data with segments
        personalization_strategies : dict
            Personalization strategies by segment

        Returns:
        --------
        dict
            Comprehensive business impact analysis
        """
        print(f"\n💰 CALCULATING COMPREHENSIVE BUSINESS IMPACT")
        print("=" * 60)

        total_customers = len(customer_data)
        current_churn_rate = customer_data['Churn'].mean()

        print(f"📊 Baseline Metrics:")
        print(f"   Total Customers: {total_customers:,}")
        print(f"   Current Churn Rate: {current_churn_rate:.1%}")
        print(f"   Average Customer Value: ${self.avg_customer_value:,}")

        # Calculate segment-specific impacts
        self._calculate_segment_impacts(customer_data, personalization_strategies)

        # Calculate overall business impact
        self._calculate_overall_impact(total_customers, current_churn_rate)

        # Perform scenario analysis
        self._perform_scenario_analysis()

        # Calculate time-series projections
        self._calculate_time_series_projections()

        print(f"\n✅ Business impact analysis completed!")
        print(f"💰 Total Annual Revenue Impact: ${self.overall_impact['total_revenue_impact']:,.0f}")
        print(f"🎯 ROI: {self.overall_impact['roi_percentage']:.0f}%")
        print(f"📈 Payback Period: {self.overall_impact['payback_period_months']:.1f} months")

        return self.overall_impact

    def _calculate_segment_impacts(self, customer_data, strategies):
        """Calculate detailed impact for each customer segment."""

        print(f"\n🎯 Calculating segment-specific impacts:")

        for segment_id, strategy in strategies.items():
            segment_data = customer_data[customer_data['CustomerSegment'] == segment_id]

            if len(segment_data) == 0:
                continue

            # Base segment metrics
            segment_size = len(segment_data)
            current_churn_rate = segment_data['Churn'].mean()
            current_churned = int(segment_size * current_churn_rate)

            # Strategy effectiveness (from personalization engine)
            strategy_type = strategy['strategy_type']
            effectiveness = self._get_strategy_effectiveness(strategy_type)

            # Calculate retention impact
            customers_retained = int(current_churned * effectiveness['churn_reduction'])
            retention_revenue = customers_retained * self.avg_customer_value

            # Calculate value increase impact (existing customers)
            remaining_customers = segment_size - current_churned + customers_retained
            value_increase_revenue = remaining_customers * self.avg_customer_value * effectiveness['value_increase']

            # Calculate engagement impact
            engagement_revenue = segment_size * self.business_assumptions['average_order_value'] * effectiveness[
                'engagement_increase']

            # Calculate costs
            implementation_cost_segment = self._estimate_segment_implementation_cost(segment_size, strategy_type)
            annual_operational_cost_segment = implementation_cost_segment * 0.2  # 20% of implementation cost annually

            # Net impact calculations
            total_revenue_impact = retention_revenue + value_increase_revenue + engagement_revenue
            total_costs = implementation_cost_segment + annual_operational_cost_segment
            net_impact = total_revenue_impact - total_costs
            roi_segment = (net_impact / total_costs * 100) if total_costs > 0 else 0

            # Store segment impact
            self.segment_impacts[segment_id] = {
                'segment_size': segment_size,
                'strategy_type': strategy_type,
                'current_churn_rate': current_churn_rate,
                'current_churned_customers': current_churned,
                'customers_retained': customers_retained,
                'retention_revenue': retention_revenue,
                'value_increase_revenue': value_increase_revenue,
                'engagement_revenue': engagement_revenue,
                'total_revenue_impact': total_revenue_impact,
                'implementation_cost': implementation_cost_segment,
                'annual_operational_cost': annual_operational_cost_segment,
                'total_costs': total_costs,
                'net_impact': net_impact,
                'roi_percentage': roi_segment,
                'effectiveness_metrics': effectiveness
            }

            print(f"   Segment {segment_id} ({strategy_type}):")
            print(f"     💰 Revenue Impact: ${total_revenue_impact:,.0f}")
            print(f"     👥 Customers Retained: {customers_retained}")
            print(f"     📈 Segment ROI: {roi_segment:.1f}%")

    def _get_strategy_effectiveness(self, strategy_type):
        """Get effectiveness multipliers for different strategy types."""

        effectiveness_map = {
            'Premium_Retention': {
                'churn_reduction': 0.40,  # 40% reduction in churn
                'value_increase': 0.25,   # 25% increase in customer value
                'engagement_increase': 0.30,  # 30% increase in engagement/orders
                'implementation_complexity': 'High',
                'time_to_impact_months': 2
            },
            'Retention_Focus': {
                'churn_reduction': 0.25,
                'value_increase': 0.15,
                'engagement_increase': 0.20,
                'implementation_complexity': 'Medium',
                'time_to_impact_months': 3
            },
            'Growth_Premium': {
                'churn_reduction': 0.15,
                'value_increase': 0.30,
                'engagement_increase': 0.25,
                'implementation_complexity': 'Medium-High',
                'time_to_impact_months': 4
            },
            'Engagement_Amplification': {
                'churn_reduction': 0.20,
                'value_increase': 0.20,
                'engagement_increase': 0.40,
                'implementation_complexity': 'High',
                'time_to_impact_months': 5
            },
            'Standard_Growth': {
                'churn_reduction': 0.10,
                'value_increase': 0.10,
                'engagement_increase': 0.15,
                'implementation_complexity': 'Low',
                'time_to_impact_months': 6
            }
        }

        return effectiveness_map.get(strategy_type, effectiveness_map['Standard_Growth'])

    def _estimate_segment_implementation_cost(self, segment_size, strategy_type):
        """Estimate implementation cost for a specific segment."""

        # Base cost per customer by strategy complexity
        cost_per_customer = {
            'Premium_Retention': 25,  # High-touch, expensive
            'Retention_Focus': 15,    # Medium automation
            'Growth_Premium': 20,     # Premium features
            'Engagement_Amplification': 18,  # Gamification/social features
            'Standard_Growth': 8      # Basic automation
        }

        per_customer_cost = cost_per_customer.get(strategy_type, 12)

        # Calculate with economies of scale
        base_cost = segment_size * per_customer_cost

        # Apply economies of scale (larger segments have lower per-customer costs)
        if segment_size > 1000:
            scale_factor = 0.85  # 15% discount for large segments
        elif segment_size > 500:
            scale_factor = 0.90  # 10% discount for medium segments
        else:
            scale_factor = 1.0   # No discount for small segments

        return base_cost * scale_factor

    def _calculate_overall_impact(self, total_customers, current_churn_rate):
        """Calculate overall business impact across all segments."""

        # Aggregate segment impacts
        total_revenue_impact = sum(s['total_revenue_impact'] for s in self.segment_impacts.values())
        total_implementation_cost = sum(s['implementation_cost'] for s in self.segment_impacts.values())
        total_operational_cost = sum(s['annual_operational_cost'] for s in self.segment_impacts.values())
        total_customers_retained = sum(s['customers_retained'] for s in self.segment_impacts.values())

        # Add company-wide operational costs
        total_costs = total_implementation_cost + total_operational_cost + self.annual_operational_cost

        # Calculate overall metrics
        net_benefit = total_revenue_impact - total_costs
        roi_percentage = (net_benefit / total_costs * 100) if total_costs > 0 else 0
        payback_period_months = (total_costs / (total_revenue_impact / 12)) if total_revenue_impact > 0 else float('inf')

        # Calculate churn rate improvement
        current_churned_customers = int(total_customers * current_churn_rate)
        new_churn_rate = (current_churned_customers - total_customers_retained) / total_customers
        churn_rate_improvement = current_churn_rate - new_churn_rate

        # Calculate customer lifetime impact
        avg_customer_lifespan_improvement = total_customers_retained / total_customers * 12  # months

        # Store overall impact
        self.overall_impact = {
            'total_customers': total_customers,
            'current_churn_rate': current_churn_rate,
            'new_churn_rate': new_churn_rate,
            'churn_rate_improvement': churn_rate_improvement,
            'total_customers_retained': total_customers_retained,
            'retention_rate_improvement': total_customers_retained / total_customers,

            # Financial metrics
            'total_revenue_impact': total_revenue_impact,
            'total_implementation_cost': total_implementation_cost,
            'total_operational_cost': total_operational_cost + self.annual_operational_cost,
            'total_costs': total_costs,
            'net_benefit': net_benefit,
            'roi_percentage': roi_percentage,
            'payback_period_months': payback_period_months,

            # Customer value metrics
            'avg_customer_value': self.avg_customer_value,
            'customer_lifespan_improvement_months': avg_customer_lifespan_improvement,
            'revenue_per_customer_increase': total_revenue_impact / total_customers,

            # Segment breakdown
            'segments_analyzed': len(self.segment_impacts),
            'segment_impacts': self.segment_impacts
        }

    def _perform_scenario_analysis(self):
        """Perform scenario analysis with different assumptions."""

        scenarios = {
            'Conservative': {
                'effectiveness_multiplier': 0.7,  # 30% lower effectiveness
                'cost_multiplier': 1.3,          # 30% higher costs
                'timeline_multiplier': 1.5       # 50% longer implementation
            },
            'Optimistic': {
                'effectiveness_multiplier': 1.3,  # 30% higher effectiveness
                'cost_multiplier': 0.8,          # 20% lower costs
                'timeline_multiplier': 0.8       # 20% faster implementation
            },
            'Realistic': {
                'effectiveness_multiplier': 1.0,  # Baseline
                'cost_multiplier': 1.0,          # Baseline
                'timeline_multiplier': 1.0       # Baseline
            }
        }

        print(f"\n📊 Performing scenario analysis:")

        for scenario_name, multipliers in scenarios.items():
            # Adjust revenue impact
            adjusted_revenue = self.overall_impact['total_revenue_impact'] * multipliers['effectiveness_multiplier']

            # Adjust costs
            adjusted_costs = self.overall_impact['total_costs'] * multipliers['cost_multiplier']

            # Calculate adjusted metrics
            adjusted_net_benefit = adjusted_revenue - adjusted_costs
            adjusted_roi = (adjusted_net_benefit / adjusted_costs * 100) if adjusted_costs > 0 else 0
            adjusted_payback = (adjusted_costs / (adjusted_revenue / 12)) if adjusted_revenue > 0 else float('inf')

            self.scenario_analyses[scenario_name] = {
                'revenue_impact': adjusted_revenue,
                'total_costs': adjusted_costs,
                'net_benefit': adjusted_net_benefit,
                'roi_percentage': adjusted_roi,
                'payback_period_months': adjusted_payback,
                'multipliers': multipliers
            }

            print(f"   {scenario_name}: ROI {adjusted_roi:.0f}%, Payback {adjusted_payback:.1f} months")

    def _calculate_time_series_projections(self):
        """Calculate month-by-month projections for 24 months."""

        months = 24
        projections = {
            'month': list(range(1, months + 1)),
            'cumulative_revenue': [],
            'cumulative_costs': [],
            'cumulative_net_benefit': [],
            'monthly_revenue': [],
            'monthly_costs': [],
            'roi_to_date': []
        }

        # Implementation timeline (6 months ramp-up)
        implementation_months = self.business_assumptions['implementation_timeline_months']
        ramp_up_months = self.business_assumptions['ramp_up_period_months']

        monthly_revenue_target = self.overall_impact['total_revenue_impact'] / 12
        monthly_cost_target = self.overall_impact['total_costs'] / 12

        cumulative_revenue = 0
        cumulative_costs = 0

        for month in range(1, months + 1):
            # Revenue ramp-up during implementation
            if month <= implementation_months:
                revenue_multiplier = 0  # No revenue during implementation
            elif month <= implementation_months + ramp_up_months:
                # Linear ramp-up during ramp-up period
                ramp_progress = (month - implementation_months) / ramp_up_months
                revenue_multiplier = ramp_progress
            else:
                revenue_multiplier = 1.0  # Full revenue after ramp-up

            # Calculate monthly values
            monthly_revenue = monthly_revenue_target * revenue_multiplier
            monthly_costs = monthly_cost_target  # Costs start immediately

            # Accumulate
            cumulative_revenue += monthly_revenue
            cumulative_costs += monthly_costs
            cumulative_net_benefit = cumulative_revenue - cumulative_costs

            # Calculate ROI to date
            roi_to_date = (cumulative_net_benefit / cumulative_costs * 100) if cumulative_costs > 0 else 0

            # Store values
            projections['cumulative_revenue'].append(cumulative_revenue)
            projections['cumulative_costs'].append(cumulative_costs)
            projections['cumulative_net_benefit'].append(cumulative_net_benefit)
            projections['monthly_revenue'].append(monthly_revenue)
            projections['monthly_costs'].append(monthly_costs)
            projections['roi_to_date'].append(roi_to_date)

        self.time_series_projections = projections

    def perform_sensitivity_analysis(self, parameter_ranges=None):
        """
        Perform sensitivity analysis on key parameters.

        Parameters:
        -----------
        parameter_ranges : dict, optional
            Custom parameter ranges for sensitivity testing
        """
        if parameter_ranges is None:
            parameter_ranges = {
                'avg_customer_value': [800, 1000, 1200, 1500, 2000],
                'churn_reduction_effectiveness': [0.15, 0.20, 0.25, 0.30, 0.35],
                'implementation_cost_factor': [0.7, 0.85, 1.0, 1.15, 1.3],
                'value_increase_effectiveness': [0.10, 0.15, 0.20, 0.25, 0.30]
            }

        print(f"\n🔬 Performing sensitivity analysis:")

        sensitivity_results = {}
        base_roi = self.overall_impact['roi_percentage']
        base_revenue = self.overall_impact['total_revenue_impact']

        for parameter, values in parameter_ranges.items():
            parameter_impact = []

            for value in values:
                # Calculate impact with adjusted parameter
                if parameter == 'avg_customer_value':
                    adjusted_revenue = base_revenue * (value / self.avg_customer_value)
                    adjusted_costs = self.overall_impact['total_costs']
                elif parameter == 'churn_reduction_effectiveness':
                    # Adjust revenue based on churn reduction effectiveness
                    retention_revenue = sum(s['retention_revenue'] for s in self.segment_impacts.values())
                    baseline_effectiveness = 0.25  # Average effectiveness
                    adjustment_factor = value / baseline_effectiveness
                    adjusted_revenue = base_revenue + (retention_revenue * (adjustment_factor - 1))
                    adjusted_costs = self.overall_impact['total_costs']
                elif parameter == 'implementation_cost_factor':
                    adjusted_revenue = base_revenue
                    adjusted_costs = self.overall_impact['total_costs'] * value
                elif parameter == 'value_increase_effectiveness':
                    value_increase_revenue = sum(s['value_increase_revenue'] for s in self.segment_impacts.values())
                    baseline_effectiveness = 0.20  # Average effectiveness
                    adjustment_factor = value / baseline_effectiveness
                    adjusted_revenue = base_revenue + (value_increase_revenue * (adjustment_factor - 1))
                    adjusted_costs = self.overall_impact['total_costs']

                # Calculate adjusted ROI
                net_benefit = adjusted_revenue - adjusted_costs
                roi = (net_benefit / adjusted_costs * 100) if adjusted_costs > 0 else 0

                parameter_impact.append({
                    'parameter_value': value,
                    'roi': roi,
                    'revenue_impact': adjusted_revenue,
                    'net_benefit': net_benefit
                })

            sensitivity_results[parameter] = parameter_impact

            # Print sensitivity summary
            roi_range = [p['roi'] for p in parameter_impact]
            print(f"   {parameter}: ROI range {min(roi_range):.0f}% - {max(roi_range):.0f}%")

        self.sensitivity_results = sensitivity_results
        return sensitivity_results

    def get_impact_summary(self):
        """
        Get comprehensive impact summary as DataFrame.

        Returns:
        --------
        pd.DataFrame
            Business impact summary
        """
        if not self.overall_impact:
            raise ValueError("Impact analysis not performed. Run calculate_comprehensive_impact() first.")

        # Overall summary
        overall_data = {
            'Metric': [
                'Total Customers',
                'Current Churn Rate',
                'New Churn Rate',
                'Customers Retained',
                'Total Revenue Impact',
                'Implementation Cost',
                'Annual Operational Cost',
                'Net Annual Benefit',
                'ROI Percentage',
                'Payback Period (months)'
            ],
            'Value': [
                f"{self.overall_impact['total_customers']:,}",
                f"{self.overall_impact['current_churn_rate']:.1%}",
                f"{self.overall_impact['new_churn_rate']:.1%}",
                f"{self.overall_impact['total_customers_retained']:,}",
                f"${self.overall_impact['total_revenue_impact']:,.0f}",
                f"${self.overall_impact['total_implementation_cost']:,.0f}",
                f"${self.overall_impact['total_operational_cost']:,.0f}",
                f"${self.overall_impact['net_benefit']:,.0f}",
                f"{self.overall_impact['roi_percentage']:.1f}%",
                f"{self.overall_impact['payback_period_months']:.1f}"
            ]
        }

        return pd.DataFrame(overall_data)

    def plot_impact_analysis(self, figsize=(18, 12)):
        """
        Create comprehensive business impact visualizations.

        Parameters:
        -----------
        figsize : tuple, default=(18, 12)
            Figure size for the plots
        """
        if not self.overall_impact:
            raise ValueError("Impact analysis not performed. Run calculate_comprehensive_impact() first.")

        fig, axes = plt.subplots(3, 3, figsize=figsize)

        # Plot 1: Revenue Impact by Segment
        segments = list(self.segment_impacts.keys())
        revenues = [self.segment_impacts[s]['total_revenue_impact']/1000 for s in segments]

        axes[0, 0].bar(segments, revenues, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Revenue Impact by Segment ($K)')
        axes[0, 0].set_xlabel('Segment ID')
        axes[0, 0].set_ylabel('Revenue Impact ($K)')

        # Plot 2: ROI by Segment
        rois = [self.segment_impacts[s]['roi_percentage'] for s in segments]
        colors = ['green' if roi > 200 else 'orange' if roi > 100 else 'red' for roi in rois]

        axes[0, 1].bar(segments, rois, color=colors, alpha=0.7)
        axes[0, 1].set_title('ROI by Segment (%)')
        axes[0, 1].set_xlabel('Segment ID')
        axes[0, 1].set_ylabel('ROI (%)')
        axes[0, 1].axhline(y=100, color='black', linestyle='--', alpha=0.5, label='Break-even')
        axes[0, 1].legend()

        # Plot 3: Customers Retained by Segment
        retained = [self.segment_impacts[s]['customers_retained'] for s in segments]

        axes[0, 2].bar(segments, retained, color='lightgreen', alpha=0.7)
        axes[0, 2].set_title('Customers Retained by Segment')
        axes[0, 2].set_xlabel('Segment ID')
        axes[0, 2].set_ylabel('Customers Retained')

        # Plot 4: Cost-Benefit Breakdown
        revenue_components = ['Retention', 'Value Increase', 'Engagement']
        revenue_values = [
            sum(s['retention_revenue'] for s in self.segment_impacts.values())/1000,
            sum(s['value_increase_revenue'] for s in self.segment_impacts.values())/1000,
            sum(s['engagement_revenue'] for s in self.segment_impacts.values())/1000
        ]

        axes[1, 0].pie(revenue_values, labels=revenue_components, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Revenue Impact Breakdown')

        # Plot 5: Scenario Analysis
        if self.scenario_analyses:
            scenarios = list(self.scenario_analyses.keys())
            scenario_rois = [self.scenario_analyses[s]['roi_percentage'] for s in scenarios]
            scenario_colors = ['red', 'blue', 'green']  # Conservative, Realistic, Optimistic

            axes[1, 1].bar(scenarios, scenario_rois, color=scenario_colors, alpha=0.7)
            axes[1, 1].set_title('ROI by Scenario')
            axes[1, 1].set_ylabel('ROI (%)')
            axes[1, 1].tick_params(axis='x', rotation=45)

        # Plot 6: Time Series Projections
        if hasattr(self, 'time_series_projections'):
            months = self.time_series_projections['month']
            cum_revenue = [r/1000 for r in self.time_series_projections['cumulative_revenue']]
            cum_costs = [c/1000 for c in self.time_series_projections['cumulative_costs']]
            net_benefit = [n/1000 for n in self.time_series_projections['cumulative_net_benefit']]

            axes[1, 2].plot(months, cum_revenue, label='Cumulative Revenue', color='green')
            axes[1, 2].plot(months, cum_costs, label='Cumulative Costs', color='red')
            axes[1, 2].plot(months, net_benefit, label='Net Benefit', color='blue')
            axes[1, 2].set_title('24-Month Financial Projections ($K)')
            axes[1, 2].set_xlabel('Months')
            axes[1, 2].set_ylabel('Amount ($K)')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)

        # Plot 7: Strategy Type Performance
        strategy_types = {}
        for segment_data in self.segment_impacts.values():
            strategy_type = segment_data['strategy_type']
            if strategy_type not in strategy_types:
                strategy_types[strategy_type] = {'revenue': 0, 'segments': 0}
            strategy_types[strategy_type]['revenue'] += segment_data['total_revenue_impact']
            strategy_types[strategy_type]['segments'] += 1

        strategy_names = list(strategy_types.keys())
        strategy_revenues = [strategy_types[s]['revenue']/1000 for s in strategy_names]

        axes[2, 0].barh(strategy_names, strategy_revenues, color='coral', alpha=0.7)
        axes[2, 0].set_title('Revenue Impact by Strategy Type ($K)')
        axes[2, 0].set_xlabel('Revenue Impact ($K)')

        # Plot 8: Payback Analysis
        if hasattr(self, 'time_series_projections'):
            # Find break-even point
            net_benefits = self.time_series_projections['cumulative_net_benefit']
            break_even_month = None
            for i, benefit in enumerate(net_benefits):
                if benefit > 0:
                    break_even_month = i + 1
                    break

            axes[2, 1].plot(months, net_benefit, color='purple', linewidth=2)
            axes[2, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            if break_even_month:
                axes[2, 1].axvline(x=break_even_month, color='green', linestyle='--',
                                 label=f'Break-even: Month {break_even_month}')
                axes[2, 1].legend()
            axes[2, 1].set_title('Payback Analysis')
            axes[2, 1].set_xlabel('Months')
            axes[2, 1].set_ylabel('Cumulative Net Benefit ($K)')
            axes[2, 1].grid(True, alpha=0.3)

        # Plot 9: Sensitivity Analysis (if available)
        if self.sensitivity_results:
            # Plot sensitivity for customer value parameter
            customer_value_data = self.sensitivity_results.get('avg_customer_value', [])
            if customer_value_data:
                values = [d['parameter_value'] for d in customer_value_data]
                rois = [d['roi'] for d in customer_value_data]

                axes[2, 2].plot(values, rois, 'o-', color='orange', linewidth=2)
                axes[2, 2].set_title('Sensitivity: Customer Value vs ROI')
                axes[2, 2].set_xlabel('Average Customer Value ($)')
                axes[2, 2].set_ylabel('ROI (%)')
                axes[2, 2].grid(True, alpha=0.3)
        else:
            axes[2, 2].text(0.5, 0.5, 'Run sensitivity_analysis()\nfor detailed sensitivity plots',
                          ha='center', va='center', transform=axes[2, 2].transAxes)
            axes[2, 2].set_title('Sensitivity Analysis')

        plt.tight_layout()
        return fig

    def save_analysis(self, file_path='reports/business_impact_analysis.json'):
        """
        Save business impact analysis to JSON file.

        Parameters:
        -----------
        file_path : str
            Path where to save the analysis
        """
        if not self.overall_impact:
            raise ValueError("Impact analysis not performed. Run calculate_comprehensive_impact() first.")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Prepare data for JSON serialization
        save_data = {
            'overall_impact': self.overall_impact,
            'segment_impacts': self.segment_impacts,
            'scenario_analyses': self.scenario_analyses,
            'sensitivity_results': self.sensitivity_results,
            'time_series_projections': getattr(self, 'time_series_projections', {}),
            'business_assumptions': self.business_assumptions,
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }

        # Convert numpy types to native Python types
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        save_data = convert_numpy_types(save_data)

        # Save to JSON
        with open(file_path, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)

        print(f"✅ Business impact analysis saved to: {file_path}")

    def generate_executive_report(self):
        """
        Generate executive-level business impact report.

        Returns:
        --------
        str
            Executive business impact report
        """
        if not self.overall_impact:
            raise ValueError("Impact analysis not performed. Run calculate_comprehensive_impact() first.")

        # Key metrics for executive summary
        total_revenue = self.overall_impact['total_revenue_impact']
        total_investment = self.overall_impact['total_costs']
        roi = self.overall_impact['roi_percentage']
        payback_months = self.overall_impact['payback_period_months']
        customers_retained = self.overall_impact['total_customers_retained']

        # Risk assessment
        conservative_roi = self.scenario_analyses.get('Conservative', {}).get('roi_percentage', 0)
        optimistic_roi = self.scenario_analyses.get('Optimistic', {}).get('roi_percentage', 0)

        report = f"""
DOHTEM PERSONALIZATION INITIATIVE - BUSINESS IMPACT ANALYSIS
===========================================================

EXECUTIVE SUMMARY:
The proposed personalization initiative represents a high-impact investment opportunity 
with compelling financial returns and significant competitive advantages.

KEY FINANCIAL METRICS:
• Total Investment Required: ${total_investment:,.0f}
• Expected Annual Revenue Impact: ${total_revenue:,.0f}
• Net Annual Benefit: ${self.overall_impact['net_benefit']:,.0f}
• Return on Investment: {roi:.0f}%
• Payback Period: {payback_months:.1f} months

CUSTOMER IMPACT:
• Total Customers Analyzed: {self.overall_impact['total_customers']:,}
• Customers at Risk Retained: {customers_retained:,}
• Churn Rate Reduction: {self.overall_impact['churn_rate_improvement']:.1%}
• Customer Lifetime Value Increase: {self.overall_impact['revenue_per_customer_increase']:.0f}

RISK ANALYSIS:
• Conservative Scenario ROI: {conservative_roi:.0f}%
• Base Case ROI: {roi:.0f}%
• Optimistic Scenario ROI: {optimistic_roi:.0f}%
• Risk Assessment: {"Low" if conservative_roi > 100 else "Medium" if roi > 150 else "High"}

IMPLEMENTATION PHASES:
Phase 1 (Months 1-6): Infrastructure setup and critical segment implementation
Phase 2 (Months 7-12): Full rollout and optimization
Phase 3 (Months 13+): Continuous improvement and scaling

STRATEGIC BENEFITS:
• Competitive differentiation through personalized customer experiences
• Improved customer satisfaction and loyalty
• Enhanced data-driven decision making capabilities
• Scalable platform for future personalization initiatives
• Reduced customer acquisition costs through improved retention

RECOMMENDATION:
PROCEED with immediate implementation. The financial case is compelling with 
{roi:.0f}% ROI and {payback_months:.1f}-month payback period. Even under conservative 
assumptions, the initiative delivers {conservative_roi:.0f}% ROI, well above typical 
hurdle rates for technology investments.

The combination of strong financial returns, customer experience improvements, and 
strategic competitive advantages makes this a high-priority initiative for immediate 
executive approval and funding allocation.
"""

        return report


def demonstrate_business_impact():
    """Demonstration function for the business impact calculator."""
    print("💰 Business Impact Calculator Demo")
    print("=" * 50)

    try:
        # Initialize calculator
        calculator = BusinessImpactCalculator(
            avg_customer_value=1200,
            implementation_cost=225000
        )

        # Create sample data
        np.random.seed(42)
        sample_data = pd.DataFrame({
            'CustomerID': range(1, 3001),
            'CustomerSegment': np.random.randint(0, 4, 3000),
            'Churn': np.random.binomial(1, 0.17, 3000)
        })

        # Create sample strategies
        sample_strategies = {
            0: {'strategy_type': 'Premium_Retention'},
            1: {'strategy_type': 'Growth_Premium'},
            2: {'strategy_type': 'Retention_Focus'},
            3: {'strategy_type': 'Standard_Growth'}
        }

        print(f"📊 Analyzing {len(sample_data)} customers across {len(sample_strategies)} segments...")

        # Calculate comprehensive impact
        impact = calculator.calculate_comprehensive_impact(sample_data, sample_strategies)

        print(f"\n✅ Business impact analysis completed!")

        # Show summary
        summary = calculator.get_impact_summary()
        print(f"\n📋 Impact Summary:")
        print(summary.to_string(index=False))

        # Perform sensitivity analysis
        print(f"\n🔬 Performing sensitivity analysis...")
        sensitivity = calculator.perform_sensitivity_analysis()

        # Generate executive report
        exec_report = calculator.generate_executive_report()
        print(f"\n📊 Executive report generated ({len(exec_report.split())} words)")

        return calculator, impact, summary

    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    # Run demonstration
    calculator, impact, summary = demonstrate_business_impact()

    if calculator is not None:
        print(f"\n🎉 Business Impact Calculator ready for production!")
        print(f"💰 Expected ROI: {impact['roi_percentage']:.0f}%")
        print(f"📈 Revenue Impact: ${impact['total_revenue_impact']:,.0f}")
        print(f"⏱️  Payback Period: {impact['payback_period_months']:.1f} months")
