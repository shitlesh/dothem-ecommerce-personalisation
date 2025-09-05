import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from typing import Dict, List, Tuple, Any
import warnings

warnings.filterwarnings('ignore')


class PersonalizationEngine:
    """
    Advanced personalization engine that creates targeted strategies for customer segments.

    Features:
    - Segment-based personalization strategies
    - Individual customer recommendations
    - Business impact estimation
    - Implementation guidelines
    - Performance tracking capabilities
    """

    def __init__(self):
        """Initialize the personalization engine."""
        self.segment_strategies = {}
        self.personalization_rules = {}
        self.category_customizations = {}
        self.impact_estimations = {}
        self.is_strategies_created = False

        # Initialize category-specific customizations
        self._initialize_category_customizations()

    def _initialize_category_customizations(self):
        """Initialize category-specific personalization customizations."""
        self.category_customizations = {
            'Mobile': {
                'homepage_focus': 'Latest mobile technology and comparisons',
                'email_content': 'Tech reviews and specification comparisons',
                'special_offers': 'Trade-in programs and extended warranties',
                'content_type': 'Video reviews and technical specifications',
                'engagement_strategy': 'Tech-focused community building',
                'retention_approach': 'Innovation showcasing and early access'
            },
            'Mobile Phone': {
                'homepage_focus': 'Phone accessories and protective gear',
                'email_content': 'Accessory recommendations and bundle deals',
                'special_offers': 'Bundle discounts and protection plans',
                'content_type': 'Compatibility guides and setup tutorials',
                'engagement_strategy': 'Accessory education and tips',
                'retention_approach': 'Cross-selling complementary products'
            },
            'Laptop & Accessory': {
                'homepage_focus': 'Productivity and professional equipment',
                'email_content': 'Professional productivity tips and product guides',
                'special_offers': 'Business discounts and bulk purchase options',
                'content_type': 'Performance comparisons and professional reviews',
                'engagement_strategy': 'Professional community and B2B focus',
                'retention_approach': 'Enterprise solutions and support'
            },
            'Fashion': {
                'homepage_focus': 'Trending styles and seasonal collections',
                'email_content': 'Style guides and fashion trends',
                'special_offers': 'Seasonal sales and style subscriptions',
                'content_type': 'Lookbooks and styling videos',
                'engagement_strategy': 'Style inspiration and social features',
                'retention_approach': 'Personal styling and trend updates'
            },
            'Grocery': {
                'homepage_focus': 'Fresh products and bulk purchase options',
                'email_content': 'Recipe suggestions and nutritional information',
                'special_offers': 'Bulk discounts and subscription services',
                'content_type': 'Recipe videos and nutritional guides',
                'engagement_strategy': 'Recipe sharing and meal planning',
                'retention_approach': 'Convenience and freshness guarantees'
            },
            'Others': {
                'homepage_focus': 'Diverse product exploration and discovery',
                'email_content': 'Product discovery and recommendations',
                'special_offers': 'Cross-category discounts and bundles',
                'content_type': 'Product discovery and comparison guides',
                'engagement_strategy': 'Exploration rewards and discovery features',
                'retention_approach': 'Personalized discovery and surprise offers'
            }
        }

    def create_personalization_strategies(self, segmentation_model, churn_model):
        """
        Create comprehensive personalization strategies for each customer segment.

        Parameters:
        -----------
        segmentation_model : CustomerSegmentation
            Trained customer segmentation model
        churn_model : ChurnPredictor
            Trained churn prediction model

        Returns:
        --------
        dict
            Personalization strategies for each segment
        """
        print(f"\n🎯 CREATING PERSONALIZATION STRATEGIES")
        print("=" * 60)

        if not hasattr(segmentation_model, 'segment_profiles') or not segmentation_model.segment_profiles:
            raise ValueError("Segmentation model must have segment profiles. Run fit_segments() first.")

        segment_profiles = segmentation_model.segment_profiles

        for segment_id, profile in segment_profiles.items():
            strategy = self._develop_comprehensive_strategy(segment_id, profile, churn_model)
            self.segment_strategies[segment_id] = strategy

        # Create cross-segment insights
        self._create_cross_segment_insights()

        # Estimate business impact
        self._estimate_strategies_impact()

        self.is_strategies_created = True

        print(f"\n✅ Personalization strategies created for {len(self.segment_strategies)} segments")
        print(f"🎯 Total customers covered: {sum(s['segment_size'] for s in self.segment_strategies.values()):,}")

        return self.segment_strategies

    def _develop_comprehensive_strategy(self, segment_id, profile, churn_model):
        """Develop comprehensive strategy for a specific customer segment."""

        # Extract key metrics
        churn_rate = profile['churn_rate']
        value_score = profile['avg_value_score']
        engagement = profile['avg_engagement']
        top_category = profile['top_category']
        satisfaction = profile['avg_satisfaction']
        segment_size = profile['size']

        # Determine strategy type and approach
        strategy_info = self._classify_segment_strategy(churn_rate, value_score, engagement, satisfaction)

        # Get category-specific customizations
        category_custom = self.category_customizations.get(top_category, self.category_customizations['Others'])

        # Create comprehensive strategy
        strategy = {
            'segment_id': segment_id,
            'segment_description': profile.get('description', f'Segment {segment_id}'),
            'segment_size': segment_size,
            'segment_percentage': profile['percentage'],

            # Risk and value classification
            'strategy_type': strategy_info['type'],
            'priority_level': strategy_info['priority'],
            'churn_risk_level': self._determine_risk_level(churn_rate),
            'value_tier': self._determine_value_tier(value_score),
            'engagement_level': self._determine_engagement_level(engagement),

            # Core metrics
            'churn_rate': churn_rate,
            'avg_value_score': value_score,
            'avg_engagement': engagement,
            'avg_satisfaction': satisfaction,
            'primary_category': top_category,

            # Personalization tactics
            'primary_tactics': strategy_info['tactics'],
            'secondary_tactics': strategy_info['secondary_tactics'],
            'retention_actions': strategy_info['retention_actions'],
            'growth_actions': strategy_info['growth_actions'],

            # Category-specific customizations
            'category_customizations': category_custom,

            # Implementation details
            'implementation_timeline': strategy_info['timeline'],
            'success_metrics': strategy_info['success_metrics'],
            'resource_requirements': strategy_info['resources'],

            # Expected impact
            'expected_impact': self._estimate_segment_impact(segment_size, churn_rate, value_score,
                                                             strategy_info['type']),

            # Communication strategy
            'communication_strategy': self._create_communication_strategy(strategy_info['type'], top_category,
                                                                          engagement),

            # Technology requirements
            'tech_requirements': strategy_info['tech_requirements']
        }

        # Print strategy summary
        print(f"\n🏷️  Segment {segment_id}: {strategy['strategy_type']}")
        print(f"   👥 Size: {segment_size:,} customers ({profile['percentage']:.1f}%)")
        print(
            f"   ⚠️  Risk: {strategy['churn_risk_level']} | Value: {strategy['value_tier']} | Priority: {strategy['priority_level']}")
        print(f"   📋 Key Tactics: {', '.join(strategy['primary_tactics'][:3])}")
        print(f"   💰 Expected Revenue Impact: ${strategy['expected_impact']['revenue_impact']:,.0f}")

        return strategy

    def _classify_segment_strategy(self, churn_rate, value_score, engagement, satisfaction):
        """Classify segment strategy based on customer characteristics."""

        # High churn risk strategies
        if churn_rate > 0.25:
            if value_score > 2.0:  # High value, high risk
                return {
                    'type': 'Premium_Retention',
                    'priority': 'Critical',
                    'tactics': [
                        'Dedicated account manager assignment',
                        'Exclusive VIP loyalty program access',
                        'Proactive customer success outreach',
                        'Premium customer support priority',
                        'Personalized product recommendations',
                        'Special pricing and early access offers'
                    ],
                    'secondary_tactics': [
                        'Executive-level relationship building',
                        'Custom product bundles',
                        'White-glove service delivery'
                    ],
                    'retention_actions': [
                        'Immediate intervention for satisfaction issues',
                        'Quarterly business reviews',
                        'Success metrics tracking and optimization'
                    ],
                    'growth_actions': [
                        'Cross-category expansion opportunities',
                        'Referral program with premium incentives',
                        'Beta product access and feedback collection'
                    ],
                    'timeline': '1-2 weeks implementation',
                    'success_metrics': ['Churn reduction >40%', 'NPS increase >20 points', 'Revenue per customer +25%'],
                    'resources': ['Customer success team', 'Premium support staff', 'Personalization technology'],
                    'tech_requirements': ['Real-time alerts', 'Advanced analytics', 'Personalization engine']
                }
            else:  # Standard value, high risk
                return {
                    'type': 'Retention_Focus',
                    'priority': 'High',
                    'tactics': [
                        'Targeted discount campaigns (10-15%)',
                        'Product recommendation improvements',
                        'Satisfaction survey follow-ups',
                        'Category-specific promotions',
                        'Re-engagement email series',
                        'Customer service priority handling'
                    ],
                    'secondary_tactics': [
                        'Loyalty point bonuses',
                        'Free shipping upgrades',
                        'Extended return policies'
                    ],
                    'retention_actions': [
                        'Monthly satisfaction check-ins',
                        'Proactive issue resolution',
                        'Usage behavior monitoring'
                    ],
                    'growth_actions': [
                        'Category exploration incentives',
                        'Bundle recommendations',
                        'Community engagement programs'
                    ],
                    'timeline': '2-4 weeks implementation',
                    'success_metrics': ['Churn reduction >25%', 'Engagement increase +30%', 'Order frequency +15%'],
                    'resources': ['Marketing automation', 'Customer service', 'Analytics team'],
                    'tech_requirements': ['Automated campaigns', 'Behavioral tracking', 'A/B testing platform']
                }

        # High value, low churn strategies
        elif value_score > 2.5 and churn_rate < 0.15:
            return {
                'type': 'Growth_Premium',
                'priority': 'High',
                'tactics': [
                    'Cross-category product recommendations',
                    'Premium product showcasing',
                    'Bulk purchase incentives',
                    'Subscription service offers',
                    'VIP event invitations',
                    'Referral program with bonuses'
                ],
                'secondary_tactics': [
                    'Early access to new products',
                    'Exclusive member pricing',
                    'Personal shopping assistance'
                ],
                'retention_actions': [
                    'Maintain high service quality',
                    'Regular value demonstration',
                    'Appreciation programs'
                ],
                'growth_actions': [
                    'Wallet share expansion',
                    'Premium tier upselling',
                    'Network effect leveraging'
                ],
                'timeline': '3-6 weeks implementation',
                'success_metrics': ['Revenue per customer +30%', 'Category expansion +50%', 'Referral rate +25%'],
                'resources': ['Product team', 'VIP services', 'Event management'],
                'tech_requirements': ['Advanced recommendations', 'Event management system', 'Referral tracking']
            }

        # High engagement strategies
        elif engagement > 2.5:
            return {
                'type': 'Engagement_Amplification',
                'priority': 'Medium-High',
                'tactics': [
                    'Gamification elements integration',
                    'Social sharing incentives',
                    'User-generated content campaigns',
                    'Community forum access',
                    'Beta product testing invitations',
                    'Achievement and badge systems'
                ],
                'secondary_tactics': [
                    'Influencer collaboration opportunities',
                    'Content creation partnerships',
                    'Social media features'
                ],
                'retention_actions': [
                    'Engagement reward programs',
                    'Community building initiatives',
                    'Feedback loop optimization'
                ],
                'growth_actions': [
                    'Advocacy program development',
                    'Social proof utilization',
                    'Network expansion strategies'
                ],
                'timeline': '4-8 weeks implementation',
                'success_metrics': ['Engagement time +40%', 'Social sharing +60%', 'Community participation +35%'],
                'resources': ['Community management', 'Social media team', 'Gamification platform'],
                'tech_requirements': ['Gamification engine', 'Social features', 'Community platform']
            }

        # Standard growth strategies
        else:
            return {
                'type': 'Standard_Growth',
                'priority': 'Medium',
                'tactics': [
                    'Seasonal promotional campaigns',
                    'Category exploration incentives',
                    'Standard loyalty point accumulation',
                    'Email newsletter optimization',
                    'Mobile app push notifications',
                    'Product discovery features'
                ],
                'secondary_tactics': [
                    'Cross-selling recommendations',
                    'Seasonal content marketing',
                    'Basic personalization features'
                ],
                'retention_actions': [
                    'Regular communication cadence',
                    'Basic satisfaction monitoring',
                    'Standard customer service'
                ],
                'growth_actions': [
                    'Purchase frequency optimization',
                    'Average order value increase',
                    'Category penetration improvement'
                ],
                'timeline': '6-12 weeks implementation',
                'success_metrics': ['Purchase frequency +15%', 'AOV +10%', 'Category expansion +20%'],
                'resources': ['Marketing team', 'Basic automation', 'Standard analytics'],
                'tech_requirements': ['Email platform', 'Basic personalization', 'Standard analytics']
            }

    def _determine_risk_level(self, churn_rate):
        """Determine risk level based on churn rate."""
        if churn_rate >= 0.25:
            return "Critical"
        elif churn_rate >= 0.20:
            return "High"
        elif churn_rate >= 0.15:
            return "Medium"
        else:
            return "Low"

    def _determine_value_tier(self, value_score):
        """Determine value tier based on value score."""
        if value_score >= 2.5:
            return "Premium"
        elif value_score >= 2.0:
            return "High"
        elif value_score >= 1.0:
            return "Standard"
        else:
            return "Basic"

    def _determine_engagement_level(self, engagement_score):
        """Determine engagement level based on engagement score."""
        if engagement_score >= 3.0:
            return "Very High"
        elif engagement_score >= 2.5:
            return "High"
        elif engagement_score >= 1.5:
            return "Medium"
        else:
            return "Low"

    def _estimate_segment_impact(self, segment_size, churn_rate, value_score, strategy_type):
        """Estimate business impact for segment strategy."""

        # Base assumptions
        avg_customer_value = 1200  # Average customer lifetime value

        # Impact multipliers based on strategy type
        impact_multipliers = {
            'Premium_Retention': {'churn_reduction': 0.40, 'value_increase': 0.25},
            'Retention_Focus': {'churn_reduction': 0.25, 'value_increase': 0.15},
            'Growth_Premium': {'churn_reduction': 0.10, 'value_increase': 0.30},
            'Engagement_Amplification': {'churn_reduction': 0.15, 'value_increase': 0.20},
            'Standard_Growth': {'churn_reduction': 0.10, 'value_increase': 0.10}
        }

        multiplier = impact_multipliers.get(strategy_type, impact_multipliers['Standard_Growth'])

        # Calculate impact
        churned_customers_current = int(segment_size * churn_rate)
        customers_retained = int(churned_customers_current * multiplier['churn_reduction'])
        retention_revenue = customers_retained * avg_customer_value

        # Value increase for existing customers
        remaining_customers = segment_size - churned_customers_current + customers_retained
        value_increase_revenue = remaining_customers * avg_customer_value * multiplier['value_increase']

        total_revenue_impact = retention_revenue + value_increase_revenue

        return {
            'segment_size': segment_size,
            'customers_at_risk': churned_customers_current,
            'customers_retained': customers_retained,
            'retention_revenue': retention_revenue,
            'value_increase_revenue': value_increase_revenue,
            'revenue_impact': total_revenue_impact,
            'churn_reduction_rate': multiplier['churn_reduction'],
            'value_increase_rate': multiplier['value_increase']
        }

    def _create_communication_strategy(self, strategy_type, category, engagement_level):
        """Create communication strategy for the segment."""

        base_strategy = {
            'email_frequency': 'Weekly',
            'push_notification_frequency': 'Bi-weekly',
            'sms_frequency': 'Monthly',
            'content_tone': 'Professional',
            'personalization_level': 'Medium'
        }

        # Adjust based on strategy type
        if strategy_type == 'Premium_Retention':
            base_strategy.update({
                'email_frequency': 'Bi-weekly',
                'push_notification_frequency': 'Weekly',
                'content_tone': 'Premium/Exclusive',
                'personalization_level': 'High',
                'dedicated_support': True,
                'proactive_outreach': True
            })
        elif strategy_type == 'Retention_Focus':
            base_strategy.update({
                'email_frequency': '2-3x per week',
                'push_notification_frequency': 'Daily',
                'content_tone': 'Helpful/Supportive',
                'personalization_level': 'High',
                'win_back_campaigns': True
            })
        elif strategy_type == 'Engagement_Amplification':
            base_strategy.update({
                'email_frequency': 'Daily',
                'push_notification_frequency': 'Multiple daily',
                'content_tone': 'Engaging/Fun',
                'personalization_level': 'Very High',
                'social_integration': True,
                'gamification_elements': True
            })

        # Category-specific adjustments
        if category in ['Mobile', 'Mobile Phone']:
            base_strategy['content_focus'] = 'Technical specifications and comparisons'
        elif category == 'Fashion':
            base_strategy['content_focus'] = 'Style inspiration and trends'
        elif category == 'Grocery':
            base_strategy['content_focus'] = 'Recipe ideas and nutritional information'

        return base_strategy

    def _create_cross_segment_insights(self):
        """Create insights that span across segments."""

        if not self.segment_strategies:
            return

        # Analyze patterns across segments
        high_risk_segments = [s for s in self.segment_strategies.values() if
                              s['churn_risk_level'] in ['Critical', 'High']]
        high_value_segments = [s for s in self.segment_strategies.values() if s['value_tier'] in ['Premium', 'High']]

        # Priority segments (high value + high risk)
        priority_segments = [s for s in high_risk_segments if s in high_value_segments]

        self.cross_segment_insights = {
            'total_segments': len(self.segment_strategies),
            'high_risk_segments': len(high_risk_segments),
            'high_value_segments': len(high_value_segments),
            'priority_segments': len(priority_segments),
            'total_customers': sum(s['segment_size'] for s in self.segment_strategies.values()),
            'high_risk_customers': sum(s['segment_size'] for s in high_risk_segments),
            'high_value_customers': sum(s['segment_size'] for s in high_value_segments),
            'priority_customers': sum(s['segment_size'] for s in priority_segments),
            'category_distribution': self._analyze_category_distribution(),
            'strategy_type_distribution': self._analyze_strategy_distribution()
        }

    def _analyze_category_distribution(self):
        """Analyze distribution of categories across segments."""
        categories = {}
        for segment in self.segment_strategies.values():
            category = segment['primary_category']
            if category not in categories:
                categories[category] = {'segments': 0, 'customers': 0, 'avg_churn': []}
            categories[category]['segments'] += 1
            categories[category]['customers'] += segment['segment_size']
            categories[category]['avg_churn'].append(segment['churn_rate'])

        # Calculate average churn per category
        for category in categories:
            categories[category]['avg_churn_rate'] = np.mean(categories[category]['avg_churn'])
            del categories[category]['avg_churn']

        return categories

    def _analyze_strategy_distribution(self):
        """Analyze distribution of strategy types."""
        strategies = {}
        for segment in self.segment_strategies.values():
            strategy_type = segment['strategy_type']
            if strategy_type not in strategies:
                strategies[strategy_type] = {'segments': 0, 'customers': 0, 'total_impact': 0}
            strategies[strategy_type]['segments'] += 1
            strategies[strategy_type]['customers'] += segment['segment_size']
            strategies[strategy_type]['total_impact'] += segment['expected_impact']['revenue_impact']

        return strategies

    def _estimate_strategies_impact(self):
        """Estimate overall business impact of all strategies."""

        total_revenue_impact = sum(s['expected_impact']['revenue_impact'] for s in self.segment_strategies.values())
        total_customers_retained = sum(
            s['expected_impact']['customers_retained'] for s in self.segment_strategies.values())
        total_customers = sum(s['segment_size'] for s in self.segment_strategies.values())

        # Implementation costs (estimated)
        implementation_costs = {
            'Premium_Retention': 50000,
            'Retention_Focus': 30000,
            'Growth_Premium': 40000,
            'Engagement_Amplification': 35000,
            'Standard_Growth': 15000
        }

        total_implementation_cost = sum(
            implementation_costs.get(s['strategy_type'], 20000) for s in self.segment_strategies.values()
        )

        self.impact_estimations = {
            'total_revenue_impact': total_revenue_impact,
            'total_customers_retained': total_customers_retained,
            'total_customers': total_customers,
            'retention_rate_improvement': total_customers_retained / total_customers if total_customers > 0 else 0,
            'total_implementation_cost': total_implementation_cost,
            'net_benefit': total_revenue_impact - total_implementation_cost,
            'roi_percentage': ((
                                           total_revenue_impact - total_implementation_cost) / total_implementation_cost * 100) if total_implementation_cost > 0 else 0,
            'payback_period_months': (
                        total_implementation_cost / (total_revenue_impact / 12)) if total_revenue_impact > 0 else 0
        }

    def generate_customer_recommendations(self, customer_data, churn_model, top_n_actions=8):
        """
        Generate specific recommendations for individual customers.

        Parameters:
        -----------
        customer_data : pd.DataFrame
            Individual customer data
        churn_model : ChurnPredictor
            Trained churn prediction model
        top_n_actions : int, default=8
            Number of top actions to recommend

        Returns:
        --------
        dict
            Personalized recommendations for the customer
        """
        if not self.is_strategies_created:
            raise ValueError("Strategies must be created first. Call create_personalization_strategies().")

        if len(customer_data) != 1:
            raise ValueError("This method handles one customer at a time.")

        # Get customer segment
        if 'CustomerSegment' not in customer_data.columns:
            raise ValueError("Customer data must include 'CustomerSegment' column.")

        customer_segment = customer_data['CustomerSegment'].iloc[0]
        customer_id = customer_data.get('CustomerID', ['Unknown']).iloc[0]

        # Get churn risk
        churn_prob, risk_category = churn_model.predict_churn_risk(customer_data)

        # Get strategy for this segment
        if customer_segment not in self.segment_strategies:
            raise ValueError(f"No strategy found for segment {customer_segment}")

        strategy = self.segment_strategies[customer_segment]

        # Create specific action plan
        actions = self._create_individual_action_plan(customer_data, strategy, churn_prob[0])

        # Generate recommendations
        recommendations = {
            'customer_id': customer_id,
            'segment': customer_segment,
            'segment_description': strategy['segment_description'],
            'churn_probability': float(churn_prob[0]),
            'risk_level': risk_category[0],
            'strategy_type': strategy['strategy_type'],
            'priority_level': strategy['priority_level'],
            'personalization_actions': actions[:top_n_actions],
            'communication_strategy': strategy['communication_strategy'],
            'expected_outcomes': self._estimate_individual_outcomes(customer_data, strategy, churn_prob[0]),
            'implementation_timeline': strategy['implementation_timeline'],
            'success_metrics': strategy['success_metrics'][:3]  # Top 3 metrics
        }

        return recommendations

    def _create_individual_action_plan(self, customer_data, strategy, churn_prob):
        """Create specific action plan for individual customer."""

        actions = []

        # High churn risk actions
        if churn_prob > 0.7:
            actions.extend([
                "🚨 URGENT: Trigger immediate retention campaign within 24 hours",
                "📞 Assign to customer success team for personal outreach",
                "💰 Send personalized discount offer (15-20% off next purchase)",
                "📋 Schedule immediate satisfaction survey and follow-up call",
                "⚡ Provide priority customer support and fast-track issue resolution"
            ])
        elif churn_prob > 0.4:
            actions.extend([
                "📧 Include in targeted retention email campaign",
                "🎁 Show loyalty program benefits and exclusive offers",
                "🏷️ Offer category-specific promotions and bundles",
                "📱 Send re-engagement push notifications with personalized content"
            ])

        # Value-based actions
        customer_value = customer_data.get('CustomerValueScore', [0]).iloc[0]
        if customer_value > 2.0:
            actions.extend([
                "⭐ Showcase premium product recommendations and exclusive items",
                "🎖️ Offer VIP customer benefits and priority service access",
                "🚀 Provide early access to new products and beta features",
                "🎯 Create personalized product bundles based on purchase history"
            ])
        elif customer_value > 1.0:
            actions.extend([
                "📈 Implement upselling strategies with relevant premium options",
                "🔄 Cross-sell complementary products from other categories"
            ])

        # Category-specific actions
        preferred_category = customer_data.get('PreferedOrderCat', ['Others']).iloc[0]
        category_actions = {
            'Mobile': [
                "📱 Show latest mobile device comparisons and reviews",
                "🔄 Offer trade-in programs with bonus incentives",
                "🛡️ Promote extended warranty and protection plans"
            ],
            'Mobile Phone': [
                "🔌 Recommend compatible accessories and charging solutions",
                "📦 Offer accessory bundles with discount pricing",
                "🎧 Suggest complementary audio and protective gear"
            ],
            'Fashion': [
                "👗 Display trending fashion items and seasonal collections",
                "✨ Send personalized style guide emails and lookbooks",
                "🎨 Offer styling consultations and personal shopper services"
            ],
            'Grocery': [
                "🥕 Promote bulk purchase discounts and family packs",
                "🍽️ Suggest recipe content and meal planning tools",
                "🚚 Offer subscription services for regular purchases"
            ],
            'Laptop & Accessory': [
                "💻 Highlight productivity tools and business solutions",
                "📊 Offer professional-grade equipment and bulk discounts",
                "🏢 Provide enterprise solutions and dedicated support"
            ],
            'Others': [
                "🔍 Enable advanced product discovery features",
                "🎯 Provide personalized recommendations across all categories",
                "🎁 Offer surprise and delight campaigns"
            ]
        }

        if preferred_category in category_actions:
            actions.extend(category_actions[preferred_category])

        # Engagement-based actions
        hours_on_app = customer_data.get('HourSpendOnApp', [0]).iloc[0]
        if hours_on_app > 3:
            actions.extend([
                "🎮 Enable advanced app features and gamification elements",
                "🏆 Offer power-user rewards and recognition programs",
                "👥 Invite to exclusive community forums and beta programs"
            ])
        elif hours_on_app < 2:
            actions.extend([
                "📚 Send app usage tips and feature highlight tutorials",
                "🎯 Create guided onboarding for better app experience",
                "📲 Implement gentle engagement nudges and helpful notifications"
            ])

        # Satisfaction-based actions
        satisfaction = customer_data.get('SatisfactionScore', [3]).iloc[0]
        if satisfaction <= 2:
            actions.extend([
                "🔧 Implement proactive customer service and issue resolution",
                "📞 Schedule satisfaction improvement consultation",
                "🎯 Focus on service quality improvements and personalized attention"
            ])
        elif satisfaction >= 4:
            actions.extend([
                "⭐ Leverage as potential brand advocate and referral source",
                "📣 Invite to provide testimonials and success stories",
                "🎁 Offer referral bonuses and advocacy rewards"
            ])

        # Complaint handling
        has_complaints = customer_data.get('Complain', [0]).iloc[0] == 1
        if has_complaints:
            actions.extend([
                "🛠️ Priority review of previous complaints and resolution status",
                "💬 Proactive follow-up on complaint resolution satisfaction",
                "🎯 Implement service recovery and rebuilding trust initiatives"
            ])

        return list(dict.fromkeys(actions))  # Remove duplicates while preserving order

    def _estimate_individual_outcomes(self, customer_data, strategy, churn_prob):
        """Estimate expected outcomes for individual customer."""

        base_value = customer_data.get('CustomerValueScore', [1.0]).iloc[0] * 1200  # Assume $1200 base value

        outcomes = {
            'churn_risk_reduction': 'Medium',
            'expected_engagement_increase': '15-25%',
            'expected_value_increase': '10-20%',
            'timeline_to_impact': '2-4 weeks'
        }

        # Adjust based on churn probability
        if churn_prob > 0.7:
            outcomes.update({
                'churn_risk_reduction': 'High Priority - 30-40% reduction expected',
                'expected_engagement_increase': '25-40%',
                'timeline_to_impact': '1-2 weeks'
            })
        elif churn_prob < 0.3:
            outcomes.update({
                'churn_risk_reduction': 'Maintenance - Focus on growth',
                'expected_engagement_increase': '10-15%',
                'expected_value_increase': '20-30%'
            })

        # Adjust based on strategy type
        if strategy['strategy_type'] == 'Premium_Retention':
            outcomes['expected_value_increase'] = '25-35%'
        elif strategy['strategy_type'] == 'Growth_Premium':
            outcomes['expected_value_increase'] = '30-40%'

        return outcomes

    def generate_sample_recommendations(self, customer_data, churn_model, n_samples_per_segment=2):
        """
        Generate sample recommendations for different customer types.

        Parameters:
        -----------
        customer_data : pd.DataFrame
            Customer data with segments
        churn_model : ChurnPredictor
            Trained churn prediction model
        n_samples_per_segment : int, default=2
            Number of sample customers per segment

        Returns:
        --------
        list
            Sample customer recommendations
        """
        if not self.is_strategies_created:
            raise ValueError("Strategies must be created first. Call create_personalization_strategies().")

        sample_recommendations = []

        for segment_id in self.segment_strategies.keys():
            # Get sample customers from this segment
            segment_customers = customer_data[customer_data['CustomerSegment'] == segment_id]

            if len(segment_customers) == 0:
                continue

            # Sample customers (mix of different risk levels if possible)
            n_samples = min(n_samples_per_segment, len(segment_customers))
            sample_customers = segment_customers.sample(n=n_samples, random_state=42)

            for _, customer in sample_customers.iterrows():
                customer_df = pd.DataFrame([customer])
                recommendation = self.generate_customer_recommendations(customer_df, churn_model)
                sample_recommendations.append(recommendation)

        return sample_recommendations

    def get_strategy_summary(self):
        """
        Get strategy summary as a DataFrame.

        Returns:
        --------
        pd.DataFrame
            Summary of all personalization strategies
        """
        if not self.is_strategies_created:
            raise ValueError("Strategies must be created first. Call create_personalization_strategies().")

        summary_data = []
        for seg_id, strategy in self.segment_strategies.items():
            summary_data.append({
                'Segment': seg_id,
                'Description': strategy['segment_description'],
                'Strategy_Type': strategy['strategy_type'],
                'Priority': strategy['priority_level'],
                'Size': f"{strategy['segment_size']:,}",
                'Percentage': f"{strategy['segment_percentage']:.1f}%",
                'Churn_Risk': strategy['churn_risk_level'],
                'Value_Tier': strategy['value_tier'],
                'Category': strategy['primary_category'],
                'Expected_Revenue_Impact': f"${strategy['expected_impact']['revenue_impact']:,.0f}",
                'Customers_Retained': strategy['expected_impact']['customers_retained'],
                'Implementation_Timeline': strategy['implementation_timeline']
            })

        return pd.DataFrame(summary_data)

    def plot_strategy_overview(self, figsize=(15, 10)):
        """
        Create comprehensive visualization of personalization strategies.

        Parameters:
        -----------
        figsize : tuple, default=(15, 10)
            Figure size for the plots
        """
        if not self.is_strategies_created:
            raise ValueError("Strategies must be created first. Call create_personalization_strategies().")

        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # Extract data for plotting
        segments = list(self.segment_strategies.keys())
        strategy_types = [s['strategy_type'] for s in self.segment_strategies.values()]
        segment_sizes = [s['segment_size'] for s in self.segment_strategies.values()]
        churn_rates = [s['churn_rate'] * 100 for s in self.segment_strategies.values()]
        value_scores = [s['avg_value_score'] for s in self.segment_strategies.values()]
        revenue_impacts = [s['expected_impact']['revenue_impact'] for s in self.segment_strategies.values()]
        priority_levels = [s['priority_level'] for s in self.segment_strategies.values()]

        # Plot 1: Strategy Type Distribution
        strategy_type_counts = pd.Series(strategy_types).value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(strategy_type_counts)))

        axes[0, 0].pie(strategy_type_counts.values, labels=strategy_type_counts.index,
                       autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0, 0].set_title('Strategy Type Distribution')

        # Plot 2: Revenue Impact by Segment
        axes[0, 1].bar(segments, [r / 1000 for r in revenue_impacts], color='skyblue', alpha=0.7)
        axes[0, 1].set_title('Expected Revenue Impact by Segment (K$)')
        axes[0, 1].set_xlabel('Segment ID')
        axes[0, 1].set_ylabel('Revenue Impact ($K)')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Plot 3: Segment Size vs Churn Rate
        colors_risk = ['red' if rate > 25 else 'orange' if rate > 15 else 'green' for rate in churn_rates]
        scatter = axes[0, 2].scatter(segment_sizes, churn_rates, c=colors_risk, s=100, alpha=0.7)
        axes[0, 2].set_title('Segment Size vs Churn Rate')
        axes[0, 2].set_xlabel('Segment Size (customers)')
        axes[0, 2].set_ylabel('Churn Rate (%)')

        # Add segment labels
        for i, seg in enumerate(segments):
            axes[0, 2].annotate(f'S{seg}', (segment_sizes[i], churn_rates[i]),
                                xytext=(5, 5), textcoords='offset points', fontsize=9)

        # Plot 4: Priority Level Distribution
        priority_counts = pd.Series(priority_levels).value_counts()
        priority_order = ['Critical', 'High', 'Medium-High', 'Medium']
        ordered_priorities = [priority_counts.get(p, 0) for p in priority_order]
        priority_colors = ['red', 'orange', 'yellow', 'lightblue']

        axes[1, 0].bar(priority_order, ordered_priorities, color=priority_colors, alpha=0.7)
        axes[1, 0].set_title('Segment Priority Distribution')
        axes[1, 0].set_xlabel('Priority Level')
        axes[1, 0].set_ylabel('Number of Segments')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Plot 5: Value Score vs Revenue Impact
        axes[1, 1].scatter(value_scores, [r / 1000 for r in revenue_impacts],
                           c=churn_rates, cmap='RdYlGn_r', s=segment_sizes, alpha=0.7)
        axes[1, 1].set_title('Value Score vs Revenue Impact\n(Size=Segment Size, Color=Churn Rate)')
        axes[1, 1].set_xlabel('Average Value Score')
        axes[1, 1].set_ylabel('Revenue Impact ($K)')

        # Add colorbar
        cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
        cbar.set_label('Churn Rate (%)')

        # Plot 6: Implementation Timeline Distribution
        timelines = [s['implementation_timeline'] for s in self.segment_strategies.values()]
        timeline_counts = pd.Series(timelines).value_counts()

        axes[1, 2].barh(range(len(timeline_counts)), timeline_counts.values, color='lightgreen', alpha=0.7)
        axes[1, 2].set_yticks(range(len(timeline_counts)))
        axes[1, 2].set_yticklabels(timeline_counts.index)
        axes[1, 2].set_title('Implementation Timeline Distribution')
        axes[1, 2].set_xlabel('Number of Segments')

        plt.tight_layout()
        return fig

    def save_strategies(self, file_path='strategies/personalization_strategies.json'):
        """
        Save personalization strategies to JSON file.

        Parameters:
        -----------
        file_path : str
            Path where to save the strategies
        """
        if not self.is_strategies_created:
            raise ValueError("Strategies must be created first. Call create_personalization_strategies().")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Prepare data for JSON serialization
        save_data = {
            'strategies': self.segment_strategies,
            'cross_segment_insights': getattr(self, 'cross_segment_insights', {}),
            'impact_estimations': self.impact_estimations,
            'category_customizations': self.category_customizations,
            'creation_timestamp': pd.Timestamp.now().isoformat()
        }

        # Convert numpy types to native Python types for JSON serialization
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

        print(f"✅ Personalization strategies saved to: {file_path}")

    def load_strategies(self, file_path):
        """
        Load personalization strategies from JSON file.

        Parameters:
        -----------
        file_path : str
            Path to the saved strategies file
        """
        try:
            with open(file_path, 'r') as f:
                load_data = json.load(f)

            # Restore strategies
            self.segment_strategies = load_data['strategies']
            self.cross_segment_insights = load_data.get('cross_segment_insights', {})
            self.impact_estimations = load_data.get('impact_estimations', {})
            self.category_customizations = load_data.get('category_customizations', {})

            self.is_strategies_created = True
            print(f"✅ Personalization strategies loaded from: {file_path}")
            print(f"📊 {len(self.segment_strategies)} strategies loaded")

        except FileNotFoundError:
            raise FileNotFoundError(f"Strategies file not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error loading strategies: {str(e)}")

    def generate_implementation_guide(self):
        """
        Generate comprehensive implementation guide for the personalization strategies.

        Returns:
        --------
        dict
            Detailed implementation guide with phases, resources, and timelines
        """
        if not self.is_strategies_created:
            raise ValueError("Strategies must be created first. Call create_personalization_strategies().")

        # Analyze strategy complexity and requirements
        tech_requirements = set()
        resource_requirements = set()
        total_customers = sum(s['segment_size'] for s in self.segment_strategies.values())

        for strategy in self.segment_strategies.values():
            tech_requirements.update(strategy.get('tech_requirements', []))
            resource_requirements.update(strategy.get('resource_requirements', []))

        implementation_guide = {
            'executive_summary': {
                'total_segments': len(self.segment_strategies),
                'total_customers': total_customers,
                'total_revenue_impact': self.impact_estimations.get('total_revenue_impact', 0),
                'total_implementation_cost': self.impact_estimations.get('total_implementation_cost', 0),
                'expected_roi': f"{self.impact_estimations.get('roi_percentage', 0):.0f}%",
                'payback_period': f"{self.impact_estimations.get('payback_period_months', 0):.1f} months"
            },

            'implementation_phases': {
                'Phase_1_Foundation': {
                    'duration': '4-6 weeks',
                    'priority': 'Critical',
                    'description': 'Set up core infrastructure and high-priority segments',
                    'tasks': [
                        'Deploy data pipeline and real-time analytics',
                        'Set up A/B testing framework',
                        'Implement customer segmentation in production',
                        'Launch Premium_Retention strategies (highest ROI)',
                        'Set up monitoring and alerting systems'
                    ],
                    'segments_to_implement': [
                        seg_id for seg_id, strategy in self.segment_strategies.items()
                        if strategy['priority_level'] == 'Critical'
                    ],
                    'expected_impact': sum(
                        s['expected_impact']['revenue_impact']
                        for s in self.segment_strategies.values()
                        if s['priority_level'] == 'Critical'
                    ),
                    'resources_needed': ['Data engineering team', 'ML engineers', 'DevOps', 'Customer success']
                },

                'Phase_2_Expansion': {
                    'duration': '6-8 weeks',
                    'priority': 'High',
                    'description': 'Roll out to high-priority segments and retention focus',
                    'tasks': [
                        'Implement Retention_Focus strategies',
                        'Deploy advanced personalization features',
                        'Launch cross-category recommendation engine',
                        'Integrate with email and mobile push systems',
                        'Begin Growth_Premium strategies'
                    ],
                    'segments_to_implement': [
                        seg_id for seg_id, strategy in self.segment_strategies.items()
                        if strategy['priority_level'] == 'High'
                    ],
                    'expected_impact': sum(
                        s['expected_impact']['revenue_impact']
                        for s in self.segment_strategies.values()
                        if s['priority_level'] == 'High'
                    ),
                    'resources_needed': ['Marketing automation', 'Content team', 'UX designers', 'Product managers']
                },

                'Phase_3_Optimization': {
                    'duration': '8-12 weeks',
                    'priority': 'Medium-High',
                    'description': 'Complete rollout and optimize performance',
                    'tasks': [
                        'Implement Engagement_Amplification strategies',
                        'Launch gamification and social features',
                        'Deploy Standard_Growth strategies',
                        'Optimize based on A/B test results',
                        'Scale successful strategies across segments'
                    ],
                    'segments_to_implement': [
                        seg_id for seg_id, strategy in self.segment_strategies.items()
                        if strategy['priority_level'] in ['Medium-High', 'Medium']
                    ],
                    'expected_impact': sum(
                        s['expected_impact']['revenue_impact']
                        for s in self.segment_strategies.values()
                        if strategy['priority_level'] in ['Medium-High', 'Medium']
                    ),
                    'resources_needed': ['Community management', 'Social media', 'Gamification platform', 'Analytics']
                }
            },

            'technology_requirements': {
                'core_platform': list(tech_requirements),
                'integrations_needed': [
                    'Email marketing platform (SendGrid, Mailchimp)',
                    'Push notification service (Firebase, OneSignal)',
                    'A/B testing platform (Optimizely, VWO)',
                    'Analytics platform (Google Analytics, Mixpanel)',
                    'Customer support system (Zendesk, Intercom)'
                ],
                'data_requirements': [
                    'Real-time customer behavior tracking',
                    'Purchase history and preferences',
                    'Engagement metrics and app usage',
                    'Satisfaction scores and feedback',
                    'Support interaction history'
                ]
            },

            'resource_requirements': {
                'team_composition': list(resource_requirements),
                'estimated_team_size': {
                    'Data Scientists': 2,
                    'ML Engineers': 2,
                    'Software Engineers': 3,
                    'Product Managers': 2,
                    'Marketing Specialists': 2,
                    'Customer Success': 3,
                    'UX/UI Designers': 1
                },
                'external_vendors': [
                    'Personalization platform (e.g., Dynamic Yield, Optimizely)',
                    'Customer data platform (e.g., Segment, mParticle)',
                    'Email service provider',
                    'Analytics and BI tools'
                ]
            },

            'success_metrics': {
                'immediate_metrics': [
                    'Churn rate reduction by segment',
                    'Customer engagement improvement',
                    'Email click-through rates',
                    'App usage and session duration'
                ],
                'medium_term_metrics': [
                    'Customer lifetime value increase',
                    'Revenue per customer growth',
                    'Cross-category purchase rates',
                    'Net Promoter Score improvement'
                ],
                'long_term_metrics': [
                    'Overall churn rate reduction',
                    'Total revenue impact',
                    'Market share growth',
                    'Customer acquisition cost reduction'
                ]
            },

            'risk_mitigation': {
                'technical_risks': [
                    'Gradual rollout with fallback mechanisms',
                    'Extensive A/B testing before full deployment',
                    'Real-time monitoring and alerting',
                    'Data backup and recovery procedures'
                ],
                'business_risks': [
                    'Start with lowest-risk, highest-impact segments',
                    'Continuous performance monitoring',
                    'Regular strategy review and adjustment',
                    'Customer feedback integration'
                ]
            },

            'budget_allocation': {
                'technology_costs': f"${self.impact_estimations.get('total_implementation_cost', 0) * 0.4:.0f}",
                'personnel_costs': f"${self.impact_estimations.get('total_implementation_cost', 0) * 0.5:.0f}",
                'marketing_costs': f"${self.impact_estimations.get('total_implementation_cost', 0) * 0.1:.0f}",
                'total_budget': f"${self.impact_estimations.get('total_implementation_cost', 0):.0f}",
                'expected_return': f"${self.impact_estimations.get('total_revenue_impact', 0):.0f}"
            }
        }

        return implementation_guide

    def create_executive_summary(self):
        """
        Create executive summary of personalization strategy.

        Returns:
        --------
        str
            Executive summary report
        """
        if not self.is_strategies_created:
            raise ValueError("Strategies must be created first. Call create_personalization_strategies().")

        total_customers = sum(s['segment_size'] for s in self.segment_strategies.values())
        total_revenue = self.impact_estimations.get('total_revenue_impact', 0)
        total_cost = self.impact_estimations.get('total_implementation_cost', 0)
        roi = self.impact_estimations.get('roi_percentage', 0)

        # Count high-priority segments
        critical_segments = sum(1 for s in self.segment_strategies.values() if s['priority_level'] == 'Critical')
        high_risk_customers = sum(s['segment_size'] for s in self.segment_strategies.values() if
                                  s['churn_risk_level'] in ['Critical', 'High'])

        summary = f"""
        DOHTEM PERSONALIZATION STRATEGY - EXECUTIVE SUMMARY
        ==================================================

        BUSINESS OPPORTUNITY:
        • Total Customer Base: {total_customers:,} customers across {len(self.segment_strategies)} distinct segments
        • High-Risk Customers: {high_risk_customers:,} customers requiring immediate attention
        • Critical Priority Segments: {critical_segments} segments with highest business impact

        FINANCIAL IMPACT:
        • Expected Annual Revenue Impact: ${total_revenue:,.0f}
        • Total Implementation Investment: ${total_cost:,.0f}
        • Return on Investment: {roi:.0f}%
        • Payback Period: {self.impact_estimations.get('payback_period_months', 0):.1f} months
        • Net Annual Benefit: ${total_revenue - total_cost:,.0f}

        KEY STRATEGIC INITIATIVES:
        • Premium Retention Program for high-value at-risk customers
        • Automated retention campaigns for medium-risk segments  
        • Growth strategies for loyal high-value customers
        • Engagement amplification for active user base

        IMPLEMENTATION APPROACH:
        • Phase 1 (4-6 weeks): Critical segments and core infrastructure
        • Phase 2 (6-8 weeks): High-priority segments and advanced features
        • Phase 3 (8-12 weeks): Complete rollout and optimization

        EXPECTED OUTCOMES:
        • 25-40% reduction in churn for high-risk segments
        • 15-30% increase in customer lifetime value
        • 20-35% improvement in customer engagement
        • 10-25% growth in cross-category purchases

        NEXT STEPS:
        1. Secure executive sponsorship and budget approval
        2. Assemble cross-functional implementation team
        3. Begin Phase 1 with critical segment strategies
        4. Establish success metrics and monitoring framework

        This personalization strategy represents a significant opportunity to transform 
        customer relationships and drive sustainable revenue growth through data-driven, 
        segment-specific customer experiences.
        """

        return summary

    def demonstrate_personalization_engine(self):
        """Demonstration function for the personalization engine module."""
        print("🎯 Personalization Engine Module Demo")
        print("=" * 50)

        # Create mock segmentation and churn models with minimal data
        class MockSegmentationModel:
            def __init__(self):
                self.segment_profiles = {
                    0: {
                        'size': 1200,
                        'percentage': 21.3,
                        'churn_rate': 0.28,
                        'avg_satisfaction': 2.1,
                        'avg_orders': 1.2,
                        'avg_value_score': 1.8,
                        'avg_engagement': 1.5,
                        'top_categories': 'Mobile Phone',
                        'description': 'High-Risk Mobile Phone Customers'
                    },
                    1: {
                        'size': 850,
                        'percentage': 15.1,
                        'churn_rate': 0.05,
                        'avg_satisfaction': 4.2,
                        'avg_orders': 6.8,
                        'avg_value_score': 3.2,
                        'avg_engagement': 4.1,
                        'top_categories': 'Grocery',
                        'description': 'Premium Loyal Grocery Customers'
                    },
                    2: {
                        'size': 980,
                        'percentage': 17.4,
                        'churn_rate': 0.12,
                        'avg_satisfaction': 3.8,
                        'avg_orders': 3.5,
                        'avg_value_score': 2.1,
                        'avg_engagement': 2.8,
                        'top_categories': 'Fashion',
                        'description': 'Engaged Fashion Customers'
                    }
                }

        class MockChurnModel:
            def predict_churn_risk(self, data):
                # Mock prediction based on some simple rules
                probabilities = np.random.beta(2, 5, len(data))  # Skew toward lower probabilities
                risk_categories = np.where(
                    probabilities >= 0.7, 'High Risk',
                    np.where(probabilities >= 0.4, 'Medium Risk', 'Low Risk')
                )
                return probabilities, risk_categories

        try:
            # Initialize personalization engine
            engine = PersonalizationEngine()

            # Create mock models
            mock_segmentation = MockSegmentationModel()
            mock_churn_model = MockChurnModel()

            # Create personalization strategies
            print("\n🎯 Creating personalization strategies...")
            strategies = engine.create_personalization_strategies(mock_segmentation, mock_churn_model)

            print(f"✅ Created {len(strategies)} personalization strategies")

            # Display strategy summary
            summary_df = engine.get_strategy_summary()
            print("\n📋 Strategy Summary:")
            print(summary_df.to_string(index=False))

            # Generate implementation guide
            print("\n📚 Generating implementation guide...")
            implementation_guide = engine.generate_implementation_guide()

            print(
                f"💰 Total Expected Revenue Impact: ${implementation_guide['executive_summary']['total_revenue_impact']:,.0f}")
            print(f"🎯 Expected ROI: {implementation_guide['executive_summary']['expected_roi']}")

            # Create executive summary
            exec_summary = engine.create_executive_summary()
            print(f"\n📊 Executive Summary Created ({len(exec_summary.split())} words)")

            # Test individual customer recommendations
            print("\n👤 Testing individual customer recommendations...")

            # Create sample customer data
            sample_customer = pd.DataFrame([{
                'CustomerID': 'CUST_001',
                'CustomerSegment': 0,  # High-risk segment
                'CustomerValueScore': 2.5,
                'PreferedOrderCat': 'Mobile Phone',
                'HourSpendOnApp': 2.0,
                'SatisfactionScore': 2,
                'Complain': 1
            }])

            recommendation = engine.generate_customer_recommendations(sample_customer, mock_churn_model)

            print(f"   Customer: {recommendation['customer_id']}")
            print(f"   Risk Level: {recommendation['risk_level']}")
            print(f"   Strategy: {recommendation['strategy_type']}")
            print(f"   Top 3 Actions:")
            for i, action in enumerate(recommendation['personalization_actions'][:3], 1):
                print(f"     {i}. {action}")

            return engine, strategies, implementation_guide

        except Exception as e:
            print(f"❌ Demo failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None, None

    if __name__ == "__main__":
        # Run demonstration
        engine, strategies, guide = demonstrate_personalization_engine()

        if engine is not None:
            print(f"\n🎉 Personalization Engine ready for production!")
            print(f"🎯 {len(strategies)} strategies developed")
            print(f"💰 Expected ROI: {guide['executive_summary']['expected_roi']}")
            print(f"⏱️  Implementation timeline: 16-26 weeks total")

