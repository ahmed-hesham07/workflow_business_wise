import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import numpy as np
from scipy import stats
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import PageBreak
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import joblib
import logging
import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from ollama_client import OllamaClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('maintenance_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass

class MaintenanceAnalyzer:
    """A comprehensive analyzer for equipment maintenance data with AI-driven insights."""
    
    REQUIRED_COLUMNS = [
        'Equipment Name', 'Equipment ID', 'Criticality level',
        'Task Id', 'Task Description', 'Start date', 'End Date',
        'Duration', 'Maintenance cost'
    ]

    DATE_FORMAT = '%Y-%m-%d'

    def __init__(self, data_path: str, model_version: str = "v1", ollama_model: str = "llama2"):
        """
        Initialize the analyzer with the data file path.
        
        Args:
            data_path (str): Path to the CSV file containing maintenance data
            model_version (str): Version tag for AI models
            ollama_model (str): Name of the Ollama model to use
        """
        self.model_version = model_version
        self.models_dir = Path('maintenance_models') / model_version
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Ollama client
        self.ollama = OllamaClient(model=ollama_model)
        
        logger.info(f"Initializing MaintenanceAnalyzer with data from {data_path}")
        self._load_and_validate_data(data_path)
        self.csv_filename = Path(data_path).stem
        self._infer_and_map_columns()
        self._preprocess_data()

    def _load_and_validate_data(self, data_path: str) -> None:
        """
        Load and validate the input data.
        
        Args:
            data_path (str): Path to the CSV file
            
        Raises:
            DataValidationError: If data validation fails
        """
        try:
            # Read CSV with proper date parsing
            self.df = pd.read_csv(data_path)
            
            # Convert date columns after reading
            self.df['Start date'] = pd.to_datetime(self.df['Start date'], format=self.DATE_FORMAT)
            self.df['End Date'] = pd.to_datetime(self.df['End Date'], format=self.DATE_FORMAT)
            
        except Exception as e:
            raise DataValidationError(f"Error reading CSV file: {str(e)}")

        # Validate required columns
        missing_columns = [col for col in self.REQUIRED_COLUMNS if col not in self.df.columns]
        if missing_columns:
            raise DataValidationError(f"Missing required columns: {', '.join(missing_columns)}")

        # Validate data types
        try:
            pd.to_numeric(self.df['Maintenance cost'])
            pd.to_numeric(self.df['Duration'])
        except Exception as e:
            raise DataValidationError(f"Invalid data types in columns: {str(e)}")

        # Validate equipment data
        if len(self.df['Equipment Name'].unique()) == 0:
            raise DataValidationError("No equipment data found in the file")

        logger.info("Data validation completed successfully")

    def save_model_metadata(self) -> None:
        """Save model metadata and parameters."""
        metadata = {
            'version': self.model_version,
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(self.df),
            'n_features': len(self.df.columns),
            'column_mapping': self.column_mapping
        }
        
        with open(self.models_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)

    def _infer_and_map_columns(self):
        """Automatically infer and map columns based on content and patterns."""
        try:
            # No need for complex mapping since columns match exactly
            required_columns = {
                'Equipment Name': 'Equipment Name',
                'Equipment ID': 'Equipment ID',
                'Criticality level': 'Criticality level',
                'Task Id': 'Task Id',
                'Task Description': 'Task Description',
                'Start date': 'Start date',
                'End Date': 'End Date',
                'Duration': 'Duration',
                'Maintenance cost': 'Maintenance cost'
            }
            
            # Verify all required columns exist
            missing_columns = [col for col in required_columns.keys() if col not in self.df.columns]
            if missing_columns:
                raise DataValidationError(f"Missing required columns: {', '.join(missing_columns)}")
            
            # Keep column names as they are since they match exactly
            self.column_mapping = required_columns
            
            logger.info("Column mapping completed successfully")
        except Exception as e:
            logger.error(f"Error during column mapping: {str(e)}")
            raise DataValidationError("Failed to map columns correctly")

    def _preprocess_data(self):
        """Enhanced preprocessing with AI-driven feature engineering."""
        try:
            # Calculate derived features
            if 'Start date' in self.df.columns:
                self.df['month_year'] = self.df['Start date'].dt.to_period('M')
                self.df['Season'] = self.df['Start date'].dt.month.map({
                    12: 'Winter', 1: 'Winter', 2: 'Winter',
                    3: 'Spring', 4: 'Spring', 5: 'Spring',
                    6: 'Summer', 7: 'Summer', 8: 'Summer',
                    9: 'Fall', 10: 'Fall', 11: 'Fall'
                })
            
            if 'Maintenance cost' in self.df.columns and 'Duration' in self.df.columns:
                self.df['Cost per Day'] = self.df['Maintenance cost'] / self.df['Duration'].replace(0, 1)
            
            # AI-driven feature engineering
            self._engineer_ai_features()
            
            logger.info("Data preprocessing completed successfully")
        except Exception as e:
            logger.error(f"Error during data preprocessing: {str(e)}")
            raise DataValidationError("Failed to preprocess data correctly")

    def _engineer_ai_features(self):
        """Create AI-driven features for enhanced analysis."""
        try:
            # Prepare numeric features for clustering
            numeric_columns = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                numeric_data = self.df[numeric_columns].fillna(0)
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(numeric_data)
                
                # Equipment clustering
                kmeans = KMeans(n_clusters=min(5, len(self.df)), random_state=42)
                self.df['equipment_cluster'] = kmeans.fit_predict(scaled_data)
                
                # Anomaly detection
                iso_forest = IsolationForest(random_state=42, contamination=0.1)
                self.df['maintenance_anomaly'] = iso_forest.fit_predict(scaled_data)
                
                # Dimensionality reduction for pattern detection
                if scaled_data.shape[1] >= 2:
                    pca = PCA(n_components=2)
                    pca_result = pca.fit_transform(scaled_data)
                    self.df['maintenance_pattern_1'] = pca_result[:, 0]
                    self.df['maintenance_pattern_2'] = pca_result[:, 1]
                
                # Save models for future use
                joblib.dump(kmeans, os.path.join(self.models_dir, 'equipment_clusters.joblib'))
                joblib.dump(iso_forest, os.path.join(self.models_dir, 'anomaly_detector.joblib'))
                joblib.dump(pca, os.path.join(self.models_dir, 'pattern_detector.joblib'))
        except Exception as e:
            logger.error(f"Error during AI feature engineering: {str(e)}")
            # Continue with basic features if AI features fail
            self.df['equipment_cluster'] = 0
            self.df['maintenance_anomaly'] = 1
            self.df['maintenance_pattern_1'] = 0
            self.df['maintenance_pattern_2'] = 0

    def analyze_maintenance_patterns(self):
        """AI-driven analysis of maintenance patterns."""
        patterns = {
            'clusters': self.df.groupby('equipment_cluster').agg({
                'Maintenance cost': ['mean', 'sum'],
                'Duration': 'mean'
            }).round(2),
            'anomalies': self.df[self.df['maintenance_anomaly'] == -1],
            'patterns': self.df.groupby('equipment_cluster').agg({
                'maintenance_pattern_1': 'mean',
                'maintenance_pattern_2': 'mean'
            })
        }
        return patterns

    def generate_ai_insights(self):
        """Generate AI-driven insights and recommendations using both ML models and Ollama."""
        # Get traditional ML-based patterns
        patterns = self.analyze_maintenance_patterns()
        insights = []

        # Prepare data for Ollama
        data = {
            'total_cost': self.df['Maintenance cost'].sum(),
            'avg_cost': self.df['Maintenance cost'].mean(),
            'n_equipment': len(self.df['Equipment ID'].unique()),
            'n_tasks': len(self.df),
            'clusters': [
                {
                    'id': cluster,
                    'size': len(self.df[self.df['equipment_cluster'] == cluster]),
                    'avg_cost': self.df[self.df['equipment_cluster'] == cluster]['Maintenance cost'].mean()
                }
                for cluster in self.df['equipment_cluster'].unique()
            ],
            'anomalies': [
                {
                    'equipment': row['Equipment Name'],
                    'cost': row['Maintenance cost'],
                    'date': row['Start date'].strftime('%Y-%m-%d')
                }
                for _, row in patterns['anomalies'].iterrows()
            ],
            'patterns': patterns['patterns'].to_dict()
        }

        try:
            # Get Ollama insights
            ollama_results = self.ollama.generate_maintenance_insights(data)
            insights.extend(ollama_results['insights'])
            
            # Add traditional ML insights as fallback/supplement
            if not insights:
                insights = self._generate_traditional_insights(patterns)
            else:
                # Add any critical ML insights that Ollama might have missed
                ml_insights = self._generate_traditional_insights(patterns)
                critical_insights = [i for i in ml_insights if i['impact'] == 'High']
                insights.extend(critical_insights)
            
        except Exception as e:
            logger.error(f"Error getting Ollama insights: {str(e)}")
            insights = self._generate_traditional_insights(patterns)
        
        return insights

    def _generate_traditional_insights(self, patterns):
        """Generate insights using traditional ML methods as fallback."""
        insights = []
        
        # Cluster analysis insights
        cluster_stats = patterns['clusters']
        for cluster in cluster_stats.index:
            avg_cost = cluster_stats.loc[cluster, ('Maintenance cost', 'mean')]
            total_cost = cluster_stats.loc[cluster, ('Maintenance cost', 'sum')]
            cluster_size = len(self.df[self.df['equipment_cluster'] == cluster])
            
            insights.append({
                'category': 'Equipment Clusters',
                'finding': f'Cluster {cluster} contains {cluster_size} maintenance events',
                'details': f'Average cost: ${avg_cost:,.2f}, Total cost: ${total_cost:,.2f}',
                'impact': 'High' if avg_cost > self.df['Maintenance cost'].mean() else 'Medium',
                'action': 'Review maintenance practices for this equipment cluster'
            })

        # Anomaly detection insights
        anomalies = patterns['anomalies']
        if len(anomalies) > 0:
            total_anomaly_cost = anomalies['Maintenance cost'].sum()
            avg_anomaly_cost = anomalies['Maintenance cost'].mean()
            
            insights.append({
                'category': 'Maintenance Anomalies',
                'finding': f'Detected {len(anomalies)} unusual maintenance events',
                'details': f'Total anomaly cost: ${total_anomaly_cost:,.2f}, Average: ${avg_anomaly_cost:,.2f}',
                'impact': 'High',
                'action': 'Investigate these maintenance events for process improvements'
            })

        # Pattern analysis insights
        pattern_data = patterns['patterns']
        pattern_clusters = pattern_data.values
        if len(pattern_clusters) > 0:
            pattern_similarities = np.corrcoef(pattern_clusters)
            unique_patterns = len(np.unique(pattern_similarities.round(2)))
            
            insights.append({
                'category': 'Maintenance Patterns',
                'finding': f'Identified {unique_patterns} distinct maintenance patterns',
                'details': 'Equipment groups show different maintenance behaviors',
                'impact': 'Medium',
                'action': 'Optimize maintenance schedules based on identified patterns'
            })

        return insights

    def enhance_report_with_ai(self, story, styles):
        """Add AI-driven insights to the PDF report."""
        story.append(PageBreak())
        story.append(Paragraph('AI-Driven Insights', styles['Heading1']))
        
        insights = self.generate_ai_insights()
        for insight in insights:
            story.append(Paragraph(f"{insight['category']}", styles['Heading2']))
            story.append(Paragraph(f"Finding: {insight['finding']}", styles['Normal']))
            story.append(Paragraph(f"Details: {insight['details']}", styles['Normal']))
            story.append(Paragraph(f"Impact: {insight['impact']}", styles['Normal']))
            story.append(Paragraph(f"Recommended Action: {insight['action']}", styles['Normal']))
            story.append(Spacer(1, 12))

    def create_pdf_report(self, output_dir):
        """Create an enhanced professional PDF version of the report using reportlab."""
        # ... (existing PDF report code remains the same) ...
        
        # Add AI insights before recommendations
        self.enhance_report_with_ai(story, styles)
        
        # Continue with existing report generation
        # ... (rest of the method remains unchanged)

    def plot_ai_insights(self, save_path=None):
        """Generate AI-driven visualization of maintenance patterns."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        sns.set_style("whitegrid")
        
        # Plot 1: Equipment Clusters
        scatter1 = ax1.scatter(
            self.df['maintenance_pattern_1'],
            self.df['maintenance_pattern_2'],
            c=self.df['equipment_cluster'],
            cmap='viridis',
            alpha=0.6,
            s=100
        )
        ax1.set_title('Equipment Maintenance Clusters', fontsize=16, pad=20, fontweight='bold')
        ax1.set_xlabel('Pattern Component 1', fontsize=14)
        ax1.set_ylabel('Pattern Component 2', fontsize=14)
        legend1 = ax1.legend(*scatter1.legend_elements(), title="Clusters")
        ax1.add_artist(legend1)

        # Plot 2: Anomaly Detection
        scatter2 = ax2.scatter(
            self.df['maintenance_pattern_1'],
            self.df['maintenance_pattern_2'],
            c=self.df['maintenance_anomaly'],
            cmap='RdYlGn',
            alpha=0.6,
            s=100
        )
        ax2.set_title('Maintenance Anomalies', fontsize=16, pad=20, fontweight='bold')
        ax2.set_xlabel('Pattern Component 1', fontsize=14)
        ax2.set_ylabel('Pattern Component 2', fontsize=14)
        legend2 = ax2.legend(*scatter2.legend_elements(), title="Normal/Anomaly")
        ax2.add_artist(legend2)

        # Plot 3: Cluster Characteristics
        cluster_stats = self.df.groupby('equipment_cluster').agg({
            'Maintenance cost': ['mean', 'count']
        }).round(2)
        
        cluster_sizes = cluster_stats['Maintenance cost']['count']
        cluster_costs = cluster_stats['Maintenance cost']['mean']
        
        scatter3 = ax3.scatter(
            cluster_sizes,
            cluster_costs,
            s=200,
            alpha=0.6,
            c=cluster_stats.index,
            cmap='viridis'
        )
        ax3.set_title('Cluster Characteristics', fontsize=16, pad=20, fontweight='bold')
        ax3.set_xlabel('Number of Events', fontsize=14)
        ax3.set_ylabel('Average Cost ($)', fontsize=14)
        
        # Add cluster labels
        for i, (x, y) in enumerate(zip(cluster_sizes, cluster_costs)):
            ax3.annotate(f'Cluster {i}', (x, y), xytext=(5, 5), textcoords='offset points')

        # Plot 4: Pattern Evolution
        if 'Start date' in self.df.columns:
            timeline = self.df.sort_values('Start date').reset_index()
            scatter4 = ax4.scatter(
                range(len(timeline)),
                timeline['maintenance_pattern_1'],
                c=timeline['equipment_cluster'],
                cmap='viridis',
                alpha=0.6,
                s=100
            )
            ax4.set_title('Pattern Evolution Over Time', fontsize=16, pad=20, fontweight='bold')
            ax4.set_xlabel('Time (Event Sequence)', fontsize=14)
            ax4.set_ylabel('Pattern Strength', fontsize=14)
            legend4 = ax4.legend(*scatter4.legend_elements(), title="Clusters")
            ax4.add_artist(legend4)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_cost_distribution(self, save_path=None):
        """Plot the distribution of maintenance costs."""
        plt.figure(figsize=(12, 6))
        sns.histplot(data=self.df, x='Maintenance cost', hue='Criticality level', multiple="stack")
        plt.title('Distribution of Maintenance Costs by Criticality Level', fontsize=14, pad=20)
        plt.xlabel('Maintenance Cost ($)', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_temporal_trends(self, save_path=None):
        """Plot maintenance cost trends over time."""
        plt.figure(figsize=(12, 6))
        monthly_data = self.df.groupby('month_year')['Maintenance cost'].sum().reset_index()
        # Convert month_year to datetime for better plotting
        monthly_data['month_year'] = monthly_data['month_year'].astype(str).apply(lambda x: pd.to_datetime(x + '-01'))
        plt.plot(monthly_data['month_year'], monthly_data['Maintenance cost'], 
                marker='o', linestyle='-', linewidth=2)
        plt.title('Temporal Trends in Maintenance Costs', fontsize=14, pad=20)
        plt.xlabel('Month-Year', fontsize=12)
        plt.ylabel('Total Maintenance Cost ($)', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_seasonal_patterns(self, save_path=None):
        """Plot maintenance patterns by season."""
        # Ensure seasons are in correct order
        season_order = ['Winter', 'Spring', 'Summer', 'Fall']
        seasonal_data = self.df.groupby('Season')['Maintenance cost'].agg(['mean', 'count']).reset_index()
        seasonal_data['Season'] = pd.Categorical(seasonal_data['Season'], categories=season_order, ordered=True)
        seasonal_data = seasonal_data.sort_values('Season')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot average cost by season
        sns.barplot(data=seasonal_data, x='Season', y='mean', ax=ax1, order=season_order)
        ax1.set_title('Average Maintenance Cost by Season', fontsize=12)
        ax1.set_xlabel('Season', fontsize=10)
        ax1.set_ylabel('Average Cost ($)', fontsize=10)
        
        # Plot number of tasks by season
        sns.barplot(data=seasonal_data, x='Season', y='count', ax=ax2, order=season_order)
        ax2.set_title('Number of Maintenance Tasks by Season', fontsize=12)
        ax2.set_xlabel('Season', fontsize=10)
        ax2.set_ylabel('Number of Tasks', fontsize=10)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_reliability_metrics(self, save_path=None):
        """Plot equipment reliability metrics."""
        reliability_data = self.analyze_equipment_reliability()
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(reliability_data['Maintenance Frequency'], 
                            reliability_data['Average Cost per Day'],
                            c=reliability_data['Total Downtime'],
                            s=100, alpha=0.6, cmap='viridis')
        plt.colorbar(scatter, label='Total Downtime (days)')
        plt.title('Equipment Reliability Analysis', fontsize=14, pad=20)
        plt.xlabel('Maintenance Frequency', fontsize=12)
        plt.ylabel('Average Cost per Day ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_cost_efficiency(self, save_path=None):
        """Plot cost efficiency metrics."""
        efficiency_data = self.analyze_cost_efficiency()
        
        plt.figure(figsize=(12, 8))
        efficiency_data = efficiency_data.reset_index()
        scatter = plt.scatter(efficiency_data['Average Cost'], 
                            efficiency_data['Avg Cost/Day'],
                            c=efficiency_data['Criticality level'].astype('category').cat.codes,
                            s=100, alpha=0.6, cmap='Set3')
        plt.colorbar(scatter, label='Criticality Level')
        plt.title('Cost Efficiency Analysis', fontsize=14, pad=20)
        plt.xlabel('Average Maintenance Cost ($)', fontsize=12)
        plt.ylabel('Average Cost per Day ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def generate_unified_report(self, output_dir):
        """Generate enhanced HTML and PDF reports with AI insights."""
        # Add CSV filename to output directory
        output_dir = f"{output_dir}_{self.csv_filename}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate all analyses and plots
        equipment_analysis = self.equipment_cost_analysis()
        criticality_analysis = self.criticality_analysis()
        temporal_analysis = self.temporal_cost_analysis()
        task_analysis = self.task_type_analysis()
        seasonal_analysis = self.analyze_seasonal_patterns()
        reliability_analysis = self.analyze_equipment_reliability()
        efficiency_analysis = self.analyze_cost_efficiency()
        correlation_analysis = self.analyze_maintenance_correlations()
        
        # Generate standard plots
        self.plot_cost_distribution(os.path.join(output_dir, f'cost_distribution_{self.csv_filename}.png'))
        self.plot_temporal_trends(os.path.join(output_dir, f'temporal_trends_{self.csv_filename}.png'))
        self.plot_seasonal_patterns(os.path.join(output_dir, f'seasonal_patterns_{self.csv_filename}.png'))
        self.plot_reliability_metrics(os.path.join(output_dir, f'reliability_metrics_{self.csv_filename}.png'))
        self.plot_cost_efficiency(os.path.join(output_dir, f'cost_efficiency_{self.csv_filename}.png'))
        
        # Generate AI-driven plots
        self.plot_ai_insights(os.path.join(output_dir, f'ai_insights_{self.csv_filename}.png'))
        
        # Generate HTML report
        html_path = os.path.join(output_dir, f'comprehensive_report_{self.csv_filename}.html')
        with open(html_path, 'w') as f:
            # Add AI Insights section to HTML
            ai_insights = self.generate_ai_insights()
            
            # Write HTML header and styles
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Equipment Maintenance Analysis Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/chart.js@3.7.1/dist/chart.min.css">
    <style>
        /* ... existing styles ... */
        .ai-insight {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-left: 4px solid #007bff;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 4px;
        }
        .ai-insight.high-impact {
            border-left-color: #dc3545;
        }
        .ai-insight.medium-impact {
            border-left-color: #ffc107;
        }
        .pattern-card {
            background: white;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark fixed-top">
        <div class="container-fluid">
            <a class="navbar-brand" href="#">Equipment Maintenance Analysis</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav">
                    <li class="nav-item"><a class="nav-link" href="#overview">Overview</a></li>
                    <li class="nav-item"><a class="nav-link" href="#equipment">Equipment Analysis</a></li>
                    <li class="nav-item"><a class="nav-link" href="#reliability">Reliability</a></li>
                    <li class="nav-item"><a class="nav-link" href="#costs">Cost Analysis</a></li>
                    <li class="nav-item"><a class="nav-link" href="#ai-insights">AI Insights</a></li>
                    <li class="nav-item"><a class="nav-link" href="#recommendations">Recommendations</a></li>
                </ul>
            </div>
        </div>
    </nav>
""")

            # Add AI Insights section
            f.write("""
        <section id="ai-insights" class="section">
            <h2>AI-Driven Insights</h2>
            <div class="visualization">
                <img src="ai_insights_{}.png" alt="AI-Driven Analysis" class="img-fluid">
            </div>
""".format(self.csv_filename))

            # Add AI insights cards
            for insight in ai_insights:
                impact_class = 'high-impact' if insight['impact'] == 'High' else 'medium-impact'
                f.write(f"""
            <div class="ai-insight {impact_class}>
                <h4>{insight['category']}</h4>
                <p><strong>Finding:</strong> {insight['finding']}</p>
                <p><strong>Details:</strong> {insight['details']}</p>
                <p><strong>Recommended Action:</strong> {insight['action']}</p>
                <span class="badge bg-{'danger' if insight['impact']=='High' else 'warning'}">
                    {insight['impact']} Impact
                </span>
            </div>""")

            f.write("</section>")

            # Continue with existing sections
            # ... (rest of the HTML generation remains the same)

        # Generate PDF report
        pdf_path = self.create_pdf_report(output_dir)
        
        print(f"Enhanced reports generated successfully!\nHTML report: {html_path}\nPDF report: {pdf_path}")

    # Existing methods remain unchanged
    def equipment_cost_analysis(self):
        """Analyze maintenance costs by equipment type."""
        equipment_costs = self.df.groupby('Equipment Name').agg({
            'Maintenance cost': ['sum', 'mean', 'count'],
            'Duration': 'sum'
        }).round(2)
        
        equipment_costs.columns = ['Total Cost', 'Average Cost', 'Number of Tasks', 'Total Duration']
        return equipment_costs
    
    def criticality_analysis(self):
        """Analyze maintenance patterns by criticality level."""
        criticality_metrics = self.df.groupby('Criticality level').agg({
            'Maintenance cost': ['sum', 'mean'],
            'Duration': 'mean',
            'Equipment ID': 'nunique'
        }).round(2)
        
        criticality_metrics.columns = ['Total Cost', 'Average Cost', 'Average Duration', 'Unique Equipment Count']
        return criticality_metrics
    
    def temporal_cost_analysis(self):
        """Analyze maintenance costs over time."""
        monthly_costs = self.df.groupby('month_year')['Maintenance cost'].agg(['sum', 'count']).reset_index()
        monthly_costs.columns = ['Month', 'Total Cost', 'Number of Tasks']
        return monthly_costs
    
    def task_type_analysis(self):
        """Analyze maintenance tasks by type."""
        task_metrics = self.df.groupby('Task Description').agg({
            'Maintenance cost': ['sum', 'mean'],
            'Duration': 'mean',
            'Equipment ID': 'count'
        }).round(2)
        
        task_metrics.columns = ['Total Cost', 'Average Cost', 'Average Duration', 'Number of Tasks']
        return task_metrics
    
    def analyze_seasonal_patterns(self):
        """Analyze maintenance patterns by season."""
        seasonal_metrics = self.df.groupby('Season').agg({
            'Maintenance cost': ['sum', 'mean', 'count'],
            'Duration': 'mean'
        }).round(2)
        
        seasonal_metrics.columns = ['Total Cost', 'Average Cost', 'Number of Tasks', 'Average Duration']
        return seasonal_metrics
    
    def analyze_equipment_reliability(self):
        """Analyze equipment reliability based on maintenance frequency and costs."""
        reliability_metrics = self.df.groupby('Equipment Name').agg({
            'Task Id': 'count',
            'Maintenance cost': 'sum',
            'Duration': ['sum', 'mean'],
            'Cost per Day': 'mean'
        }).round(2)
        
        reliability_metrics.columns = ['Maintenance Frequency', 'Total Cost', 
                                     'Total Downtime', 'Average Duration', 
                                     'Average Cost per Day']
        
        # Calculate days between maintenance
        equipment_dates = self.df.groupby('Equipment ID').agg({
            'Start date': lambda x: np.mean(np.diff(sorted(x))).days 
            if len(x) > 1 else np.nan
        })
        
        reliability_metrics['Avg Days Between Maintenance'] = equipment_dates['Start date']
        return reliability_metrics
    
    def analyze_cost_efficiency(self):
        """Analyze cost efficiency of maintenance operations."""
        # Calculate cost efficiency metrics
        efficiency_metrics = self.df.groupby(['Equipment Name', 'Criticality level']).agg({
            'Maintenance cost': ['sum', 'mean', 'std'],
            'Duration': 'mean',
            'Cost per Day': ['mean', 'std']
        }).round(2)
        
        efficiency_metrics.columns = ['Total Cost', 'Average Cost', 'Cost StdDev',
                                    'Average Duration', 'Avg Cost/Day', 'Cost/Day StdDev']
        return efficiency_metrics
    
    def analyze_maintenance_correlations(self):
        """Analyze correlations between different maintenance metrics."""
        correlation_data = self.df[['Maintenance cost', 'Duration', 'Cost per Day']]
        return correlation_data.corr()

def main():
    """Main execution function with command line interface."""
    parser = argparse.ArgumentParser(description='Equipment Maintenance Analysis System')
    parser.add_argument('data_path', type=str, help='Path to the maintenance data CSV file')
    parser.add_argument('--output-dir', type=str, default='maintenance_analysis_results',
                      help='Directory for output files')
    parser.add_argument('--model-version', type=str, default='v1',
                      help='Version tag for AI models')
    parser.add_argument('--ollama-model', type=str, default='llama2',
                      help='Name of the Ollama model to use')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    try:
        # Initialize analyzer
        analyzer = MaintenanceAnalyzer(args.data_path, model_version=args.model_version, ollama_model=args.ollama_model)
        
        # Generate reports
        analyzer.generate_unified_report(args.output_dir)
        
        # Save model metadata
        analyzer.save_model_metadata()
        
        logger.info("Analysis completed successfully!")
        
    except DataValidationError as e:
        logger.error(f"Data validation error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        if args.debug:
            logger.exception("Detailed error information:")
        sys.exit(1)

if __name__ == "__main__":
    main()