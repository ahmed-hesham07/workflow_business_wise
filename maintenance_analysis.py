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

class MaintenanceAnalyzer:
    def __init__(self, data_path):
        """Initialize the analyzer with the data file path."""
        self.df = pd.read_csv(data_path)
        self.csv_filename = os.path.splitext(os.path.basename(data_path))[0]
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Preprocess the data for analysis."""
        # Convert date columns to datetime
        self.df['Start date'] = pd.to_datetime(self.df['Start date'])
        self.df['End Date'] = pd.to_datetime(self.df['End Date'])
        
        # Calculate month and year for temporal analysis
        self.df['Month-Year'] = self.df['Start date'].dt.to_period('M')
        
        # Calculate cost per day
        self.df['Cost per Day'] = self.df['Maintenance cost'] / self.df['Duration']
        
        # Calculate season
        self.df['Season'] = self.df['Start date'].dt.month.map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
    
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
        monthly_costs = self.df.groupby('Month-Year')['Maintenance cost'].agg(['sum', 'count']).reset_index()
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
    
    def plot_cost_distribution(self, save_path=None):
        """Plot the distribution of maintenance costs by equipment type."""
        plt.figure(figsize=(16, 10))
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        # Create the box plot with larger figure size
        ax = sns.boxplot(data=self.df, x='Equipment Name', y='Maintenance cost')
        
        # Enhance the plot
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('Equipment Type', fontsize=14, labelpad=15)
        plt.ylabel('Maintenance Cost ($)', fontsize=14, labelpad=15)
        plt.title('Maintenance Cost Distribution by Equipment Type', fontsize=16, pad=20, fontweight='bold')
        
        # Add value labels on the plot
        medians = self.df.groupby('Equipment Name')['Maintenance cost'].median()
        for i, median in enumerate(medians):
            plt.text(i, median, f'${median:,.0f}', ha='center', va='bottom', fontsize=10)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_temporal_trends(self, save_path=None):
        """Plot maintenance costs over time."""
        temporal_data = self.temporal_cost_analysis()
        
        plt.figure(figsize=(16, 10))
        sns.set_style("whitegrid")
        
        # Create the line plot
        plt.plot(range(len(temporal_data)), temporal_data['Total Cost'], marker='o', linewidth=2, markersize=8)
        
        # Enhance the plot
        plt.xticks(range(len(temporal_data)), 
                  [str(m) for m in temporal_data['Month']], 
                  rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        
        plt.xlabel('Month', fontsize=14, labelpad=15)
        plt.ylabel('Total Cost ($)', fontsize=14, labelpad=15)
        plt.title('Maintenance Costs Over Time', fontsize=16, pad=20, fontweight='bold')
        
        # Add value labels
        for i, v in enumerate(temporal_data['Total Cost']):
            plt.text(i, v, f'${v:,.0f}', ha='center', va='bottom', fontsize=10)
        
        # Add grid for better readability
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_seasonal_patterns(self, save_path=None):
        """Plot seasonal maintenance patterns."""
        seasonal_data = self.analyze_seasonal_patterns()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        # Plot 1: Total cost by season
        bars1 = seasonal_data['Total Cost'].plot(kind='bar', ax=ax1)
        ax1.set_title('Total Maintenance Cost by Season', fontsize=16, pad=20, fontweight='bold')
        ax1.set_xlabel('Season', fontsize=14, labelpad=15)
        ax1.set_ylabel('Total Cost ($)', fontsize=14, labelpad=15)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        
        # Add value labels
        for i, v in enumerate(seasonal_data['Total Cost']):
            ax1.text(i, v, f'${v:,.0f}', ha='center', va='bottom', fontsize=10)
        
        # Plot 2: Number of tasks by season
        bars2 = seasonal_data['Number of Tasks'].plot(kind='bar', ax=ax2)
        ax2.set_title('Number of Maintenance Tasks by Season', fontsize=16, pad=20, fontweight='bold')
        ax2.set_xlabel('Season', fontsize=14, labelpad=15)
        ax2.set_ylabel('Number of Tasks', fontsize=14, labelpad=15)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        
        # Add value labels
        for i, v in enumerate(seasonal_data['Number of Tasks']):
            ax2.text(i, v, str(int(v)), ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_reliability_metrics(self, save_path=None):
        """Plot reliability metrics for equipment."""
        reliability_data = self.analyze_equipment_reliability()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        # Plot 1: Maintenance frequency vs Total cost
        scatter1 = ax1.scatter(reliability_data['Maintenance Frequency'], 
                             reliability_data['Total Cost'],
                             s=100, alpha=0.6)
        ax1.set_xlabel('Maintenance Frequency', fontsize=14, labelpad=15)
        ax1.set_ylabel('Total Cost ($)', fontsize=14, labelpad=15)
        ax1.set_title('Maintenance Frequency vs Total Cost', fontsize=16, pad=20, fontweight='bold')
        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 2: Average duration vs Cost per day
        scatter2 = ax2.scatter(reliability_data['Average Duration'],
                             reliability_data['Average Cost per Day'],
                             s=100, alpha=0.6)
        ax2.set_xlabel('Average Duration (days)', fontsize=14, labelpad=15)
        ax2.set_ylabel('Average Cost per Day ($)', fontsize=14, labelpad=15)
        ax2.set_title('Duration vs Daily Cost', fontsize=16, pad=20, fontweight='bold')
        ax2.tick_params(axis='both', which='major', labelsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 3: Total downtime by equipment
        reliability_data['Total Downtime'].sort_values().plot(kind='bar', ax=ax3)
        ax3.set_title('Total Downtime by Equipment', fontsize=16, pad=20, fontweight='bold')
        ax3.set_xlabel('Equipment', fontsize=14, labelpad=15)
        ax3.set_ylabel('Total Downtime (days)', fontsize=14, labelpad=15)
        ax3.tick_params(axis='both', which='major', labelsize=12)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 4: Days between maintenance
        reliability_data['Avg Days Between Maintenance'].sort_values().plot(kind='bar', ax=ax4)
        ax4.set_title('Average Days Between Maintenance', fontsize=16, pad=20, fontweight='bold')
        ax4.set_xlabel('Equipment', fontsize=14, labelpad=15)
        ax4.set_ylabel('Days', fontsize=14, labelpad=15)
        ax4.tick_params(axis='both', which='major', labelsize=12)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_cost_efficiency(self, save_path=None):
        """Plot cost efficiency metrics."""
        efficiency_data = self.analyze_cost_efficiency()
        
        plt.figure(figsize=(16, 12))
        sns.set_style("whitegrid")
        
        efficiency_pivot = efficiency_data.reset_index().pivot(
            index='Equipment Name', 
            columns='Criticality level', 
            values='Avg Cost/Day'
        )
        
        # Create heatmap with enhanced visibility
        ax = sns.heatmap(efficiency_pivot, 
                        annot=True, 
                        fmt='.0f', 
                        cmap='YlOrRd',
                        cbar_kws={'label': 'Average Cost per Day ($)'},
                        annot_kws={'size': 10})
        
        plt.title('Cost Efficiency by Equipment and Criticality Level', 
                 fontsize=16, pad=20, fontweight='bold')
        plt.xlabel('Criticality Level', fontsize=14, labelpad=15)
        plt.ylabel('Equipment Type', fontsize=14, labelpad=15)
        
        # Rotate axis labels for better readability
        plt.xticks(rotation=0, fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def calculate_kpis(self):
        """Calculate Key Performance Indicators."""
        kpis = {
            'total_cost': self.df['Maintenance cost'].sum(),
            'avg_cost_per_task': self.df['Maintenance cost'].mean(),
            'total_downtime': self.df['Duration'].sum(),
            'avg_task_duration': self.df['Duration'].mean(),
            'maintenance_efficiency': (self.df['Maintenance cost'].sum() / self.df['Duration'].sum()),
            'critical_tasks_ratio': len(self.df[self.df['Criticality level'] == 'High']) / len(self.df) * 100,
            'equipment_utilization': 100 - (self.df['Duration'].sum() / (len(self.df['Equipment ID'].unique()) * 365) * 100)
        }
        return kpis

    def analyze_trends(self):
        """Analyze maintenance trends and patterns."""
        trends = {
            'cost_trend': self.df.groupby(self.df['Start date'].dt.to_period('M'))['Maintenance cost'].mean().pct_change().mean(),
            'duration_trend': self.df.groupby(self.df['Start date'].dt.to_period('M'))['Duration'].mean().pct_change().mean(),
            'high_cost_equipment': self.df.groupby('Equipment Name')['Maintenance cost'].sum().nlargest(5),
            'frequent_maintenance': self.df.groupby('Equipment Name').size().nlargest(5)
        }
        return trends

    def generate_recommendations(self):
        """Generate concise, high-impact recommendations."""
        kpis = self.calculate_kpis()
        trends = self.analyze_trends()
        recommendations = []
        
        # Get key metrics
        high_cost_equipment = self.df.groupby('Equipment Name')['Maintenance cost'].sum().sort_values(ascending=False).head(3)
        equipment_reliability = self.analyze_equipment_reliability()
        most_frequent = equipment_reliability.sort_values('Maintenance Frequency', ascending=False).head(2)
        seasonal_patterns = self.analyze_seasonal_patterns()
        peak_season = seasonal_patterns['Number of Tasks'].idxmax()

        # 1. Cost Optimization (High Priority)
        if trends['cost_trend'] > 0:
            recommendations.append({
                'category': 'Cost Optimization',
                'recommendation': 'Implement predictive maintenance for high-cost equipment',
                'situation': f'Top 3 cost contributors: {", ".join(high_cost_equipment.index)}',
                'actions': [
                    'Deploy condition monitoring sensors',
                    'Establish predictive maintenance schedules',
                    'Review maintenance procedures'
                ],
                'impact': 'High',
                'savings': f'Potential savings: ${high_cost_equipment.sum() * 0.2:,.0f}'
            })

        # 2. Equipment Reliability (High Priority)
        recommendations.append({
            'category': 'Equipment Reliability',
            'recommendation': f'Upgrade or replace frequent maintenance equipment',
            'situation': f'Most problematic: {most_frequent.index[0]} ({most_frequent["Maintenance Frequency"].iloc[0]} interventions)',
            'actions': [
                'Evaluate replacement options',
                'Implement enhanced monitoring',
                'Review maintenance protocols'
            ],
            'impact': 'High',
            'savings': f'Estimated cost reduction: ${most_frequent["Total Cost"].iloc[0] * 0.4:,.0f}'
        })

        # 3. Seasonal Optimization (Medium Priority)
        recommendations.append({
            'category': 'Maintenance Scheduling',
            'recommendation': 'Optimize seasonal maintenance distribution',
            'situation': f'Peak activity in {peak_season}',
            'actions': [
                'Redistribute non-critical tasks',
                'Increase staff during peak periods',
                'Implement preventive measures'
            ],
            'impact': 'Medium',
            'savings': 'Workload optimization and reduced overtime costs'
        })

        return recommendations

    def create_pdf_report(self, output_dir):
        """Create an enhanced professional PDF version of the report using reportlab."""
        from reportlab.lib.units import inch, cm
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
        from reportlab.platypus import Frame, PageTemplate, BaseDocTemplate, NextPageTemplate
        from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER, TA_RIGHT
        
        # Update PDF path to include CSV filename
        pdf_path = os.path.join(output_dir, f'comprehensive_report_{self.csv_filename}.pdf')
        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#2c3e50')
        )
        
        heading1_style = ParagraphStyle(
            'Heading1',
            parent=styles['Heading1'],
            fontSize=18,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor('#2c3e50')
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceAfter=12
        )
        
        # Title
        story.append(Paragraph('Equipment Maintenance Analysis Report', title_style))
        story.append(Spacer(1, 20))
        
        # Generate analyses
        equipment_analysis = self.equipment_cost_analysis()
        criticality_analysis = self.criticality_analysis()
        seasonal_analysis = self.analyze_seasonal_patterns()
        recommendations = self.generate_recommendations()
        kpis = self.calculate_kpis()
        
        # Executive Summary
        story.append(Paragraph('Executive Summary', heading1_style))
        summary_text = f"""
        This comprehensive analysis covers maintenance operations from {self.df['Start date'].min().date()} 
        to {self.df['End Date'].max().date()}. Key findings include:
        <br/><br/>
        • Total Maintenance Cost: ${kpis['total_cost']:,.2f}<br/>
        • Average Cost per Task: ${kpis['avg_cost_per_task']:,.2f}<br/>
        • Total Downtime: {kpis['total_downtime']} days<br/>
        • Equipment Utilization: {kpis['equipment_utilization']:.1f}%<br/>
        • Critical Tasks Ratio: {kpis['critical_tasks_ratio']:.1f}%
        """
        story.append(Paragraph(summary_text, normal_style))
        story.append(PageBreak())
        
        # Equipment Analysis
        story.append(Paragraph('Equipment Analysis', heading1_style))
        img_path = os.path.join(output_dir, f'cost_distribution_{self.csv_filename}.png')
        if os.path.exists(img_path):
            story.append(Image(img_path, width=450, height=300))
        story.append(Spacer(1, 12))
        
        # Convert DataFrame to table
        data = [['Equipment', 'Total Cost ($)', 'Average Cost ($)', 'Tasks', 'Duration (days)']]
        for idx, row in equipment_analysis.iterrows():
            data.append([
                idx,
                f"{row['Total Cost']:,.2f}",
                f"{row['Average Cost']:,.2f}",
                str(row['Number of Tasks']),
                str(row['Total Duration'])
            ])
        
        # Create and style table
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
        story.append(PageBreak())
        
        # Add visualizations with proper spacing and correct filenames
        for img_name, title in [
            (f'reliability_metrics_{self.csv_filename}.png', 'Reliability Analysis'),
            (f'seasonal_patterns_{self.csv_filename}.png', 'Seasonal Patterns'),
            (f'cost_efficiency_{self.csv_filename}.png', 'Cost Efficiency Analysis')
        ]:
            story.append(Paragraph(title, heading1_style))
            img_path = os.path.join(output_dir, img_name)
            if os.path.exists(img_path):
                img = Image(img_path, width=450, height=300)
                story.append(img)
            story.append(Spacer(1, 20))
            story.append(PageBreak())
        
        # Recommendations Section (Updated for conciseness)
        story.append(Paragraph('Recommendations and Action Plan', heading1_style))
        
        # Add a brief introduction
        intro_text = """
        Based on the analysis, we have identified three key areas for improvement, 
        prioritized by their potential impact and return on investment:
        """
        story.append(Paragraph(intro_text, normal_style))
        story.append(Spacer(1, 12))
        
        # Add recommendations in a more compact format
        recommendations = self.generate_recommendations()
        for i, rec in enumerate(recommendations, 1):
            # Create a compact recommendation block
            rec_text = f"""
            <b>{i}. {rec['category']} ({rec['impact']} Impact)</b><br/>
            <b>Current Status:</b> {rec['situation']}<br/>
            <b>Action:</b> {rec['recommendation']}<br/>
            <b>Key Steps:</b> {' • '.join(rec['actions'])}<br/>
            <b>Expected Outcome:</b> {rec['savings']}
            """
            story.append(Paragraph(rec_text, normal_style))
            story.append(Spacer(1, 8))
        
        # Build the PDF
        doc.build(story)
        return pdf_path

    def generate_unified_report(self, output_dir):
        """Generate enhanced HTML and PDF reports."""
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
        
        # Generate plots with CSV filename
        self.plot_cost_distribution(os.path.join(output_dir, f'cost_distribution_{self.csv_filename}.png'))
        self.plot_temporal_trends(os.path.join(output_dir, f'temporal_trends_{self.csv_filename}.png'))
        self.plot_seasonal_patterns(os.path.join(output_dir, f'seasonal_patterns_{self.csv_filename}.png'))
        self.plot_reliability_metrics(os.path.join(output_dir, f'reliability_metrics_{self.csv_filename}.png'))
        self.plot_cost_efficiency(os.path.join(output_dir, f'cost_efficiency_{self.csv_filename}.png'))
        
        # Generate HTML report with CSV filename
        html_path = os.path.join(output_dir, f'comprehensive_report_{self.csv_filename}.html')
        with open(html_path, 'w') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Equipment Maintenance Analysis Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/chart.js@3.7.1/dist/chart.min.css">
    <style>
        :root {
            --primary-color: #2c3e50;
            --secondary-color: #34495e;
            --accent-color: #3498db;
            --background-color: #f8f9fa;
            --text-color: #2c3e50;
        }
        
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: var(--background-color);
            color: var(--text-color);
            line-height: 1.6;
        }
        
        .container {
            max-width: 1200px;
            margin: 40px auto;
            padding: 20px;
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: var(--primary-color);
            margin-bottom: 1rem;
        }
        
        .card {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            transition: transform 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
        }
        
        .section {
            margin-bottom: 40px;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .visualization {
            margin: 20px 0;
            padding: 20px;
            background: white;
            border-radius: 8px;
            text-align: center;
        }
        
        .visualization img {
            max-width: 100%;
            height: auto;
        }
        
        .table-wrapper {
            overflow-x: auto;
            margin: 20px 0;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }
        
        th {
            background-color: var(--secondary-color);
            color: white;
            padding: 12px;
            text-align: left;
        }
        
        td {
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }
        
        tr:hover {
            background-color: rgba(52, 152, 219, 0.1);
        }
        
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .kpi-card {
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .kpi-value {
            font-size: 24px;
            font-weight: bold;
            color: var(--accent-color);
            margin: 10px 0;
        }
        
        .recommendation {
            border-left: 4px solid var(--accent-color);
            padding: 15px;
            margin-bottom: 15px;
            background: white;
        }
        
        .high-priority {
            border-left-color: #e74c3c;
        }
        
        .medium-priority {
            border-left-color: #f39c12;
        }
        
        .low-priority {
            border-left-color: #2ecc71;
        }
        
        @media print {
            .container {
                width: 100%;
                max-width: none;
                margin: 0;
                padding: 20px;
            }
            
            .visualization {
                break-inside: avoid;
                page-break-inside: avoid;
            }
            
            .card {
                break-inside: avoid;
            }
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
                    <li class="nav-item"><a class="nav-link" href="#recommendations">Recommendations</a></li>
                </ul>
            </div>
        </div>
    </nav>

    <div class="container" style="margin-top: 60px;">
""")

            # Add Executive Summary
            kpis = self.calculate_kpis()
            f.write("""
        <section id="overview" class="section">
            <h1 class="text-center mb-4">Equipment Maintenance Analysis Report</h1>
            <h2>Executive Summary</h2>
            <div class="kpi-grid">
""")
            
            # Add KPI cards
            kpi_items = [
                ('Total Maintenance Cost', f"${kpis['total_cost']:,.2f}"),
                ('Average Cost per Task', f"${kpis['avg_cost_per_task']:,.2f}"),
                ('Total Downtime', f"{kpis['total_downtime']} days"),
                ('Average Task Duration', f"{kpis['avg_task_duration']:.1f} days"),
                ('Critical Tasks Ratio', f"{kpis['critical_tasks_ratio']:.1f}%"),
                ('Equipment Utilization', f"{kpis['equipment_utilization']:.1f}%")
            ]
            
            for title, value in kpi_items:
                f.write(f"""
                <div class="kpi-card">
                    <h3>{title}</h3>
                    <div class="kpi-value">{value}</div>
                </div>""")
            
            f.write("</div>")  # Close kpi-grid

            # Equipment Analysis Section
            f.write("""
        </section>
        <section id="equipment" class="section">
            <h2>Equipment Analysis</h2>
            <div class="visualization">
                <img src="cost_distribution.png" alt="Cost Distribution" class="img-fluid">
            </div>
            <div class="table-wrapper">
""")
            f.write(equipment_analysis.to_html(classes='table table-striped'))
            f.write("</div>")

            # Reliability Analysis Section
            f.write("""
        </section>
        <section id="reliability" class="section">
            <h2>Reliability Analysis</h2>
            <div class="visualization">
                <img src="reliability_metrics.png" alt="Reliability Metrics" class="img-fluid">
            </div>
            <div class="table-wrapper">
""")
            f.write(reliability_analysis.to_html(classes='table table-striped'))
            f.write("</div>")

            # Cost Analysis Section
            f.write("""
        </section>
        <section id="costs" class="section">
            <h2>Cost Efficiency Analysis</h2>
            <div class="visualization">
                <img src="cost_efficiency.png" alt="Cost Efficiency" class="img-fluid">
            </div>
            <div class="table-wrapper">
""")
            f.write(efficiency_analysis.to_html(classes='table table-striped'))
            f.write("</div>")

            # Recommendations Section
            f.write("""
        </section>
        <section id="recommendations" class="section">
            <h2>Recommendations</h2>
""")
            
            recommendations = self.generate_recommendations()
            for rec in recommendations:
                priority_class = f"{'high' if rec['impact']=='High' else 'medium' if rec['impact']=='Medium' else 'low'}-priority"
                f.write(f"""
            <div class="recommendation {priority_class}>
                <h4>{rec['category']}</h4>
                <p>{rec['recommendation']}</p>
                <span class="badge bg-{'danger' if rec['impact']=='High' else 'warning' if rec['impact']=='Medium' else 'success'}">
                    {rec['impact']} Priority
                </span>
            </div>""")

            # Close main container and add scripts
            f.write("""
        </section>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.7.1/dist/chart.min.js"></script>
    <script>
        // Smooth scrolling for navigation
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                document.querySelector(this.getAttribute('href')).scrollIntoView({
                    behavior: 'smooth'
                });
            });
        });
    </script>
</body>
</html>
""")
        
        # Create PDF version
        pdf_path = self.create_pdf_report(output_dir)
        print(f"Enhanced reports generated successfully!\nHTML report: {html_path}\nPDF report: {pdf_path}")

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = MaintenanceAnalyzer('sample.csv')
    
    # Generate enhanced reports
    analyzer.generate_unified_report('maintenance_analysis_results')