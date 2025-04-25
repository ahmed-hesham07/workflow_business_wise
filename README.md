# Equipment Maintenance Analysis System

A comprehensive Python-based analysis system for industrial equipment maintenance data, providing detailed insights, visualizations, and actionable recommendations for maintenance optimization.

## Overview

This system analyzes equipment maintenance data to provide insights into costs, reliability, efficiency, and seasonal patterns. It leverages AI/ML techniques for advanced pattern detection and anomaly identification, generating both interactive HTML reports and professional PDF documents containing detailed analysis and recommendations.

## Features

### Core Analysis
- Equipment cost analysis
- Reliability metrics
- Cost efficiency evaluation
- Seasonal pattern analysis
- Maintenance scheduling optimization
- Critical equipment monitoring

### AI-Driven Features
- Automated pattern detection using clustering
- Anomaly detection in maintenance events
- Predictive maintenance insights
- Equipment behavior pattern analysis
- Cost optimization recommendations
- Dynamic data column mapping

### Advanced Visualizations
- Cost distribution charts
- Reliability metrics quadrant analysis
- Cost efficiency heatmaps
- Seasonal pattern comparisons
- Temporal trend analysis
- AI pattern visualization dashboard

### Interactive Reports
- Dynamic HTML reports with interactive elements
- Professional PDF reports with executive summaries
- AI-driven insights section
- Detailed recommendations with action plans
- Key Performance Indicators (KPIs)
- Cost-saving opportunities identification

## Requirements

- Python 3.8+
- Required packages are listed in requirements.txt

## Installation

1. Clone this repository or download the source code.

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your maintenance data in CSV format with the following required columns:
   - Equipment Name
   - Equipment ID
   - Criticality level
   - Task Id
   - Task Description
   - Start date
   - End Date
   - Duration
   - Maintenance cost

2. Run the analysis using the command-line interface:
   ```bash
   python maintenance_analysis.py data.csv [options]
   ```

   Options:
   - `--output-dir DIR`: Specify output directory (default: maintenance_analysis_results)
   - `--model-version VERSION`: Specify AI model version (default: v1)
   - `--debug`: Enable debug logging

3. The system will generate a results folder with the following structure:
   ```
   maintenance_analysis_results_[csvname]/
   ├── comprehensive_report_[csvname].html
   ├── comprehensive_report_[csvname].pdf
   ├── cost_distribution_[csvname].png
   ├── cost_efficiency_[csvname].png
   ├── reliability_metrics_[csvname].png
   ├── seasonal_patterns_[csvname].png
   ├── temporal_trends_[csvname].png
   └── ai_insights_[csvname].png
   ```

   Additionally, AI models and metadata are saved in:
   ```
   maintenance_models/[version]/
   ├── equipment_clusters.joblib
   ├── anomaly_detector.joblib
   ├── pattern_detector.joblib
   └── metadata.json
   ```

## Analysis Components

### 1. Equipment Cost Analysis
- Total and average maintenance costs
- Cost distribution by equipment type
- Cost trends over time
- AI-driven cost pattern detection

### 2. Reliability Analysis
- Maintenance frequency analysis
- Equipment downtime patterns
- Mean time between maintenance
- AI-based anomaly detection

### 3. Cost Efficiency Analysis
- Cost per maintenance day
- Efficiency metrics by equipment type
- Cost optimization opportunities
- AI-driven efficiency clustering

### 4. Seasonal Pattern Analysis
- Seasonal maintenance distribution
- Peak period identification
- Workload optimization recommendations
- Pattern evolution tracking

### 5. AI-Driven Insights
- Equipment clustering analysis
- Maintenance anomaly detection
- Pattern identification
- Predictive recommendations

## Error Handling

The system includes comprehensive error handling:
- Input data validation
- Required column checking
- Data type verification
- Automatic column mapping
- Detailed error logging

## Customization

The analysis system can be customized by:
- Modifying visualization parameters in plotting functions
- Adjusting AI model parameters
- Customizing report formatting
- Adding new analysis components
- Extending AI feature engineering

## Contributing

Contributions are welcome! Please feel free to submit pull requests with improvements or bug fixes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
