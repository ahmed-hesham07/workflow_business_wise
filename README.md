# Equipment Maintenance Analysis System

A comprehensive Python-based analysis system for industrial equipment maintenance data, providing detailed insights, visualizations, and actionable recommendations for maintenance optimization.

## Overview

This system analyzes equipment maintenance data to provide insights into costs, reliability, efficiency, and seasonal patterns. It generates both interactive HTML reports and professional PDF documents containing detailed analysis and recommendations.

## Features

- **Comprehensive Analysis**
  - Equipment cost analysis
  - Reliability metrics
  - Cost efficiency evaluation
  - Seasonal pattern analysis
  - Maintenance scheduling optimization
  - Critical equipment monitoring

- **Advanced Visualizations**
  - Cost distribution charts
  - Reliability metrics quadrant analysis
  - Cost efficiency heatmaps
  - Seasonal pattern comparisons
  - Temporal trend analysis

- **Interactive Reports**
  - Dynamic HTML reports with interactive elements
  - Professional PDF reports with executive summaries
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

1. Prepare your maintenance data in CSV format with the following columns:
   - Equipment Name
   - Equipment ID
   - Criticality level
   - Task Id
   - Task Description
   - Start date
   - End Date
   - Duration
   - Maintenance cost

2. Run the analysis:
   ```bash
   python maintenance_analysis.py
   ```

3. The system will generate a results folder and files named after your input CSV:
   - Results folder: `maintenance_analysis_results_[csvname]`
   - Interactive HTML report: `comprehensive_report_[csvname].html`
   - Professional PDF report: `comprehensive_report_[csvname].pdf`
   - Visualization files:
     - `cost_distribution_[csvname].png`
     - `cost_efficiency_[csvname].png`
     - `reliability_metrics_[csvname].png`
     - `seasonal_patterns_[csvname].png`
     - `temporal_trends_[csvname].png`

   For example, if your input file is `sample.csv`, the output will be:
   - Folder: `maintenance_analysis_results_sample`
   - Reports: `comprehensive_report_sample.html/pdf`
   - Images: `cost_distribution_sample.png`, etc.

## Output Files

### HTML Report
- Interactive web-based dashboard
- Navigation menu for easy section access
- Responsive design for all screen sizes
- Interactive data tables with sorting capabilities
- Visual KPI cards and metrics display

### PDF Report
- Professional executive summary
- Detailed equipment analysis
- High-quality visualizations
- Concise, actionable recommendations
- KPI summary and metrics
- One-page recommendations section

### Visualization Files
All visualization files are generated in high resolution (300 DPI) and include:
- Cost distribution analysis
- Cost efficiency heatmaps
- Reliability metrics quadrant
- Seasonal pattern analysis
- Temporal trend visualization

## Analysis Components

### 1. Equipment Cost Analysis
- Total and average maintenance costs
- Cost distribution by equipment type
- Cost trends over time

### 2. Reliability Analysis
- Maintenance frequency analysis
- Equipment downtime patterns
- Mean time between maintenance

### 3. Cost Efficiency Analysis
- Cost per maintenance day
- Efficiency metrics by equipment type
- Cost optimization opportunities

### 4. Seasonal Pattern Analysis
- Seasonal maintenance distribution
- Peak period identification
- Workload optimization recommendations

### 5. Recommendations
- Data-driven maintenance strategies
- Cost optimization opportunities
- Equipment replacement recommendations
- Scheduling optimization suggestions
- Inventory management improvements

## Customization

The analysis system can be customized by modifying:
- Visualization parameters in plotting functions
- Analysis thresholds in recommendation generation
- Report formatting in PDF and HTML generation
- KPI calculations in the analysis methods

## Contributing

Contributions are welcome! Please feel free to submit pull requests with improvements or bug fixes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
