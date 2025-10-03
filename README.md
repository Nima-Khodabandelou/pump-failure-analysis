# Hydraulic Pump Failure Analysis

## Project Overview
This project demonstrates Exploratory Data Analysis (EDA) applied to mechanical engineering reliability data. It analyzes synthetic data representing hydraulic pump failures to identify root causes and provide actionable recommendations.

## Business Problem
A manufacturing plant is experiencing premature failures of critical hydraulic pumps, causing costly unplanned downtime. Pumps designed for 10,000-hour lifespan are failing as early as 6,000 hours.

## Methodology
- **Data Generation**: Synthetic dataset of 100 pumps (50 failed, 50 operational)
- **Univariate Analysis**: Distribution analysis of individual variables
- **Bivariate Analysis**: Relationship exploration between key factors
- **Multivariate Analysis**: Complex interaction identification
- **Statistical Validation**: Hypothesis testing and correlation analysis

## Key Findings
- **Primary Root Cause**: Fluid contamination causing bearing seizure
- **Secondary Root Cause**: High operating temperatures causing seal leaks
- **Contributing Factor**: Extended maintenance intervals

## Files
- `main.py` - Complete analysis code with visualizations
- `requirements.txt` - Python package dependencies
- `pump_failure_analysis_dataset.csv` - Generated synthetic dataset
- `README.md` - Project documentation

## Installation & Usage

1. **Clone the repository**:
```bash
git clone https://github.com/Nima-Khodabandelou/pump-failure-analysis.git

cd pump-failure-analysis
