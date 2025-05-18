# 🦠 COVID-19 Prediction App



An advanced dashboard for analyzing COVID-19 data and making predictions using a hybrid approach combining epidemiological modeling (SIR) with deep learning (LSTM).

## Features

- **Comprehensive Data Visualization**: Interactive charts showing historical COVID-19 trends
- **Advanced Modeling**:
  - Enhanced SIR (Susceptible-Infected-Recovered) epidemiological model with time-varying parameters
  - Deep Learning model using Bidirectional LSTM networks
  - Hybrid ensemble approach combining both models
- **Prediction Capabilities**:
  - Forecast future cases, deaths, and recoveries
  - Confidence intervals for predictions
  - Multiple ensemble strategies (weighted, dynamic, boosted)
- **Detailed Reporting**:
  - Key statistics and metrics
  - Growth rate analysis
  - Peak detection
  - Summary reports with recommendations

## Technologies Used

- Python 3.8+
- Streamlit (for web interface)
- TensorFlow/Keras (for deep learning)
- Scikit-learn (for data preprocessing)
- Pandas/Numpy (for data manipulation)
- Matplotlib/Seaborn (for visualization)
- Scipy (for differential equations and optimization)

## Data Sources

The application uses two main data sources:

1. **COVID-19 Historical Data**: Cleaned complete COVID-19 dataset containing confirmed cases, deaths, and recoveries by country
2. **Population Data**: World population data for 2020 to calculate per-capita metrics


---

## 🚀 How to Run the Project

### 1. Install Python 3.9.6

Download and install Python 3.9.6 from the official Python website:

👉 [Download Python 3.9.6](https://www.python.org/downloads/release/python-396/)

---

### 2. Install Required Libraries

Open **Command Prompt (CMD)** or **Terminal** in the project directory and run:

```bash
pip install -r requirements.txt
