# Vehicle CO2 Emissions Analysis Dashboard

## Project Overview
This repository contains a data analytics dashboard built with **Streamlit** to analyze vehicle CO2 emissions. The application processes data regarding vehicle characteristics—such as engine size, cylinders, fuel consumption, and vehicle class—to visualize trends and identify factors contributing to higher carbon footprints.

The project also incorporates unsupervised machine learning (K-Means Clustering) to group vehicles based on their performance and emission profiles.

## Dataset
The application uses the **CO2 Emission by Vehicles** dataset.
* **Source:** Data collected from the Government of Canada's official open data portal (or sourced via Kaggle).
* **Key Features:** Make, Model, Vehicle Class, Engine Size (L), Cylinders, Transmission, Fuel Type, Fuel Consumption (City/Hwy/Comb), and CO2 Emissions (g/km).

## Features

### 1. Data Processing
* Automated loading and cleaning of the `CO2 Emissions_Canada.csv` dataset.
* Standardization of column names for consistent access.
* Handling of missing or inconsistent data points.

### 2. Interactive Dashboard
* **Streamlit Interface:** A user-friendly web interface to explore the dataset.
* **Visualizations:** Interactive charts generated using **Plotly Express** and **Plotly Graph Objects** to display:
    * Correlations between engine size and CO2 emissions.
    * Fuel consumption trends across different vehicle classes.
    * Distribution of emissions by fuel type.

### 3. Machine Learning Integration
* **Clustering Analysis:** Utilizes **K-Means Clustering** (from Scikit-learn) to segment vehicles into distinct groups based on numerical features like engine size and fuel efficiency.
* **Data Scaling:** Implements `StandardScaler` to normalize data before clustering for accurate results.

## Installation

1.  Clone the repository.
2.  Install the required dependencies using the provided `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To launch the dashboard, navigate to the project directory in your terminal and run the following command:

```bash
streamlit run app.py
```

Technologies Used
Language: Python

Web Framework: Streamlit

Data Manipulation: Pandas, NumPy

Visualization: Plotly Express, Plotly Graph Objects

Machine Learning: Scikit-learn (KMeans, StandardScaler)
