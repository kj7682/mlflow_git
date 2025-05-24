# Wine Quality Prediction with MLflow

This project demonstrates training a machine learning model to predict wine quality using the Wine Quality dataset. It utilizes scikit-learn for model training and MLflow for experiment tracking, parameter logging, metric logging, and model management.

## Setup

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Code

The main script `__main__.py` trains an ElasticNet regression model.

1.  **To run the script with default parameters (alpha=0.5, l1_ratio=0.5):**
    ```bash
    python __main__.py
    ```

2.  **To specify `alpha` and `l1_ratio` values:**
    ```bash
    python __main__.py <alpha_value> <l1_ratio_value>
    ```
    For example:
    ```bash
    python __main__.py 0.7 0.3
    ```

## MLflow Integration

This project uses MLflow to log experiments, parameters, metrics, and models.

*   **Experiment Data:** All MLflow data is logged locally in the `mlruns` directory.
*   **Viewing Results:** To view the MLflow UI, navigate to the project's root directory in your terminal and run:
    ```bash
    mlflow ui
    ```
    Then, open your web browser to `http://localhost:5000` (or the URL shown in the terminal). This interface allows you to see the logged runs, compare parameters and metrics, and view saved model artifacts.
