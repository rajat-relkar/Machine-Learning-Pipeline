# Vehicle CO2 Emissions Prediction

## Project Overview
This project utilizes a `RandomForestRegressor` to predict CO2 emissions from vehicles based on features such as make, model, engine size, and fuel consumption. The model is trained on the `vehicle_emissions.csv` dataset and evaluates its performance using R2 Score, Root Mean Square Error (RMSE), and Mean Absolute Error (MAE).

## Dataset
The `vehicle_emissions.csv` dataset contains the following vehicle attributes:
- Make
- Model
- Vehicle Class
- Engine Size
- Cylinders
- Transmission
- Fuel Consumption in City (L/100 km)
- Fuel Consumption in City-Highway (L/100 km)
- Fuel Consumption Combined (L/100 km)
- Smog Level
- CO2 Emissions (target variable)

## Requirements
To run this project, install the following Python libraries:

```bash
pip install pandas numpy scikit-learn joblib
```

## Project Structure
- `vehicle_emissions.csv`: Dataset used for training and testing the model.
- `vehicle_emission_prediction.py`: Main script containing data preprocessing, model training, and evaluation logic.
- `vehicle_emission_pipeline.joblib`: Saved machine learning pipeline for reuse.
- `README.md`: This documentation file.

## Code Description
The `vehicle_emission_prediction.py` script executes these steps:
1. **Imports**: Loads libraries for data handling, preprocessing, modeling, and evaluation.
2. **Data Loading**: Reads the dataset and displays initial insights (e.g., head and info).
3. **Feature and Target Creation**: Splits data into features (X) and target (CO2_Emissions, y).
4. **Pipeline Creation**:
   - Numerical features (e.g., Engine_Size, Cylinders) are imputed with mean values and scaled using `StandardScaler`.
   - Categorical features (e.g., Make, Transmission) are imputed with the most frequent value and one-hot encoded using `OneHotEncoder`.
   - A `RandomForestRegressor` is integrated into the pipeline.
5. **Model Training and Prediction**: Splits data into training (70%) and testing (30%) sets, trains the model, and generates predictions.
6. **Model Evaluation**: Computes and displays R2 Score, RMSE, and MAE for the test set.
7. **Model Saving**: Saves the trained pipeline to `vehicle_emission_pipeline.joblib`.

## How to Run
1. Ensure all required libraries are installed (see Requirements).
2. Place `vehicle_emissions.csv` in the same directory as the script.
3. Execute the script:

```bash
python vehicle_emission_prediction.py
```

4. The script will print model performance metrics (R2 Score, RMSE, MAE).

## Results
Upon running the script, the model’s performance on the test set is displayed with:
- **R2 Score**: Measures the proportion of variance explained by the model.
- **Root Mean Square Error (RMSE)**: Indicates the average prediction error magnitude.
- **Mean Absolute Error (MAE)**: Represents the average absolute difference between predictions and actual values.

## Making Predictions
To predict CO2 emissions on new data:
1. Load the saved pipeline:

```python
import joblib
pipeline = joblib.load('vehicle_emission_pipeline.joblib')
```

2. Prepare your new data in the same format as `vehicle_emissions.csv` (same feature names and types, excluding CO2_Emissions).
3. Generate predictions:

```python
predictions = pipeline.predict(new_data)
```

The pipeline automatically handles preprocessing (imputation, scaling, encoding) for new data.

## Future Improvements
- Test alternative regression models (e.g., Gradient Boosting, XGBoost).
- Optimize `RandomForestRegressor` hyperparameters using grid search or random search.
- Enhance feature engineering by deriving new features or addressing outliers.
- Expand the dataset with additional vehicle samples or features.

## License
This project is licensed under the MIT License.

## Contact
For questions or feedback, please open an issue on this GitHub repository.