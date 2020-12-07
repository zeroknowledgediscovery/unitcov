# Prediction for next week's covid cases and deaths
In the `unitcov/forecast_pipeline` folder, run notebook in exactly the following order
1. Run `pipeline_data_gathering.ipynb`
2. Run `pipeline_GLM.ipynb`
3. Run `pipeline_forecast_case.ipynb` and `pipeline_forecast_death.ipynb`
Alternatively, one can run `forecast_for_next_week.ipynb` which is a combination of the steps above.
The longform data frame ready for submission to the covid19 hub will be in the `submission` folder.
