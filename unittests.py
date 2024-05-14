import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from data import ProcessDietaryChange, processWorkout, processDietary
from analysis import correlateDietToWorkout, predictPerformance, dietEffectiveness, AlignDataforNutrition, standardize_weights, nutritionAnalysis


#Only did unittests for my Data and Analysis files as they have direct and testable features while the visualizations require manual inspection 
#And the GUI files links everything together and there is minimal testing to be done.

class DataProcess(unittest.TestCase):
    def test_process_workout(self):
        """\nEnsure the function filters out zeroes"""
        workout_data = pd.DataFrame({ 'Exercise': ['Bench Press', 'Squat', 'Deadlift', 'Bench Press'], 'Weight (kg)': [100, 200, 300, 0], 
            'Sets': [1, 1, 1, 1],
            'Reps': [10, 10, 10, 10]
        })
        processed_data = processWorkout(workout_data)
        self.assertFalse((processed_data['Weight (kg)'] == 0).any())

    def test_process_dietary(self):
        """\nCheck if calories are categorized correctly into 'Calorie Category'."""
        # setup the initial data to test the function
        data = {'Calories': [1500, 2500, 3500]}
        dietary_df = pd.DataFrame(data)
        processed = processDietary(dietary_df.copy())
        
        # define what the right categories should look like
        expected_categories = ['Low', 'Medium', 'High']
        
        # make sure the processed data matches our expectations
        self.assertTrue((processed['Calorie Category'] == expected_categories).all(), msg="Calorie Category assignment failed")

    def test_process_dietary_change(self):
        """\nEnsure it correctly splits data before and after a given intervention date."""
        # setup the initial data to test the function
        workout_data = pd.DataFrame({
            'Date': pd.to_datetime(['2023-01-01', '2023-01-15', '2023-02-01']),
            'Sets': [5, 10, 15]
        })
        dietary_data = pd.DataFrame({
            'Date': pd.to_datetime(['2023-01-01', '2023-01-15', '2023-02-01']),
            'Calories': [2000, 2500, 3000]
        })
        
        # this is when the diet started changing
        intervention_date = '2023-01-20'
        
        # run our function to split the data
        before_data, after_data = ProcessDietaryChange(workout_data, dietary_data, intervention_date)
        
        # define what the dates should look like in each split
        expected_before_dates = pd.to_datetime(['2023-01-01', '2023-01-15'])
        expected_after_dates = pd.to_datetime(['2023-02-01'])

        # check if the data was split on the right dates
        self.assertTrue((before_data['Date'] == expected_before_dates).all(), msg="Data before the intervention date is incorrect")
        self.assertTrue((after_data['Date'] == expected_after_dates).all(), msg="Data after the intervention date is incorrect")
        
        # verify if we still have all the data after the split
        self.assertEqual(len(before_data) + len(after_data), len(workout_data), msg="Data length mismatch after splitting")


class Fitnessanalysis(unittest.TestCase):
    def test_correlate_diet_to_workout(self):
        """\nTest if the function computes the correlation matrix correctly between dietary and workout data."""
        # set up some dummy data to check if our correlation stuff works
        workout_data = pd.DataFrame({
            'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'Total Volume': [1000, 1500, 1200]
        })
        dietary_data = pd.DataFrame({
            'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'Calories': [2000, 2500, 2300],
            'Protein (g)': [50, 60, 55],
            'Carbs (g)': [300, 350, 320],
            'Fats (g)': [70, 80, 75]
        })

        # actually run the correlation matrix function
        correlation_matrix = correlateDietToWorkout(workout_data, dietary_data)

        # make sure the matrix is the right size and has the right stuff in it
        self.assertEqual(correlation_matrix.shape, (5, 5), msg="Correlation matrix shape is incorrect")
        expected_columns = ['Total Volume', 'Calories', 'Protein (g)', 'Carbs (g)', 'Fats (g)']
        for col in expected_columns:
            self.assertIn(col, correlation_matrix.columns, msg=f"{col} is missing in the correlation matrix")
        self.assertTrue(np.issubdtype(correlation_matrix.dtypes[0], np.number), msg="Correlation matrix values must be numeric")

    def test_predict_performance(self):
        """\nCheck if predictPerformance properly returns a LinearRegression model."""
        # setup some basic data
        workout_data = pd.DataFrame({
            'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'Total Volume': [1000, 1500, 1200]
        })
        dietary_data = pd.DataFrame({
            'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'Calories': [2000, 2500, 2300]
        })

        # ensure dates are datetime for the function to work right
        workout_data['Date'] = pd.to_datetime(workout_data['Date'])
        dietary_data['Date'] = pd.to_datetime(dietary_data['Date'])

        # let's see if we get a regression model back
        model = predictPerformance(workout_data, dietary_data)
        self.assertIsInstance(model, LinearRegression, msg="Function should return a LinearRegression model")

    def test_diet_effectiveness(self):
        """\nMake sure the dietEffectiveness function correctly computes the t-test between before and after data."""
        # prepare some data that's all nice and clean
        before_data = pd.DataFrame({
            'Total Volume': [100, 150, 200]
        })
        after_data = pd.DataFrame({
            'Total Volume': [110, 160, 210]
        })

        # running the t-test on this data
        results = dietEffectiveness(before_data, after_data)
        self.assertIsNotNone(results['T-statistic'], "T-statistic should not be None")
        self.assertIsNotNone(results['P-value'], "P-value should not be None")

    def test_with_nan_values(self):
        """\nEnsure proper handling of NaN values in dietEffectiveness function."""
        # setting up some data where there are NaNs to see how the function handles it
        before_data = pd.DataFrame({
            'Total Volume': [100, 150, None]
        })
        after_data = pd.DataFrame({
            'Total Volume': [None, None, None]
        })

        # testing what happens when the data isn't all there
        results = dietEffectiveness(before_data, after_data)
        self.assertEqual(results['T-statistic'], 'N/A', "T-statistic should be 'N/A' for empty data")
        self.assertEqual(results['P-value'], 'N/A', "P-value should be 'N/A' for empty data")

    def test_align_data_for_nutrition(self):
        """\nVerify that dietary data aligns correctly with the workout data based on the specified date shifts."""
        # here's our dietary and workout data
        dietary_data = pd.DataFrame({
            'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'Calories': [2000, 2500, 2300]
        })
        workout_data = pd.DataFrame({
            'Date': pd.to_datetime(['2023-01-03', '2023-01-04', '2023-01-05']),
            'Total Volume': [1000, 1500, 1200]
        })

        # align the data and make sure it works
        combined_data = AlignDataforNutrition(dietary_data, workout_data)
        print(combined_data)
        self.assertEqual(len(combined_data), 2, "Data should be aligned for 2 days only")

    
    def test_standardize_weights(self):
        """Check if the weights are standardized through the specific ratios"""
        self.data = pd.DataFrame({
            'Date': ['2023-01-01', '2023-01-01', '2023-01-01'],
            'Exercise': ['Bench Press', 'Squat', 'Deadlift'],
            'Weight (kg)': [90, 120, 150]
        })

        result = standardize_weights(self.data.copy())
        expected_weights_std = [30, 30, 30]  # 90/3, 120/4, 150/5
        

        self.assertEqual(list(result['Weight (kg)_std']), expected_weights_std)

    def test_nutrition_analysis(self):
        """\nCheck if nutritionAnalysis correctly links nutrition data to exercise performance through regression."""
        # setup our combined data for the test
        combined_data = pd.DataFrame({
            'Exercise': ['Bench Press', 'Bench Press', 'Bench Press'],
            'Protein (g)': [50, 60, 55],
            'Carbs (g)': [200, 220, 210],
            'Fats (g)': [50, 60, 55],
            'Calories': [2000, 2500, 2300],
            'Weight (kg)_std': [1.0, 1.5, 1.2]
        })

        # run the analysis and check the results
        results = nutritionAnalysis(combined_data, 'Bench Press')
        self.assertIsNotNone(results, "Results should not be None")
        self.assertIn('Score', results, "Score should be reported in results")

if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)