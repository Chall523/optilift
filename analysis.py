#imports

import pandas as pd
from scipy.stats import ttest_rel, ttest_ind
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import data

def correlateDietToWorkout(workout_data, dietary_data):
    combined_data = pd.merge(workout_data, dietary_data, on='Date', how='inner')
    correlation_matrix = combined_data[['Total Volume', 'Calories', 'Protein (g)', 'Carbs (g)', 'Fats (g)']].corr()

    return correlation_matrix


def predictPerformance(workout_data, dietary_data):
    # merge data along date
    combined_data = pd.merge(workout_data, dietary_data, on='Date', how='inner')


    X = combined_data[['Calories']].values.reshape(-1, 1)  # Features
    y = combined_data['Total Volume']                      # Target

    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)

    # extend the calorie range by simulating a month of future data
    max_calories = combined_data['Calories'].max()
    future_calories = pd.DataFrame({'Calories': range(int(max_calories + 1), int(max_calories + 1000), 50)})
    future_predictions = model.predict(future_calories)


    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Actual data')
    plt.plot(X, predictions, color='red', label='Prediction')

    # give future predictions
    plt.plot(future_calories, future_predictions, color='red', linestyle='--', label='Future Predictions')



    plt.title('Linear Regression to Predict Total Volume from Calories')
    plt.xlabel('Calories')
    plt.ylabel('Total Volume')
    plt.legend()
    plt.grid(True)
    plt.show()

    # return the model for later 
    return model


def dietEffectiveness(before_data, after_data):
    # drops the NA values which became an issue
    before_data = before_data['Total Volume'].dropna()
    after_data = after_data['Total Volume'].dropna()

    # make sure neither data set is empty (wrong intervention date)
    if before_data.empty or after_data.empty:
        return {'T-statistic': 'N/A', 'P-value': 'N/A'}

    # perform statistic tests
    t_stat, p_value = ttest_ind(after_data, before_data, equal_var=False)  # unequal variances

    return {'T-statistic': t_stat, 'P-value': p_value}
def AlignDataforNutrition(dietary_data, workout_data):
    #this aligns the workout data with the dietary data as having the nutrition the day previous to the workout
    dietary_data['Date'] = pd.to_datetime(dietary_data['Date'])
    workout_data['Date'] = pd.to_datetime(workout_data['Date'])
    dietary_data['Prev_Date'] = dietary_data['Date'] + pd.Timedelta(days=1)
    combined_data = pd.merge(workout_data, dietary_data, left_on='Date', right_on='Prev_Date', how='inner')
    return combined_data
    #this aligns the workout data with the dietary data as having the nutrition the day previous to the workout


def standardize_weights(workout_data):
    # Using standard ratios to ensure that effort is the same across lifts using the 3:4:5 rule
    strength_ratios = {
        'Bench Press': 3,  
        'Squat': 4,       
        'Deadlift': 4.5     
    }

    workout_data['Weight (kg)_std'] = 0  # create column

    # apply ratios 
    for index, row in workout_data.iterrows():
        exercise = row['Exercise']
        weight = row['Weight (kg)']
        ratio = strength_ratios[exercise]
        standardized_weight = weight / ratio
        workout_data.at[index, 'Weight (kg)_std'] = standardized_weight

    return workout_data

def nutritionAnalysis(combined_data, exercise):
#this relates the nutrition (macros) with the weight performed for an exercise
    exercise_data = combined_data[combined_data['Exercise'] == exercise]
    X = exercise_data[['Protein (g)', 'Carbs (g)', 'Fats (g)', 'Calories']]
    y = exercise_data['Weight (kg)_std']
    model = LinearRegression()
    model.fit(X, y)
    return { #dictionary of different values 
        'Intercept': model.intercept_,
        'Protein Coefficient': model.coef_[0],
        'Carbs Coefficient': model.coef_[1],
        'Fats Coefficient': model.coef_[2],
        'Calories Coefficient': model.coef_[3],
        'Score': model.score(X, y)
    }

# def analyze_data(workout_data, dietary_data, selected_exercise):
