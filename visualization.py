import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

def MacroDist(combined_data):
    # Calculate the mean of macronutrients across all available data
    mean_values = combined_data[['Protein (g)', 'Carbs (g)', 'Fats (g)']].mean()

    # Prepare labels and values for the pie chart
    nutrients = ['Protein (g)', 'Carbs (g)', 'Fats (g)']
    values = [mean_values[nutrient] for nutrient in nutrients]

    # Plotting the pie chart
    plt.figure(figsize=(8, 8))  # Set the figure size for better visibility
    plt.pie(values, labels=nutrients, autopct='%1.1f%%', startangle=140)
    plt.title('Average Macronutrient Distribution')
    plt.axis('equal')  # This makes the pie chart circular
    plt.show()

def gains(combined_data):
    plt.figure(figsize=(12, 15))
    exercises = ['Bench Press', 'Squat', 'Deadlift'] #compound lifts

    # loop through each exercise and plot its data
    for exercise in exercises:
        exercise_data = combined_data[combined_data['Exercise'] == exercise]
        plt.plot(exercise_data['Date'], exercise_data['Weight (kg)'], label=exercise)

   
    plt.title('Strength Gains Over Time')
    plt.xlabel('Date')
    plt.ylabel('Max Weight Lifted (kg)')
    plt.legend()
    plt.show()

def exerciseProgress(combined_data, exercise, goal_value):
    # find data for the exercise chosen
    exercise_data = combined_data[combined_data['Exercise'] == exercise]
    # check if there is any data, and get the last recorded weight
    if not exercise_data.empty:
        current_value = exercise_data.iloc[-1]['Weight (kg)']  # get the most recent weight
    else:
        current_value = 0  # set to 0 if no data exists

    # create a figure and axis for the bar plot
    fig, ax = plt.subplots()
    # make a horizontal bar for current pr
    ax.barh([exercise], [current_value], color='lightblue')
    # draw a vertical line for the goal value 
    ax.axvline(x=goal_value, color='green', label=f'Goal for {exercise}')
    # acommodate the xaxis for largest value
    ax.set_xlim(0, max(goal_value, current_value) + 10)
    ax.set_xlabel('Weight Lifted (kg)')
    ax.set_title(f'Progress Towards Goal for {exercise}')
    plt.legend()
    plt.show()

def heatmap(correlation_matrix):
    plt.figure(figsize=(12,10))
    sns.heatmap(correlation_matrix, annot=True, cmap='plasma')
    plt.title("Correlation between Dietary Metrics and Workout Total Volume")
    plt.show()

def forecastLift(combined_data, exercise, future_sessions=5):
    # filter data for the specific exercise
    exercise_data = combined_data[combined_data['Exercise'] == exercise]
    exercise_data = exercise_data.sort_values('Date')  # Ensure data is sorted by date

    
    X = np.arange(len(exercise_data)).reshape(-1, 1)
    y = exercise_data['Weight (kg)'].values

    model = LinearRegression()
    model.fit(X, y)

    # create dates for predictions
    last_date = pd.to_datetime(exercise_data['Date'].iloc[-1])
    future_dates = [last_date + pd.DateOffset(days=30*i) for i in range(1, future_sessions+1)]

    # predict future values
    future_X = np.arange(len(X), len(X) + future_sessions).reshape(-1, 1)
    future_predictions = model.predict(future_X)

    # plot historical data
    plt.figure(figsize=(10, 5))
    plt.plot(exercise_data['Date'], y, label='Historical Data', marker='o')

    # plot predictions
    plt.plot(future_dates, future_predictions, 'r--', label='Predicted Future Weights')
    plt.xlabel('Date')
    plt.ylabel('Weight Lifted (kg)')
    plt.title(f'Historical Data and Predictions for {exercise}')
    plt.legend()
    plt.xticks(rotation=45)

    plt.show()
