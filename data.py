
# imports
import pandas as pd
import numpy as np
from analysis import standardize_weights
 
#Data is mostly manually processed when put into excel sheet beforehand

def loadWorkout(file):
    #open file and read it 
    #turn data into dataframe and return that
    workout_data = pd.read_excel(file)
    workout_data["Date"] = pd.to_datetime(workout_data['Date'], format = '%m/%d/%Y')
    return workout_data


def loadDietary(file):
    #open file and read it 
    #turn data into dataframe and return that
    dietary_data = pd.read_excel(file)
    dietary_data["Date"] = pd.to_datetime(dietary_data['Date'], format = '%m/%d/%Y')
    return dietary_data

def processWorkout(workout_data):
    # filter our zero weights
    workout_data = workout_data[workout_data['Weight (kg)'] != 0]
    workout_data = standardize_weights(workout_data)
    # total volume = sets by reps by weight (std)
    workout_data['Total Volume'] = workout_data['Sets'] * workout_data['Reps'] * workout_data['Weight (kg)_std']
    # categorize intensities based off volume
    workout_data['Intensity Category'] = pd.cut(workout_data['Total Volume'], 
                                                 bins=[0, 5000, 10000, 9999999999], 
                                                 labels=['Low', 'Medium', 'High'])

    return workout_data

def processDietary(dietary_data):
    # ratios for later use possibly
    # dietary_data['Protein to Carb Ratio'] = dietary_data['Protein (g)'] / dietary_data['Carbs (g)']
    # dietary_data['Fat to Carb Ratio'] = dietary_data['Fats (g)'] / dietary_data['Carbs (g)']

    # categorize calorie days (more cals equal more energy)
    dietary_data['Calorie Category'] = pd.cut(dietary_data['Calories'], bins=[0, 2000, 3000, 99999999], labels=['Low', 'Medium', 'High'])

    return dietary_data

def ProcessDietaryChange(workout_data, dietary_data, intervention_date):
    #get two sets of data, one for before a dietary change was made and one for after
    intervention_datetime = pd.to_datetime(intervention_date)

    combined_data = pd.merge(workout_data, dietary_data, on='Date', how='inner')

    before_data = combined_data[combined_data['Date'] < intervention_datetime]
    after_data = combined_data[combined_data['Date'] >= intervention_datetime]
    
    return before_data, after_data

