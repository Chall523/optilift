import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import pandas as pd
import data
import analysis
import visualization
from ttkthemes import ThemedTk

data_storage = {} #global variable

def launch_main_window():
    #used a theme to make the GUI more aesthetic
    root = ThemedTk(theme="clearlooks")
    root.title('OptiLift Fitness Tracker')
    root.geometry('1200x900')  

    # configure style of the widgets
    style = ttk.Style()
    style.configure('TButton', font=('Georgia', 14))
    style.configure('TLabel', font=('Georgia', 14))
    style.configure('TEntry', font=('Georgia', 14))
    
    # create and pack the title label
    title_label = ttk.Label(root, text="OptiLift Fitness Tracker", font=("Georgia", 24))
    title_label.pack(pady=20)

    # upload work datta
    upload_workout_button = ttk.Button(root, text="Upload Workout Data", command=lambda: upload_data('workout'))
    upload_workout_button.pack(pady=10)

    # button to upload diet data
    upload_diet_button = ttk.Button(root, text="Upload Dietary Data",
                                    command = lambda: upload_data('diet'))
    upload_diet_button.pack(pady=10)

    # set up analysis options and dropdown menu
    analysis_options = tk.StringVar(root)
    analysis_options.set("Select Analysis")
    analysis_dropdown = ttk.Combobox(root, textvariable=analysis_options, values=["Correlate Diet to Workout", "Predict Performance", "Diet Effectiveness", "Nutrition Analysis"])
    analysis_dropdown.pack(pady=12)

    # create and pack the button to run analysis
    run_analysis_button = ttk.Button(root, text="Run Analysis",
                                     command =lambda: run_analysis(analysis_options.get()))
    run_analysis_button.pack(pady=10)

    # set up visualization options and dropdown menu
    visualization_options = tk.StringVar(root)
    visualization_options.set("Select Visualization")
    visualization_dropdown = ttk.Combobox(root, textvariable=visualization_options,
                                          values=["Macro Distribution", "Performance Gains", "Exercise Progress", "Forecast Specific Lift"])
    visualization_dropdown.pack(pady=10)

    # show visual button
    run_visualization_button = ttk.Button(root, text="Show Visualization",
                                          command=lambda: run_visualization(visualization_options.get()))
    run_visualization_button.pack(pady=10)

    return root

def upload_data(data_type):
    # ask for an excel file
    file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
    # check if a file was selected
    if file_path:
        # if the data type is 'workout', load and process workout data
        if data_type == 'workout':
            dataframe = data.loadWorkout(file_path)
            processed_data = data.processWorkout(dataframe)
            data_storage[data_type] = processed_data
        # if the data type is 'diet', load dietary data
        elif data_type == 'diet':
            dataframe = data.loadDietary(file_path)
            data_storage[data_type] = dataframe
        
        # check if both workout and diet data are available and combine them
        if 'workout' in data_storage and 'diet' in data_storage:
            data_storage['combined'] = pd.merge(data_storage['workout'], data_storage['diet'], on='Date', how='inner')
        
        # display a success message
        messagebox.showinfo("Success", f"{data_type.capitalize()} data uploaded successfully.")

def run_analysis(option):
    # based on user selection will do a certain analysis function as prev written
    if option == "Correlate Diet to Workout":
        correlation_matrix = analysis.correlateDietToWorkout(data_storage['workout'], data_storage['diet'])
        result = visualization.heatmap(correlation_matrix)
        messagebox.showinfo("Analysis Result", str(result))

    elif option == "Predict Performance":
        result = analysis.predictPerformance(data_storage['workout'], data_storage['diet'])
        messagebox.showinfo("Analysis Result", str(result))

    elif option == "Diet Effectiveness":
        # prompt user to enter the date of dietary intervention
        intervention_date = simpledialog.askstring("Input", "Enter the intervention date (YYYY-MM-DD):", parent=main_window)
        # process data before and after the intervention
        before_data, after_data = data.ProcessDietaryChange(data_storage['workout'], data_storage['diet'], intervention_date)
        result = analysis.dietEffectiveness(before_data, after_data)
        messagebox.showinfo("Diet Effectiveness Result", f"T-statistic: {result['T-statistic']}, P-value: {result['P-value']}")

    elif option == "Nutrition Analysis":
        # prompt user to select an exercise for analysis
        exercise = simpledialog.askstring("Input", "Enter the exercise name (Bench Press, Squat, or Deadlift):", parent=main_window)
        combined_data = analysis.AlignDataforNutrition(data_storage['diet'], data_storage['workout'])
        # standardize weights for consistent analysis
        combined_data = analysis.standardize_weights(combined_data)
        nutrition_results = analysis.nutritionAnalysis(combined_data, exercise)
        # show the analysis results in a message box
        results_message = (
            f"Intercept: {nutrition_results['Intercept']}\n"
            f"Protein Coefficient: {nutrition_results['Protein Coefficient']}\n"
            f"Carbs Coefficient: {nutrition_results['Carbs Coefficient']}\n"
            f"Fats Coefficient: {nutrition_results['Fats Coefficient']}\n"
            f"Calories Coefficient: {nutrition_results['Calories Coefficient']}\n"
            f"Score: {nutrition_results['Score']}")
        messagebox.showinfo("Nutrition Analysis Results", results_message)

def run_visualization(option):
    # visual based on user selection
    if option == "Macro Distribution":
        visualization.MacroDist(data_storage['combined'])

    elif option == "Performance Gains":
        visualization.gains(data_storage['workout'])

    elif option == "Exercise Progress":
        # prompt user to enter exercise name and goal weight
        exercise = simpledialog.askstring("Input", "Enter the exercise name:", parent=main_window)
        goal_value = simpledialog.askinteger("Input", "Enter your goal weight (kg):", parent=main_window)
        visualization.exerciseProgress(data_storage['combined'], exercise, goal_value)

    elif option == "Forecast Specific Lift":
        # prompt user to enter exercise name and number of future sessions (in months)
        exercise = simpledialog.askstring("Input", "Enter the exercise name:", parent=main_window)
        future_sessions = simpledialog.askinteger("Input", "Enter the number of future sessions:", parent=main_window)
        visualization.forecastLift(data_storage['combined'], exercise, future_sessions)

if __name__ == "__main__":
    main_window = launch_main_window()
    main_window.mainloop()