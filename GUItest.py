import tkinter as tk
from tkinter import filedialog
import subprocess  # Used to run main.py

input_path = ""
weightage_value = 0.0

# Create the main GUI window
root = tk.Tk()
root.geometry("800x400")
root.title("CCTV Placement Optimization")

# Create a frame for better organization
frame = tk.Frame(root)
frame.pack(padx=20, pady=20)

# Create a label for setting weightage
weightage_label = tk.Label(frame, text="Minimum CCTV Distance during Initialization\nSet Weightage:")
weightage_label.grid(row=0, column=0, padx=10, pady=10)

# Create an entry field for weightage
weightage_entry = tk.Entry(frame)
weightage_entry.grid(row=0, column=1, padx=10, pady=10)

# Create a label to display the selected floor plan path
selected_path_label = tk.Label(frame, text="Selected Floor Plan:")
selected_path_label.grid(row=1, column=0, padx=10, pady=10, columnspan=2)

# Function to set the weightage value
def set_weightage():
    return float(weightage_entry.get())

# Function to select a floor plan
def select_floor_plan():
    global input_path, weightage_value
    # Open a file dialog to select a floor plan image
    input_path = tk.filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")])
    # Display the selected file path in the label
    selected_path_label.config(text=f"Selected Floor Plan: {input_path}")
    # Get the weightage value
    weightage_value = set_weightage()

# Button to select a floor plan
select_button = tk.Button(frame, text="Select Floor Plan", command=select_floor_plan)
select_button.grid(row=2, column=0, padx=10, pady=10)

# Function to run the algorithm
def run_algorithm():
    global input_path, weightage_value
    # Replace 'path/to/main.py' with the actual path to your main.py script
    main_script = 'main.py'
    
    # You can pass any arguments to main.py using a list like this
    args = ['python', main_script, '--floorplan', input_path, '--weightage', str(weightage_value)]
    
    root.destroy()

    try:
        # Run main.py as a separate process
        subprocess.run(args)
        print("Algorithm executed successfully.")
    except subprocess.CalledProcessError:
        print("Error running the algorithm.")

# Button to run the algorithm
run_button = tk.Button(frame, text="Run Algorithm", command=run_algorithm)
run_button.grid(row=2, column=1, padx=10, pady=10)

# Start the GUI main loop
root.mainloop()
