import tkinter as tk
from tkinter import filedialog
import subprocess  # Used to run main.py

input_path = ""
weightage_value = 0.0

# Validate float input
def validate_float(input):
    try:
        if input:
            float(input)
        return True
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter a valid float value.")
        return False

def set_weightage():
    # Retrieve the value from the weightage entry field
    weightage_value = weightage_entry.get()

    return weightage_value

# def select_floor_plan():
#     # Open a file dialog to select a floor plan image
#     input_path = tk.filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")])
#     # Display the selected file path in the label
#     selected_path_label.config(text=f"Selected Floor Plan: {input_path}")

#     weightage_value = set_weightage() # str

#     root.destroy()
#     subprocess.run(["python", "main.py", "--floorplan", input_path, "--weightage", weightage_value])

import os
from tkinter import messagebox

def select_floor_plan():
    try:
        # Open a file dialog to select a floor plan image
        input_path = tk.filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")])

        if not input_path:
            # User canceled the file dialog
            return
        
        # Check if input_path is a file and has a .png extension
        if not (os.path.isfile(input_path) and input_path.lower().endswith(".png")):
            raise ValueError("Invalid file format. Please select a .png image file.")

        weightage_value = set_weightage()  # Get the weightage value

        # Check if weightage_value is empty or not a valid float
        if not weightage_value or not validate_float(weightage_value):
            messagebox.showerror("Error", "Please enter a valid weightage value.")
            return

        # Display the selected file path in the label
        selected_path_label.config(text=f"Selected Floor Plan: {input_path}")

        root.destroy()  # Close the GUI window
        subprocess.run(["python", "main.py", "--floorplan", input_path, "--weightage", str(weightage_value)])
    except ValueError as ve:
        # Handle specific value errors (e.g., invalid file format)
        messagebox.showerror("Error", str(ve))
    except Exception as e:
        # Handle any unexpected errors gracefully
        messagebox.showerror("Error", f"An error occurred: {str(e)}")






def run_algorithm():
    # Replace 'path/to/main.py' with the actual path to your main.py script
    main_script = 'main.py'
    
    # You can pass any arguments to main.py using a list like this
    args = ['python', main_script, '--floorplan', input_path, '--weightage', weightage_value]
    
    try:
        # Run main.py as a separate process
        subprocess.run(args)
        print("Algorithm executed successfully.")
    except subprocess.CalledProcessError:
        print("Error running the algorithm.")

    # subprocess.run(["python", "main.py", "--floorplan", file_path])


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

validate_float_cmd = frame.register(validate_float)
weightage_entry.config(validate="key", validatecommand=(validate_float_cmd, "%P"))

# Create a label to display the selected floor plan path
selected_path_label = tk.Label(frame, text="Selected Floor Plan:")
selected_path_label.grid(row=1, column=0, padx=10, pady=10, columnspan=2)

# Create a button to select a floor plan
select_button = tk.Button(frame, text="Select Floor Plan", command=select_floor_plan)
select_button.grid(row=2, column=0, padx=10, pady=10, columnspan=2)

# ----





# ----

# Create a button to run the algorithm
run_button = tk.Button(root, text="Run Algorithm", command=run_algorithm)
# run_button.pack()

# Start the GUI main loop
root.mainloop()
