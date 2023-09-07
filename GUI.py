import tkinter as tk
from tkinter import filedialog
import subprocess  # Used to run main.py

input_path = ""
weightage_value = 0.0



def set_weightage():
    # Retrieve the value from the weightage entry field
    weightage_value = weightage_entry.get()

    return weightage_value

def select_floor_plan():
    # Open a file dialog to select a floor plan image
    input_path = tk.filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")])
    # Display the selected file path in the label
    selected_path_label.config(text=f"Selected Floor Plan: {input_path}")

    weightage_value = set_weightage() # str

    root.destroy()
    subprocess.run(["python", "main.py", "--floorplan", input_path, "--weightage", weightage_value])



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
root.geometry("640x480")
root.title("CCTV Placement Optimization")

# Create an entry field for setting weightage
weightage_label = tk.Label(root, text="\n\n\nMinimum CCTV Distance during INITIALIZATION\n\nSet Weightage: ")
weightage_label.pack()
weightage_entry = tk.Entry(root, text="\n\n\n")
weightage_entry.pack()

# Create a label to display the selected floor plan path
selected_path_label = tk.Label(root, text="\n\n\nSelected Floor Plan: \n")
selected_path_label.pack()

# Create a button to select a floor plan
select_button = tk.Button(root, text="Select Floor Plan", command=select_floor_plan)
select_button.pack()

# Create a button to run the algorithm
run_button = tk.Button(root, text="Run Algorithm", command=run_algorithm)
# run_button.pack()

# Start the GUI main loop
root.mainloop()
