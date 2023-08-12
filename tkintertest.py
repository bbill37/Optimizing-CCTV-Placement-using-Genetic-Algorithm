# importing everything from tkinter
from tkinter import *

# create gui window
root = Tk()

# set the configuration
# of the window
root.geometry("1280x720")

# define a function
# for setting the new text
def java():
	my_string_var.set("You must go with Java")

# define a function
# for setting the new text
def python():
	my_string_var.set("You must go with Python")
	
# define a function
# for setting the new text
def increaseQty():
	my_string_var.set("qty increased")

# define a function
# for setting the new text
def decreaseQty():
	my_string_var.set("qty decreased")

# create a Button widget and attached
# with java function
btn_1 = Button(root,
			text = "I love Android",
			command = java)

# create a Button widget and attached
# with python function
btn_2 = Button(root,
			text = "I love Machine Learning",
			command = python)

# create a Button widget and attached
# with python function
btn_3 = Button(root,
			text = "+",
			command = increaseQty)

# create a Button widget and attached
# with python function
btn_4 = Button(root,
			text = "-",
			command = decreaseQty)

# create a StringVar class
my_string_var = StringVar()

# set the text
my_string_var.set("What should I learn")

# create a label widget
my_label = Label(root,
				textvariable = my_string_var)


# place widgets into
# the gui window
btn_1.pack()
btn_2.pack()
btn_3.pack()
btn_4.pack()
my_label.pack()

# Start the GUI
# root.mainloop()

# -------------------------------------------

from tkinter import filedialog

def openFile():
    tf = filedialog.askopenfilename(
        initialdir="C:/Users/MainFrame/Desktop/", 
        title="Open Text file", 
        filetypes=(("Text Files", "*.txt"),)
        )
    pathh.insert(END, tf)
    tf = open(tf)  # or tf = open(tf, 'r')
    data = tf.read()
    txtarea.insert(END, data)
    tf.close()

txtarea = Text(root, width=40, height=20)
txtarea.pack(pady=20)

pathh = Entry(root)
pathh.pack(side=LEFT, expand=True, fill=X, padx=20)

Button(
    root, 
    text="Open File", 
    command=openFile
    ).pack(side=RIGHT, expand=True, fill=X, padx=20)


# -------------------------------------------

# from tkinter import *
# root = Tk()

root.title('Performance Dashboard')

menu = Menu(root)
root.config(menu=menu)
filemenu = Menu(menu)
menu.add_cascade(label='File', menu=filemenu)
filemenu.add_command(label='New')
filemenu.add_command(label='Open...')
filemenu.add_separator()
filemenu.add_command(label='Exit', command=root.quit)
helpmenu = Menu(menu)
menu.add_cascade(label='Help', menu=helpmenu)
helpmenu.add_command(label='About')

# T = Text(root, height=30, width=50)
# T.pack()
# T.insert(END, 'GeeksforGeeks\nBEST WEBSITE\n')

mainloop()


# T.insert(END, 'awawawa')

# root.update()
