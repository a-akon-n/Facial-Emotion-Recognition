import tkinter as tk

class MyView:
    def __init__(self, controller):
        self.controller = controller

        self.root = tk.Tk()
        self.root.title("Facial Emotion Recognition")
        self.root.config(bg="skyblue")
        self.root.maxsize(900, 600)
        
        # preparing frames
        self.left_frame = tk.Frame(self.root, width=200, height=350, bg='grey')
        self.left_frame.grid(row=0, column=0, padx=10, pady=5)
        self.right_frame = tk.Frame(self.root, width=650, height=350, bg='grey')
        self.right_frame.grid(row=0, column=1, padx=10, pady=5)
        self.bottom_frame = tk.Frame(self.root, width=850, bg='grey')
        self.bottom_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=5)
        
        # setting text on the bottom
        self.console_label = tk.Label(self.bottom_frame, width=120)
        self.console_label.pack()
        
        #setting image on the right
        
        
        # setting the toolbar
        self.tool_bar = tk.Frame(self.left_frame, width=180, height=185)
        self.tool_bar.grid(row=2, column=0, padx=5, pady=5)
        
        # adding buttons to toolbar
        # self.summary = tk.Button(self.tool_bar, text="Summary", width=15, pady=5)
        self.load_image = tk.Button(self.tool_bar, text="Load Image", width=15, pady=5, command=self.controller.select_file)
        self.predict = tk.Button(self.tool_bar, text="Predict", width=15, pady=5, command=self.controller.predict)
        self.save_model = tk.Button(self.tool_bar, text="Save Model", width=15, pady=5, command=self.controller.save_model)
        self.load_model = tk.Button(self.tool_bar, text="Load Model", width=15, pady=5, command=self.controller.load_model)
        self.read_data = tk.Button(self.tool_bar, text="Read Data", width=15, pady=5, command=self.controller.read_data)
        self.train_model = tk.Button(self.tool_bar, text="Train Model", width=15, pady=5, command=controller.train)
        self.evaluate = tk.Button(self.tool_bar, text="Evaluate", width=15, pady=5, command=self.controller.evaluate)
        self.load_checkpoint = tk.Button(self.tool_bar, text="Load Checkpoint", width=15, pady=5, command=self.controller.load_checkpoint)
        self.exit = tk.Button(self.tool_bar, text="Exit", width=15, pady=5, command=self.controller.stop)
        
        # self.summary.pack()
        self.load_image.pack()
        self.predict.pack()
        self.save_model.pack()
        self.load_model.pack()
        self.read_data.pack()
        self.train_model.pack()
        self.evaluate.pack()
        self.load_checkpoint.pack()
        self.exit.pack()
        
    def run(self):
        self.root.mainloop()
        
    def stop(self):
        self.root.destroy()
