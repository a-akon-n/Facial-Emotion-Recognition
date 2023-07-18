import tkinter as tk
import shutil
import os
import PIL
from PIL import Image, ImageTk
from tkinter import filedialog, PhotoImage
from domain.CNN import CNN

from view.view import MyView


class MyController:
    def __init__(self):
        self.view = MyView(self)
        self.destination_dir = './predict/'
        self.model = CNN((100, 100, 3), 'models/checkpoint/cp.ckpt', 'models/trained_model', epochs=20)
        self.emotions = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']

    def select_file(self):
        self.empty_folder(self.destination_dir) #clears the contents of the destination_folder
        file_name = self.copy_file()
        print(file_name)
        if file_name != '' or None:
            self.load_image('./predict/' + file_name)
        
    def copy_file(self):
        source_file = filedialog.askopenfilename() #select the image to load the application
        if source_file != '':
            file_name = source_file.split('/')[-1].split('.')[0] + '.png' #extract the file name and convert it to .png
            self.view.console_label.config(text=file_name) #view the name of the file in a label
            shutil.copy2(source_file, self.destination_dir + file_name) #copy the selected file into the destination directory
            return file_name

    def load_image(self, image_path):
        image = Image.open(image_path)
        self.view.image = ImageTk.PhotoImage(image)
        self.view.img_label = tk.Label(self.view.right_frame, image=self.view.image, width=650, height=350).grid(
            row=0, column=0, padx=5, pady=5)

    def empty_folder(self, folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    def save_model(self):
        if self.model.save_model():
            self.view.console_label.config(text="Model saved")

    def load_model(self):
        if self.model.load_model():
            self.view.console_label.config(text="Model loaded")
        
    def read_data(self):
        if self.model.read_data():
            self.view.console_label.config(text='Data Loaded')
        
    def train(self):
        self.model.train("yes")
    
    def evaluate(self):
        loss, accuracy = self.model.evaluate()
        self.view.console_label.config(text=f"Evaluation complete, loss: {loss}, accuracy: {accuracy*100}%")

    def predict(self):
        emotion = self.emotions[self.model.predict()]
        self.view.console_label.config(text=emotion)
    
    def load_checkpoint(self):
        if self.model.load_checkpoint():
            self.view.console_label.config(text='Checkpoint Loaded')
        
    # def on_summary_click(self):
        # summary = self.model.model_summary()
        # self.view.console_label.config(text=summary)

    def start(self):
        self.view.run()
    
    def stop(self):
        self.view.stop()
