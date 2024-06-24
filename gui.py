import tkinter as tk
from tkinter import filedialog, Label, Button, messagebox
from unittest import result
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2

def load_model(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json (loaded_model_json)
    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss = 'sparse_categorial_crossentropy', metrics= ['accuracy'])
    
    return model

top = tk.Tk()
top.geometry('1000x600')
top.title('Animal Emotion and Name Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD',font =('arial',15,'bold'))
label2 = Label(top, background='#CDCDCD',font =('arial',15,'bold'))
label3 = Label(top, background='#CDCDCD',font =('arial',15,'bold'))
sign_image = Label(top)

facec = cv2.CascadeClassifier('haarcascade_animal.xml')
emotion_model = load_model("model_emotion.json","animal_emotion.weights.h5")
animal_model = load_model("model_name.json","animal_name.weights.h5")

EMOTION_LIST = ["happy", "sad", "angry"]
ANIMAL_LIST = ["cat", "wild", "dog"]

def detect(file_path):
    global label1, label2, label3, result_frame
    image = cv2.imread(file_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = facec.detectMultiScale(rgb_image, 1.3,5)
    
    results = []

    try:
        for (x,y,w,h) in faces:
            fc = rgb_image[y:y+h, x:x+w]
            roi = cv2.resize(fc, (224,224))
            roi = roi/255.0
            emotion_pred = EMOTION_LIST[np.argmax(emotion_model.predict(roi[np.newaxis,:,:,:]))] 
            animal_pred = ANIMAL_LIST[np.argmax(emotion_model.predict(roi[np.newaxis,:,:,:]))]
            results.append((animal_pred, emotion_pred))

        result_text = "\n".join([f"Animal:{res[0]}, Emotion:{res[1]}"for res in results])
        messagebox.showinfo("Detection Results", result_text)

        for widget in result_frame.winfo_children():
            widget.destroy()

        result_label = Label(result_frame, text = result_text, font = ('arial', 12, 'bold'), justify = 'left')
        result_label.pack()

    except:
        label1.configure(foreground="#011638",text = "Unable to detect")

def show_Detect_button(file_path):
    detect_b = Button(top,text="Detect Emotion and Animal", command = lambda: detect(file_path),padx=10,pady=5)
    detect_b.configure(background="#364156",foreground='white',font=('arial',10,'bold'))
    detect_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image = im)
        sign_image.image = im
        label2.configure(text = '')
        label3.configure(text = '')
        show_Detect_button(file_path) 
    except:
        pass

upload = Button(top,text = "Upload Image", command = upload_image, padx=10, pady=5)  
upload.configure(background="#364156",foreground='white',font=('arial',20,'bold'))
upload.pack(side='bottom',pady=50)
sign_image.pack(side='bottom',expand='True')
label2.pack(side='bottom',expand='True')
label3.pack(side='bottom',expand='True')

result_frame = tk.Frame(top, background = '#CDCDCD')
result_frame.pack(side='bottom', fill= 'both', expand = True)

heading = Label(top,text='Animal Emotion and Name Detector',pady=20,font=('arial',25,'bold'))
heading.configure(background= '#CDCDCD',foreground="#364156")
heading.pack()
top.mainloop()