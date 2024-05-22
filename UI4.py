import tkinter as tk
import win32gui
from PIL import ImageGrab
import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from tkinter.colorchooser import askcolor
import winsound

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.recognition_type = tk.StringVar()
        self.recognition_type.set("Digit Recognition")  # Default choice

        self.canvas = tk.Canvas(self, width=300, height=300, bg="black", cursor="cross")
        self.label = tk.Label(self, text="...", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text="Recognize", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)
        self.color_btn = tk.Button(self, text="Choose Color", command=self.choose_color)
        self.sound_language = tk.StringVar()
        self.sound_language.set("English")

        self.sound_lang_frame = tk.Frame(self)
        self.sound_lang_frame.grid(row=4, column=0, pady=2)
        self.eng_radio = tk.Radiobutton(self.sound_lang_frame, text="English", variable=self.sound_language, value="English")
        self.eng_radio.grid(row=0, column=0)
        self.fr_radio = tk.Radiobutton(self.sound_lang_frame, text="Indonesia", variable=self.sound_language, value="Indonesia")
        self.fr_radio.grid(row=1, column=0)

        self.recognition_menu = tk.OptionMenu(self, self.recognition_type, "Digit Recognition", "Lowercase Recognition", "Uppercase Recognition")
        self.recognition_menu.config(width=20)
        self.recognition_menu.grid(row=2, column=1, pady=2)

        self.brush_size_label = tk.Label(self, text="Brush Size")
        self.brush_size_slider = tk.Scale(self, from_=1, to=20, orient=tk.HORIZONTAL, length=150)
        self.brush_size_slider.set(10)  # Default brush size
        self.brush_size_label.grid(row=3, column=0, pady=2)
        self.brush_size_slider.grid(row=3, column=1, pady=2)

        self.canvas.grid(row=0, column=0, pady=2, sticky="W")
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        self.color_btn.grid(row=2, column=0, pady=2)

        # Bind the paint method to the B1-Motion event
        self.canvas.bind("<B1-Motion>", self.paint)

        # Default brush color and size
        self.brush_color = "white"
        self.brush_size = 10

    def paint(self, event):
            # Retrieve brush size from the slider
        brush_size = self.brush_size_slider.get()
        x1, y1 = (event.x - brush_size), (event.y - brush_size)
        x2, y2 = (event.x + brush_size), (event.y + brush_size)
        self.canvas.create_oval(x1, y1, x2, y2, fill=self.brush_color, width=0)

    def clear_all(self):
        self.canvas.delete("all")
        self.label.configure(text="Thinking..")


    def classify_handwriting(self):
        if self.sound_language.get() == "English":
            sound_map ={
                # uppercase letters
                "A": "Sound\A.wav",
                "B": "Sound\B.wav",
                "C": "Sound\C.wav",
                "D": "Sound\D.wav",
                "E": "Sound\E.wav",
                "F": "Sound\F.wav",
                "G": "Sound\G.wav",
                "H": "Sound\H.wav",
                "I": "Sound\I.wav",
                "J": "Sound\J.wav",
                "K": "Sound\K.wav",
                "L": "Sound\L.wav",
                "M": "Sound\M.wav",
                "N": r"Sound\N.wav",
                "O": "Sound\O.wav",
                "P": "Sound\P.wav",
                "Q": "Sound\Q.wav",
                "R": "Sound\R.wav",
                "S": "Sound\S.wav",
                "T": "Sound\T.wav",
                "U": r"Sound\U.wav",
                "V": "Sound\V.wav",
                "W": "Sound\W.wav",
                "X": "Sound\X.wav",
                "Y": "Sound\Y.wav",
                "Z": "Sound\Z.wav",
                # lowercase letters
                "a": "Sound\A.wav",
                "b": "Sound\B.wav",
                "c": "Sound\C.wav",
                "d": "Sound\D.wav",
                "e": "Sound\E.wav",
                "f": "Sound\F.wav",
                "g": "Sound\G.wav",
                "h": "Sound\H.wav",
                "i": "Sound\I.wav",
                "j": "Sound\J.wav",
                "k": "Sound\K.wav",
                "l": "Sound\L.wav",
                "m": "Sound\M.wav",
                "n": r"Sound\N.wav",
                "o": "Sound\O.wav",
                "p": "Sound\P.wav",
                "q": "Sound\Q.wav",
                "r": "Sound\R.wav",
                "s": "Sound\S.wav",
                "t": "Sound\T.wav",
                "u": r"Sound\U.wav",
                "v": "Sound\V.wav",
                "w": "Sound\W.wav",
                "x": "Sound\X.wav",
                "y": "Sound\Y.wav",
                "z": "Sound\Z.wav",
                #digit
                "0": r"Sound\0.wav",
                "1": r"Sound\1.wav",
                "2": r"Sound\2.wav",
                "3": r"Sound\3.wav",
                "4": r"Sound\4.wav",
                "5": r"Sound\5.wav",
                "6": r"Sound\6.wav",
                "7": r"Sound\7.wav",
                "8": r"Sound\8.wav",
                "9": r"Sound\9.wav"
            }
        
        elif self.sound_language.get() == "Indonesia":
            sound_map ={
                # uppercase letters
                "A": "Sound\Indonesia\A.wav",
                "B": "Sound\Indonesia\B.wav",
                "C": "Sound\Indonesia\C.wav",
                "D": "Sound\Indonesia\D.wav",
                "E": "Sound\Indonesia\E.wav",
                "F": "Sound\Indonesia\F.wav",
                "G": "Sound\Indonesia\G.wav",
                "H": "Sound\Indonesia\H.wav",
                "I": "Sound\Indonesia\I.wav",
                "J": "Sound\Indonesia\J.wav",
                "K": "Sound\Indonesia\K.wav",
                "L": "Sound\Indonesia\L.wav",
                "M": "Sound\Indonesia\M.wav",
                "N": r"Sound\Indonesia\N.wav",
                "O": "Sound\Indonesia\O.wav",
                "P": "Sound\Indonesia\P.wav",
                "Q": "Sound\Indonesia\Q.wav",
                "R": "Sound\Indonesia\R.wav",
                "S": "Sound\Indonesia\S.wav",
                "T": "Sound\Indonesia\T.wav",
                "U": r"Sound\Indonesia\U.wav",
                "V": "Sound\Indonesia\V.wav",
                "W": "Sound\Indonesia\W.wav",
                "X": "Sound\Indonesia\X.wav",
                "Y": "Sound\Indonesia\Y.wav",
                "Z": "Sound\Indonesia\Z.wav",
                # lowercase letters
                "a": "Sound\Indonesia\A.wav",
                "b": "Sound\Indonesia\B.wav",
                "c": "Sound\Indonesia\C.wav",
                "d": "Sound\Indonesia\D.wav",
                "e": "Sound\Indonesia\E.wav",
                "f": "Sound\Indonesia\F.wav",
                "g": "Sound\Indonesia\G.wav",
                "h": "Sound\Indonesia\H.wav",
                "i": "Sound\Indonesia\I.wav",
                "j": "Sound\Indonesia\J.wav",
                "k": "Sound\Indonesia\K.wav",
                "l": "Sound\Indonesia\L.wav",
                "m": "Sound\Indonesia\M.wav",
                "n": r"Sound\Indonesia\N.wav",
                "o": "Sound\Indonesia\O.wav",
                "p": "Sound\Indonesia\P.wav",
                "q": "Sound\Indonesia\Q.wav",
                "r": "Sound\Indonesia\R.wav",
                "s": "Sound\Indonesia\S.wav",
                "t": "Sound\Indonesia\T.wav",
                "u": r"Sound\Indonesia\U.wav",
                "v": "Sound\Indonesia\V.wav",
                "w": "Sound\Indonesia\W.wav",
                "x": "Sound\Indonesia\X.wav",
                "y": "Sound\Indonesia\Y.wav",
                "z": "Sound\Indonesia\Z.wav",
                #digit
                "0": r"Sound\Indonesia\0.wav",
                "1": r"Sound\Indonesia\1.wav",
                "2": r"Sound\Indonesia\2.wav",
                "3": r"Sound\Indonesia\3.wav",
                "4": r"Sound\Indonesia\4.wav",
                "5": r"Sound\Indonesia\5.wav",
                "6": r"Sound\Indonesia\6.wav",
                "7": r"Sound\Indonesia\7.wav",
                "8": r"Sound\Indonesia\8.wav",
                "9": r"Sound\Indonesia\9.wav"
            }

        HWND = self.canvas.winfo_id()
        rect = win32gui.GetWindowRect(HWND)
        im = ImageGrab.grab(rect)

        digit, acc = predict_digit(im, self.recognition_type.get())

        predicted_label, acc = predict_digit(im, self.recognition_type.get())
        self.label.configure(text=str(digit) + ', ' + str(int(acc * 100)) + '%')

        if predicted_label in sound_map:
            sound_file = sound_map[predicted_label]
            winsound.PlaySound(sound_file, winsound.SND_FILENAME)

    def choose_color(self):
        color = askcolor()[1]  # Opens a color picker dialog and returns the chosen color
        if color:
            self.brush_color = color

def predict_digit(image, recognition_type):

    if recognition_type == "Digit Recognition":
        model_path = "digitsmodel.h5"
        labels_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    elif recognition_type == "Lowercase Recognition":
        model_path = "lowercasemodel.h5"
        labels_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    elif recognition_type == "Uppercase Recognition":
        model_path = "uppercasemodel.h5"
        labels_list = labels_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    else:
        raise ValueError("Invalid recognition type")

    model = load_model(model_path)

    # Convert RGB to grayscale
    img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Resize the image to (32, 32)
    img_resized = cv2.resize(img_gray, (32, 32))

    # Stack grayscale image to create 3 channels
    img_rgb = np.stack([img_resized] * 3, axis=-1)

    # Normalize the pixel values to the range [0, 1]
    img_normalized = img_rgb / 255.0

    img_test = img_normalized[np.newaxis, ...]

    # Make prediction
    prediction = model.predict(img_test)

    # Display the input image and predicted label using Matplotlib
    predicted_label_index = np.argmax(prediction)

    if 0 <= predicted_label_index < len(labels_list):
        predicted_label = labels_list[predicted_label_index]
    else:
        predicted_label = 'unknown'

    # plt.figure(figsize=(4, 4))
    # plt.imshow(img_resized, cmap='gray')  # Display the grayscale image
    # plt.title(f"Predicted: {predicted_label}")
    # plt.axis('off')
    # plt.show()

    return predicted_label, np.max(prediction)

if __name__ == "__main__":
    app = App()
    app.title("Gimanani v1.0")
    app.mainloop()
