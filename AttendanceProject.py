import cv2
import os
import numpy as np
import pandas as pd
import face_recognition
from tkinter import filedialog
from PIL import Image

from kivy.app import App
from kivy.uix.camera import Camera
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.uix.button import Button
from kivy.uix.spinner import Spinner
from kivy.clock import Clock


def enhance(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def findEncodings(images):
    encodelist = []
    for img in images:
        img = enhance(img)
        encoding = face_recognition.face_encodings(img)[0]
        encodelist.append(encoding)
    return encodelist


def add_database(directory):

    # create Codes.csv file if it doesn't exist
    if not os.path.exists('Codes.csv'):
        with open('Codes.csv', 'w') as f1:
            f1.write('ID\n')
            
    if not os.path.exists('Encodings.csv'):
        with open('Encodings.csv', 'w') as f2:
            f2.write('encoding')
            for i in range(127):
                f2.write(',encoding')
            f2.write('\n')
    # load existing codes
    
    codes_df = pd.read_csv('Codes.csv')
    existing_codes = set(codes_df['ID'].values.tolist())
    skip_counter = 0
    add_counter = 0
    images = []
    classNames = []
    for foldername in os.listdir(directory):
        if str(foldername).strip() in [str(code).strip() for code in existing_codes]:
            skip_counter += 1 
            continue
        add_counter += 1
        folderpath = os.path.join(directory, foldername)
        if os.path.isdir(folderpath):
            for filename in os.listdir(folderpath):
                filepath = os.path.join(folderpath, filename)
                img = cv2.imread(filepath)
                images.append(img)
                classNames.append(foldername)


    print(f'Adding {add_counter} students.')
    print(f'Skipping {skip_counter} students.')
    if len(images) == 0:
        print('No new images found in directory.')
        return
    encodelist = findEncodings(images)
    df1 = pd.DataFrame(encodelist)
    df1.to_csv('Encodings.csv',index=False, mode='a', header = False)
    df2 = pd.DataFrame(classNames)
    df2.to_csv('Codes.csv', index=False, mode='a', header = False)
    print(f'{len(classNames)} new images added to the database.')

          
def mark_attendance(name, session_num):
    filename = f"attendanceSheet.csv"
    df = pd.read_csv(filename)
    for i, col in enumerate(df['ID'].tolist()):
        if name == col:
            df.loc[i, session_num] = 1      
    df.to_csv(filename, index=False)


test = 0


def who_is_it(file_paths, session_num):
    count = 0
    for file_path in file_paths:
        encoode = pd.read_csv('Encodings.csv')
        Class = pd.read_csv('Codes.csv')
        Class = Class['ID'].tolist()
        global test
        test = face_recognition.load_image_file(file_path)
        test = enhance(test)
        loc = face_recognition.face_locations(test)
        encoodingstest = face_recognition.face_encodings(test)
        counter = 0
        for encodeFace, faceLoc in zip(encoodingstest, loc):
            matches = face_recognition.compare_faces(encoode, encodeFace)
            faceDis = face_recognition.face_distance(encoode, encodeFace)
            matchInd = np.argmin(faceDis)
            if matches[matchInd]:
                name = Class[matchInd]
                print(name)
                y1, x2, y2, x1 = faceLoc
                cv2.rectangle(test, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(test, str(name), (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                mark_attendance(name, session_num)
            else:
                counter += 1
                y1, x2, y2, x1 = faceLoc
                cv2.rectangle(test, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(test, 'NA', (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        with open('Output.txt', 'a') as f:
            # Append the text to the file
            f.write("Found " + str(counter) + " people who are not in the database..\n")


         #display image from the texture

        cv2.imshow('group' + str(count), test)
        count += 1


#############################################

def select_file():
    # Show a file dialog to select one or more image files
    filepaths = filedialog.askopenfilenames(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])
    num_files = len(filepaths)
    print(f"Selected {num_files} files: {filepaths}")
    return filepaths

def select_folder():
    folderpath = filedialog.askdirectory()
    print(f"Selected {folderpath}")
    return folderpath


class AttendanceSystem(App):
    def build(self):
        self.window = GridLayout()
        self.window.cols = 1
        self.window.pos_hint = {"center_x":0.5, "center_y":0.5}

        self.button1 = Button(text='Take Attendance',
                              on_press=self.on_button1_press,
                              size_hint = (1, 0.5),
                              font_size = 22,
                              )
        self.button2 = Button(text='Add to Data base',
                              on_press=self.on_button2_press,
                              size_hint = (1, 0.5),
                              font_size = 22,
                              )
        self.button3 = Button(text='Start Camera',
                              on_press=self.on_button3_press,
                              size_hint = (1, 0.5),
                              font_size = 22,
                              )

        self.spinner = Spinner(
            text='Session 1',  # Default text displayed in the dropdown list
            values = ["Session 1", "Session 2", "Session 3", "Session 4", "Session 5", "Session 6"],  # options for session selection
            size_hint=(1, 0.5),  # Set the size of the dropdown list
            size=(100, 44),  # Set the size of the dropdown list
            font_size = 22,
            pos_hint={'center_x': 0.5, 'center_y': 0.5}  # Set the position of the dropdown list
        )

        # Create the button widget

        self.window.add_widget(self.spinner)
        self.window.add_widget(self.button1)
        self.window.add_widget(self.button2)
        self.window.add_widget(self.button3)

        return self.window
    
    def on_button1_press(self, instance):
        who_is_it(select_file(), self.spinner.text)
        buf1 = cv2.flip(test, 0)
        buf = buf1.tobytes()
        texture1 = Texture.create(size=(test.shape[1], test.shape[0]), colorfmt='bgr')
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.window.cols = 4
        self.image = Image(texture = texture1,
                           size_hint = (1.5,0.5))
        self.window.add_widget(self.image)

    def on_button2_press(self, instance):
        add_database(select_folder())

    def on_button3_press(self, instance):
        self.camera = Camera(resolution=(1280, 720), size_hint=(1, 0.8), play=True, index = 0)
        self.window.add_widget(self.camera)
        snap_button = Button(text="Take Picture", size_hint=(1, 0.2),
                             on_press=self.take_picture)

        self.window.add_widget(snap_button)

    def take_picture(self, instance):
        timestamp = int(Clock.get_time())
        filename = f"capture_{timestamp}.png"
        # Taking the snap
        foldername = "captures"
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        filepath = os.path.join(foldername, filename)
        self.camera.export_to_png(filepath)


AttendanceSystem().run()


