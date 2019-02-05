import cv2
import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

#Local Dependencies
from process import Process
from webcam import Webcam
from video import Video
from interface import plotXY, waitKey
from groundtruth import GroundTruth

MAIN_WINDOW_WIDTH = 640
MAIN_WINDOW_HEIGHT = 360

SECONDARY_WINDOW_WIDTH = 256
SECONDARY_WINDOW_HEIGHT = 256

def get_change(current, previous):
    if current == previous:
        return 100.0
    try:
        return (abs(current - previous) / previous) * 100.0
    except ZeroDivisionError:
        return 0

class Communicate(QObject):
    closeApp = pyqtSignal()


class Thread(QThread):
    changePixmap = pyqtSignal(QImage)

    def run(self):
        self.process = Process(False)
        self.input = Webcam()
        self.frame = np.zeros((10,10,3),np.uint8)
        self.bpm = 0
        self.bpms = []
        self.input.start()
        while True:
            frame = self.input.get_frame()

            if frame is not None:
                self.process.frame_in = frame
                self.process.run_online_2()
                self.frame = self.process.frame_out #get the frame to show in GUI
                self.f_fr = self.process.frame_ROI #get the face to show in GUI
                self.bpm = self.process.bpm #get the bpm change over the time
                
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
                
                cv2.putText(self.frame, "FPS "+str(float("{:.2f}".format(self.process.fps))),
                           (20,460), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255),2)


                if self.process.bpms.__len__() > 3:
                    if(max(self.process.bpms[-3:]-np.mean(self.process.bpms[-3]))<5): #show HR if it is stable -the change is not over 5 bpm- for 3s
                        print("Heart rate: " + str(float("{:.2f}".format(np.mean(self.process.bpms)))) + " bpm")
                    cv2.putText(self.frame, "Heart rate: " + str(float("{:.2f}".format(np.mean(self.process.bpms)))) + " bpm", (20,160), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255),2)
                img = QImage(self.frame, self.frame.shape[1], self.frame.shape[0], 
                                self.frame.strides[0], QImage.Format_RGB888)
                self.changePixmap.emit(img)

#Class to run Method1    
class GUI(QMainWindow):
    def __init__(self, use_camera, dataset_location='../dataset/data'):
        super(GUI,self).__init__()

        self.use_camera = use_camera
        self.initUI()

        if self.use_camera:
            self.statusBar.showMessage("Input: webcam",5000)
            self.btnStart.setEnabled(True)
            self.status = False
            self.process = Process(False)
            self.reset()
            self.input = Webcam()
            self.process.reset()
            self.status = False
            self.frame = np.zeros((10,10,3),np.uint8)
            self.bpm = 0
            self.bpms = []
        else:   
            self.statusBar.showMessage("Input: dataset",5000)
            self.process = Process(True, 25)

            self.video_filefolder = os.path.join(dataset_location, 'Video')
            self.groundtruth_filepath = os.path.join(dataset_location, 'ground_truths.csv')

            self.groundtruth = GroundTruth(self.groundtruth_filepath)
            self.input_array = self.create_input_array()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.lblDisplay.setPixmap(QPixmap.fromImage(image))

    def initUI(self):
        #set font
        font = QFont()
        font.setPointSize(16)
        
        #widgets
        self.btnStart = QPushButton("Start Camera", self)
        self.btnStart.move(680,340)
        self.btnStart.setFixedWidth(200)
        self.btnStart.setFixedHeight(50)
        self.btnStart.setFont(font)
        self.btnStart.clicked.connect(self.run)

        self.lblDisplay = QLabel(self) #label to show frame from camera
        self.lblDisplay.setGeometry(10, 10, MAIN_WINDOW_WIDTH, MAIN_WINDOW_HEIGHT)
        self.lblDisplay.setStyleSheet("background-color: #000000")
        
        self.lblROI = QLabel(self) #label to show face with ROIs
        self.lblROI.setGeometry(660, 10, SECONDARY_WINDOW_WIDTH, SECONDARY_WINDOW_HEIGHT)
        self.lblROI.setStyleSheet("background-color: #000000")
        
        self.lblHR = QLabel(self) #label to show HR change over time
        self.lblHR.setGeometry(680,450,300,40)
        self.lblHR.setFont(font)

        if self.use_camera:
            th = Thread(self)
            th.changePixmap.connect(self.setImage)
            th.start()
        
        self.lblHR2 = QLabel(self) #label to show stable HR
        self.lblHR2.setGeometry(680,485,300,40)
        self.lblHR2.setFont(font)
        
        self.lblPlot = QLabel(self) #label to show plot
        
        self.statusBar = QStatusBar()
        self.statusBar.setFont(font)
        self.setStatusBar(self.statusBar)

        #event close
        self.c = Communicate()
        self.c.closeApp.connect(self.close)
                
        #config main window
        self.setGeometry(100,100,950,540)
        #self.center()
        self.setWindowTitle("Heart rate Estimation")
        self.show()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        
    def closeEvent(self, event):
        reply = QMessageBox.question(self,"Message", "Are you sure want to quit",
            QMessageBox.Yes|QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            event.accept()
            self.input.stop()
            cv2.destroyAllWindows()
        else: 
            event.ignore()
    
    def mousePressEvent(self, event):
        self.c.closeApp.emit() 

    def key_handler(self):
        """
        cv2 window must be focused for keypresses to be detected.
        """
        self.pressed = waitKey(1) & 255  # wait for keypress for 10 ms
        if self.pressed == 27:  # exit program on 'esc'
            print("[INFO] Exiting")
            self.webcam.stop()
            sys.exit()

    def reset(self):
        self.status = False
        self.btnStart.setEnabled(True)
        self.process.reset()
        self.lblDisplay.clear()
        self.lblDisplay.setStyleSheet("background-color: #000000")

    def create_input_array(self):
        self.video_files = []
        for (dirpath, dirnames, filenames) in os.walk(self.video_filefolder):
            self.video_files.extend(filenames)

        input_array = []

        self.video_files = ["Capstone.mov"]#["p10_normal.MP4", "p13_normal.MP4", "p13_normal.MP4", "p13_physical.MP4", "p14_normal.MP4", "p14_physical.MP4", "p15_normal.MP4", "p15_physical.MP4"]

        for video_file in self.video_files:
            video_filepath = os.path.join(self.video_filefolder, video_file)
            video_label = os.path.splitext(video_file)[0]
            video_groundtruth = self.groundtruth.data[video_label]
            video_input = Video(video_filepath)

            input_object = {
                'filepath': video_filepath,
                'label': video_label,
                'hr_groundtruth': video_groundtruth,
                'input': video_input,
                'hr_measured': 0,
                'error': 0
            }

            input_array.append(input_object)

        return input_array

    def make_bpm_plot(self):
    
        plotXY([[self.process.times[20:],
                     self.process.samples[20:]],
                    [self.process.freqs,
                     self.process.fft]],
                    labels=[False, True],
                    showmax=[False, "bpm"],
                    label_ndigits=[0, 0],
                    showmax_digits=[0, 1],
                    skip=[3, 3],
                    name="Plot",
                    bg=None)
    
    def run(self):
        self.reset()
        self.input = Webcam()
        self.process.reset()
        self.status = False
        self.frame = np.zeros((10,10,3),np.uint8)
        self.bpm = 0
        self.bpms = []

        if self.status == False:
            self.status = True
            self.input.start()
            self.btnStart.setEnabled(False)
            self.lblHR2.clear()
            self.main_loop_online()
        elif self.status == True:
            self.status = False
            input.stop()

    def run_dataset(self):
        for input_obj in self.input_array:
            self.input = input_obj.get('input')
            self.process.reset()
            self.status = False
            self.frame = np.zeros((10,10,3),np.uint8)
            self.bpm = 0
            self.bpms = []

            self.input.start()
            while self.input.finished == False:
                self.main_loop_offline()

            input_obj['hr_measured'] = np.mean(self.process.bpms)
            input_obj['error'] = get_change(input_obj['hr_measured'], input_obj['hr_groundtruth'])
            print("\n\nvideo is: " + input_obj['label'])
            print("hr_actual: " + str(input_obj['hr_groundtruth']))
            print("hr_measured: " + str(input_obj['hr_measured']))
            print("error: " + str(input_obj['error']) + "%")

    def update_main_window(self, frame, fps):
        cv2.putText(frame, "FPS "+str(float("{:.2f}".format(fps))),
           (20,20), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255),2)

        output_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        output_frame = cv2.resize(output_frame, (MAIN_WINDOW_WIDTH, MAIN_WINDOW_HEIGHT))

        output_image = QImage(output_frame, output_frame.shape[1], output_frame.shape[0], 
                        output_frame.strides[0], QImage.Format_RGB888)
        self.lblDisplay.setPixmap(QPixmap.fromImage(output_image))

    def update_secondary_window(self, frame):
        output_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.lblROI.setGeometry(660, 10, output_frame.shape[1], output_frame.shape[0])
        output_frame = np.transpose(output_frame,(0,1,2)).copy()
        output_image = QImage(output_frame, output_frame.shape[1], output_frame.shape[0], 
                       output_frame.strides[0], QImage.Format_RGB888)
        self.lblROI.setPixmap(QPixmap.fromImage(output_image))
    def main_loop_offline(self):
    
        frame = self.input.get_frame()

        if frame is not None:
            self.process.frame_in = frame
            self.process.run_offline_2()
            
            self.frame = self.process.frame_out #get the frame to show in GUI
            self.f_fr = self.process.frame_ROI #get the face to show in GUI
            self.bpm = self.process.bpm #get the bpm change over the time

            # plt.imshow(self.frame)
            # print("current bpm: " + str(np.mean(self.process.bpms)))

            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
            
            # if self.process.bpms.__len__() > 3:
            #     if(max(self.process.bpms[-3:]-np.mean(self.process.bpms[-3]))<5): #show HR if it is stable -the change is not over 5 bpm- for 3s
            #         print("Heart rate: " + str(float("{:.2f}".format(np.mean(self.process.bpms)))) + " bpm")
    
    def main_loop_online(self):
    
        while self.status == True:
            frame = self.input.get_frame()

            if frame is not None:
                self.process.frame_in = frame
                self.process.run_online_2()


                if not self.process.no_face:                    
                    self.update_main_window(self.process.frame_out, self.process.fps)
                    self.update_secondary_window(self.process.frame_ROI)

                    #get the bpm change over the time
                    self.bpm = self.process.bpm 
                    self.lblHR.setText("Current BPM: " + str(float("{:.2f}".format(self.bpm))))
                    
                    if self.process.bpms.__len__() >50:
                        if(max(self.process.bpms-np.mean(self.process.bpms))<5): #show HR if it is stable -the change is not over 5 bpm- for 3s
                            self.lblHR2.setText("Average BPM: " + str(float("{:.2f}".format(np.mean(self.process.bpms)))) + " bpm")

                    bg = np.zeros((480, 680, 3), np.uint8)
                    graph = plotXY([[self.process.times,
                         self.process.samples],
                        [self.process.freqs,
                         self.process.fft]],
                        labels=[False, True],
                        showmax=[False, "bpm"],
                        label_ndigits=[0, 0],
                        showmax_digits=[0, 1],
                        skip=[3, 3],
                        name="Plot",
                        bg=bg)

                    try:
                        cv2.imshow("Processed", graph)
                    except Exception as error:
                        print(error)
                        cv2.imshow("Processed", self.frame)
                else:
                    self.process.frame_ROI = np.zeros((SECONDARY_WINDOW_HEIGHT, SECONDARY_WINDOW_WIDTH, 3), np.uint8)
                    cv2.putText(self.process.frame_ROI, "No face detected",
                       (SECONDARY_WINDOW_HEIGHT/2, SECONDARY_WINDOW_WIDTH/2 - 50), cv2.FONT_HERSHEY_PLAIN, 0.75, (255, 255, 0),1)
                    
                    self.update_main_window(frame, self.input.get_fps())
                    self.update_secondary_window(self.process.frame_ROI)

                    cv2.imshow("Processed", frame)

                self.key_handler() #if not the GUI cant show anything
if __name__ == '__main__':
    #dataset_location = sys.argv[1] 
    dataset_location = '../dataset/data/'

    app = QApplication(sys.argv)

    window = GUI(False, dataset_location)
    window.show()
    # while True:
    #     window.run()

    app.exec_()
        
        
        
        
        
    