import cv2
import numpy as np
import sys
import os
import time
from process import Process
from webcam import Webcam
from video import Video
from interface import plotXY
from groundtruth import GroundTruth

def get_change(current, previous):
    if current == previous:
        return 100.0
    try:
        return (abs(current - previous) / previous) * 100.0
    except ZeroDivisionError:
        return 0

#Class to run Method1    
class TestPPG():
    def __init__(self, dataset_location):
        self.video_filefolder = os.path.join(dataset_location, 'Video')
        self.groundtruth_filepath = os.path.join(dataset_location, 'ground_truths.csv')

        self.groundtruth = GroundTruth(self.groundtruth_filepath)

        self.input_array = self.create_input_array()

    def create_input_array(self):
        self.video_files = []
        for (dirpath, dirnames, filenames) in os.walk(self.video_filefolder):
            self.video_files.extend(filenames)

        input_array = []

        self.video_files = ["p13_normal.MP4", "p13_physical.MP4", "p14_normal.MP4", "p14_physical.MP4", "p15_normal.MP4", "p15_physical.MP4"]

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
    
    def run_dataset(self):
        for input_obj in self.input_array:
            self.input = input_obj.get('input')
            self.process = Process(True, 25.0)
            self.process.reset()
            self.status = False
            self.frame = np.zeros((10,10,3),np.uint8)
            #self.plot = np.zeros((10,10,3),np.uint8)
            self.bpm = 0
            self.bpms = []

            self.input.start()
            while self.input.finished == False:
                self.main_loop()

            #input_obj['hr_measured'] = np.average(self.bpms)
            input_obj['hr_measured'] = np.mean(self.bpms)
            input_obj['error'] = get_change(input_obj['hr_measured'], input_obj['hr_groundtruth'])
            print("video is: " + input_obj['label'])
            print("hr_actual: " + str(input_obj['hr_groundtruth']))
            print("hr_measured: " + str(input_obj['hr_measured']))
            print("error: " + str(input_obj['error']) + "%")

    def main_loop(self):
    
        frame = self.input.get_frame()

        if frame is not None:
            self.process.frame_in = frame
            # self.process.run_offline()
            self.process.run_offline_jade()

            cv2.imshow("Processed", frame)
            
            self.frame = self.process.frame_out #get the frame to show in GUI
            self.f_fr = self.process.frame_ROI #get the face to show in GUI
            self.bpm = self.process.bpm #get the bpm change over the time
            
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
            
            if self.process.bpms.__len__() > 50:
                # if(max(self.process.bpms-np.mean(self.process.bpms))<5): #show HR if it is stable -the change is not over 5 bpm- for 3s
                # print("Heart rate: " + str(float("{:.2f}".format(np.mean(self.process.bpms)))) + " bpm")
                self.bpms.append(np.mean(self.process.bpms))
            #We need to open a cv2.imshow() window to handle a pause 
            #self.make_bpm_plot()
        
if __name__ == '__main__':
    #dataset_location = sys.argv[1] 
    dataset_location = '../dataset/data/'

    testppg = TestPPG(dataset_location)
    testppg.run_dataset()

    sys.exit()  
        
        
        
        
        
    