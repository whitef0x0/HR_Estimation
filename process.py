import cv2
import numpy as np
import time, math
import jade
from face_detection import FaceDetection
from scipy import signal
from sklearn.decomposition import FastICA
from sklearn.preprocessing import minmax_scale

# import matplotlib.pyplot as plt
import scipy.fftpack

class Process(object):
    def __init__(self, using_video, video_fps):
        self.using_video = using_video
        if using_video:
            self.fps = video_fps
        else:
            self.fps = 0
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        self.frame_ROI = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
        self.samples = []
        self.buffer_size = 100
        self.times = [] 
        self.data_buffer = []
        self.fps = 25.0
        self.fft = []
        self.freqs = []
        self.t0 = time.time()
        self.bpm = 0
        self.fd = FaceDetection()
        self.bpms = []
        self.peaks = []
        #self.red = np.zeros((256,256,3),np.uint8)
        
    def extractGreenColor(self, frame):
        
        #r = np.mean(frame[:,:,0])
        g = np.mean(frame[:,:,1])
        #b = np.mean(frame[:,:,2])
        #return r, g, b
        return g   

    def run(self):
        self.times.append(time.time() - self.t0)
        frame, face_frame, ROI1, ROI2, status, mask = self.fd.face_detect(self.frame_in)
        
        self.frame_out = frame
        self.frame_ROI = face_frame
        
        g1 = self.extractGreenColor(ROI1)
        g2 = self.extractGreenColor(ROI2)
        
        L = len(self.data_buffer)
        
        #calculate average green value of 2 ROIs
        #r = (r1+r2)/2
        g = g
        #b = (b1+b2)/2
        
        #remove sudden change, if the avg value change is over 10, use the mean of the data_buffer
        if(abs(g-np.mean(self.data_buffer))>10 and L>(self.buffer_size-1)): 
            g = self.data_buffer[-1]
        
        self.data_buffer.append(g)

        
        #only process in a fixed-size buffer
        if L > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]
            self.bpms = self.bpms[-self.buffer_size//2:]
            L = self.buffer_size
            
        processed = np.array(self.data_buffer)
        
        # start calculating after the first 100 frames
        if L > 15:

            #calculate HR using a true fps of processor of the computer, not the fps the camera provide
            self.fps = float(L) / (self.times[-1] - self.times[0])
            even_times = np.linspace(self.times[0], self.times[-1], L)
            
            #detrend the signal to avoid interference of light change
            processed = signal.detrend(processed)

            #Apply one-dimensional linear interpolation to data
            interpolated = np.interp(even_times, self.times, processed) #interpolation by 1

            #Apply Hamming window to signal with length of L frames
            interpolated = np.hamming(L) * interpolated

            #Normalize Data
            norm = (interpolated - np.mean(interpolated))/np.std(interpolated)
            norm = interpolated/np.linalg.norm(interpolated)

            #do real fft with the normalization multiplied by 10
            raw = np.fft.rfft(norm*30)
            
            self.freqs = float(self.fps) / L * np.arange(L / 2 + 1)
            freqs = 60. * self.freqs
            
            # idx_remove = np.where((freqs < 50) & (freqs > 180))
            # raw[idx_remove] = 0
            
            #get amplitude spectrum
            self.fft = np.abs(raw)**2
        
            #the range of frequency that HR is supposed to be within 
            idx = np.where((freqs > 40) & (freqs < 180))
            
            pruned = self.fft[idx]
            pfreq = freqs[idx]
            
            self.freqs = pfreq 
            self.fft = pruned
            
            #max in the range can be HR
            idx2 = np.argmax(pruned)
            
            self.bpm = self.freqs[idx2]
            self.bpms.append(self.bpm)
        
            processed = self.butter_bandpass_filter(processed,0.8,3,self.fps,order=3)
            #ifft = np.fft.irfft(raw)
        self.samples = processed 
        # multiply the signal with 5 for easier to see in the plot
        
        # if(mask.shape[0]!=10): 
        #     out = np.zeros_like(face_frame)
        #     mask = mask.astype(np.bool)
        #     out[mask] = face_frame[mask]
        #     if(processed[-1]>np.mean(processed)):
        #         out[mask,2] = 180 + processed[-1]*10
        #     face_frame[mask] = out[mask]
            
            
        #cv2.imshow("face", face_frame)
        #out = cv2.add(face_frame,out)
        # else:
            # cv2.imshow("face", face_frame)

    def run_offline(self):
        frame, face_frame, ROI1, ROI2, ROI3, status, mask = self.fd.face_detect(self.frame_in)
        
        self.frame_out = frame
        self.frame_ROI = face_frame
        
        g1 = self.extractGreenColor(ROI1)
        g2 = self.extractGreenColor(ROI2)
        g3 = self.extractGreenColor(ROI3)
        
        L = len(self.data_buffer)

        # print("ROI1.shape: " + str(ROI1.shape[1]))
        # print("ROI2.shape: " + str(ROI2.shape[1]))
        
        chin1_width = ROI1.shape[1]
        chin2_width = ROI2.shape[1]
        total_cheek_width = chin1_width + chin2_width
        # print("chin1_width: " + str(chin1_width))
        # print("chin2_width: " + str(chin2_width))

        #calculate average green value of 2 ROIs
        #r = (r1+r2)/2

        if chin1_width / total_cheek_width >= 0.6:
            g = g1
        elif chin2_width / total_cheek_width >= 0.6:
            g = g2
        else:
            g = (g1+g2)/2
        
        #remove sudden change, if the avg value change is over 10, use the mean of the data_buffer
        if(abs(g-np.mean(self.data_buffer))>10 and L>(self.buffer_size-1)): 
            g = self.data_buffer[-1]
        
        self.data_buffer.append(g)

        
        #only process in a fixed-size buffer
        if L > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            self.bpms = self.bpms[-self.buffer_size//2:]
            L = self.buffer_size
            
        processed = np.array(self.data_buffer)
        
        # start calculating after the first 25 frames
        if L > 10:            
            #detrend the signal to avoid interference of light change
            processed = signal.detrend(processed)

            #Apply Hamming window to signal with length of L frames
            interpolated = np.hamming(processed.shape[0]) * processed

            #Normalize Data
            norm = (interpolated - np.mean(interpolated))/np.std(interpolated)
            norm = interpolated/np.linalg.norm(interpolated)

            #do real fft with the normalization multiplied by 10
            raw = np.fft.rfft(norm*30)
            
            self.freqs = float(self.fps) / L * np.arange(L / 2 + 1)
            freqs = 60. * self.freqs
            
            #get amplitude spectrum
            self.fft = np.abs(raw)**2
        
            #the range of frequency that HR is supposed to be within 
            idx = np.where((freqs > 40) & (freqs < 180))
            pruned = self.fft[idx]
            pfreq = freqs[idx]
            
            self.freqs = pfreq 
            self.fft = pruned
            
            #max in the range can be HR
            idx2 = np.argmax(pruned)
            
            self.bpm = self.freqs[idx2]
            self.bpms.append(self.bpm)

        self.samples = processed 

    def extractFrequency(self, L, fftArray, framerate):
        timestep = 1/framerate

        freqs = np.fft.fftfreq(L, d=timestep)

        # calculates the power spectra and returns 
        # the frequency, in Hz, of the most prominent 
        # frequency between 0.67 Hz and 3 Hz (40 - 180 bpm)
        filteredFFT = []
        filteredFreqBin = []

        freqObj = dict(zip(freqs, fftArray))
        for freq in freqObj:
            if freq > 0.67 and freq < 3:
                filteredFFT.append(freqObj[freq])
                filteredFreqBin.append((freq)/1)

        def normalizeFreqFunc(n):
            return math.pow(math.fabs(n), 2)

        filteredFFT = [normalizeFreqFunc(f) for f in filteredFFT]
        normalizedFreqs = np.array(filteredFFT)
        idx = np.argmax(normalizedFreqs)
        freq_in_hertz = filteredFreqBin[idx]
        
        freqs = {'normalizedFreqs': normalizedFreqs, 'filteredFreqBin': filteredFreqBin, 'freq_in_hertz': freq_in_hertz}
        return freqs

    def extractAllColors(self, frame): 
        r = np.mean(frame[:,:,0])
        g = np.mean(frame[:,:,1])
        b = np.mean(frame[:,:,2])
        return [r, g, b]

    def normalize_matrix(self, matrix):
        # ** for matrix
        for array in matrix:
            average_of_array = np.mean(array)
            std_dev = np.std(array)

            for i in range(len(array)):
                array[i] = ((array[i] - average_of_array)/std_dev)
        return matrix

    def parse_ICA_results(self, ICA, buffer_window): #time
        signals = {}
        signals["bufferWindow"] = buffer_window

        # ** FOR RGB CHANNELS & ICA **
        one = np.squeeze(np.asarray(ICA[:, 0])).tolist()
        two = np.squeeze(np.asarray(ICA[:, 1])).tolist()
        three = np.squeeze(np.asarray(ICA[:, 2])).tolist()
        
        one = (np.hamming(len(one)) * one)
        two = (np.hamming(len(two)) * two)
        three = (np.hamming(len(three)) * three)

        one = np.absolute(np.square(np.fft.irfft(one))).astype(float).tolist()
        two = np.absolute(np.square(np.fft.irfft(two))).astype(float).tolist()
        three = np.absolute(np.square(np.fft.irfft(three))).astype(float).tolist()

        power_ratio = [0, 0, 0]
        power_ratio[0] = np.sum(one)/np.amax(one)
        power_ratio[1] = np.sum(two)/np.amax(two)
        power_ratio[2] = np.sum(three)/np.amax(three)

        if np.argmax(power_ratio) == 0:
            signals["array"] = one
        elif np.argmax(power_ratio) == 1:
            signals["array"] = two
        else:
            signals["array"] = three

        return signals

    def run_offline_jade(self):
        frame, face_frame, ROI1, ROI2, ROI3, status, mask = self.fd.face_detect(self.frame_in)
        
        self.frame_out = frame
        self.frame_ROI = face_frame
        
        L = len(self.ica_data_buffer[0])

        colors = self.extractAllColors(ROI1)
        
        #remove sudden change, if the avg value change is over 10, use the mean of the ica_data_buffer
        if(abs(colors[0]-np.mean(self.ica_data_buffer[0]))>10 and L>(self.buffer_size-1)): 
            for i in range(3):
                colors[i] = self.ica_data_buffer[i][-1]
        
        for i in range(3):
            self.ica_data_buffer[i].append(colors[i])

        #only process in a fixed-size buffer
        if L > self.buffer_size:
            for i in range(3):
                self.ica_data_buffer[i] = self.ica_data_buffer[i][-self.buffer_size:]
            self.bpms = self.bpms[-self.buffer_size//2:]
            L = self.buffer_size

        # start calculating after the first 30 frames
        if L > 30:
            processed = np.ndarray(shape=(3, L), buffer=np.array(self.ica_data_buffer))     

            #Do JADE ICA
            processed = self.normalize_matrix(processed)
            ICA = jade.main(processed)

            parsedICADict = self.parse_ICA_results(ICA, L)

            resultDict = self.extractFrequency(L, parsedICADict["array"], 25)

            self.bpm = resultDict["freq_in_hertz"] * 60
            self.bpms.append(self.bpm)
    
    def reset(self):
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        self.frame_ROI = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
        self.samples = []
        self.times = [] 
        self.data_buffer = []
        self.ica_data_buffer = [[], [], []]
        if self.using_video:
            self.fps = 25.0
        else:
            self.fps = 0
        self.fft = []
        self.freqs = []
        self.t0 = time.time()
        self.bpm = 0
        self.bpms = []
        
    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = signal.lfilter(b, a, data)
        return y    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        