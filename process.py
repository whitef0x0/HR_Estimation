import cv2
import numpy as np
import time, math

import jade
from face_detection import FaceDetection
from graph import Graph 


import scipy
from scipy import signal
from sklearn.decomposition import FastICA
from sklearn.preprocessing import minmax_scale

from sys import platform as sys_pf
import matplotlib.pyplot as plt
from scipy import fftpack

class Process(object):
    def __init__(self, using_video, video_fps):
        self.using_video = using_video
        if using_video:
            self.fps = video_fps
        else:
            self.fps = 0
        self.reset()


    def reset(self):
        if not self.using_video:
            self.fps = 0
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        self.frame_ROI = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
        self.samples = {
            "processed": [],
            "raw_fft": [],
            "unprocessed": []
        }
        self.buffer_size = 400
        self.times = [] 
        self.data_buffer = []

        self.data_buffer_roi_mean = [[], [], []]

        self.ica_data_buffer = [[], [], []]
        self.fft = []
        self.freqs = []
        self.t0 = time.time()
        self.bpm = 0
        self.fd = FaceDetection()
        self.bpms = []
        self.peaks = []
        self.peak_freqs = []
    
    # Extracts Green Color from ROI, ignoring values that have been set to black by mask
    def extractGreenColorMask(self, frame):
        green_pixels = frame[:,:,1].ravel()
        green_masked_pixels = green_pixels[green_pixels > 0]
        g = np.mean(green_masked_pixels)
        return g   

    def extractGreenColor(self, frame):
        g = np.mean(frame[:,:,1])
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
        g = (g1+g2)/2
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
        self.samples["processed"] = processed

    def run_offline_1(self):
        frame, face_frame, ROI1, ROI2, status, mask = self.fd.face_detect(self.frame_in)
        
        self.frame_out = frame
        self.frame_ROI = face_frame

        # cv2.imshow("face align", frame)
        # cv2.waitKey()
        
        g1 = self.extractGreenColor(ROI1)
        g2 = self.extractGreenColor(ROI2)
        
        L = len(self.data_buffer)
        
        chin1_width = ROI1.shape[1]
        chin2_width = ROI2.shape[1]
        total_cheek_width = chin1_width + chin2_width
        # print("chin1_width: " + str(chin1_width))
        # print("chin2_width: " + str(chin2_width))

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
            self.times = self.times[-self.buffer_size:]
            self.bpms = self.bpms[-self.buffer_size//2:]
            L = self.buffer_size
            
        processed = np.array(self.data_buffer)
        
        # start calculating after the first 100 frames
        if L > 100:
            # frame_period = 1/self.fps
            # x_array = np.arange(0, (L+1)*frame_period, frame_period)      
            data_graph = Graph("Data Graph", 4)
            data_graph.addSubPlot(title="Signal vs Time", x_axis_title="Time", y_axis_title="Signal", x_data=range(len(processed)), y_data=processed)

            #detrend the signal to avoid interference of light change
            detrended = signal.detrend(processed)
            data_graph.addSubPlot(title="Signal vs Time (Detrended)", x_axis_title="Time", y_axis_title="Signal", x_data=range(len(detrended)), y_data=detrended)


            #Apply Hamming window to signal with length of L frames
            interpolated = np.hamming(detrended.shape[0]) * detrended
            data_graph.addSubPlot(title="Signal vs Time (Detrended and Hamming Window)", x_axis_title="Time", y_axis_title="Signal", x_data=range(len(interpolated)), y_data=interpolated)

            #Normalize Data
            # normalized = (interpolated - np.mean(interpolated))/np.std(interpolated)
            normalized = interpolated/np.linalg.norm(interpolated)
            data_graph.addSubPlot(title="Signal vs Time (Normalized Detrended and Hamming Window)", x_axis_title="Time", y_axis_title="Signal", x_data=range(len(normalized)), y_data=normalized)
        
            #do real fft with the normalization multiplied by 30
            raw = np.fft.rfft(normalized*30)


            freq = np.fft.rfftfreq(L, 1/30)*60
            
            self.freqs = float(self.fps) / L * np.arange(L / 2 + 1)
            freqs = 60. * self.freqs
            
            #get amplitude spectrum
            self.fft = np.abs(raw)**2
            fft_graph = Graph("FFT Graph", 2)
            fft_graph.addSubPlot(title="Raw FFT", x_axis_title="frequency", y_axis_title="magnitude", x_data=freqs, y_data=self.fft)
           

            #the range of frequency that HR is supposed to be within 
            idx = np.where((freqs > 40) & (freqs < 180))
            
            pruned = self.fft[idx]
            pfreq = freqs[idx]
            
            fft_graph.addSubPlot(title="Bandpass Filter FFT", x_axis_title="frequency", y_axis_title="magnitude", x_data=freqs, y_data=pruned)
            plt.show()
            cv2.waitKey()

            self.freqs = pfreq 
            self.fft = pruned
            
            #max in the range can be HR
            idx2 = np.argmax(pruned)
            
            self.bpm = self.freqs[idx2]
            self.bpms.append(self.bpm)
        

            processed = self.butter_bandpass_filter(processed, 0.8, 3, self.fps, order=3)

    #Implementation of Gaussian Smoothing Filter for 1D Array
    #Taken from https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    def smooth(self, x,window_len=11,window='hanning'):
        """smooth the data using a window with requested size.
        
        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal 
        (with the window size) in both ends so that transient parts are minimized
        in the begining and end part of the output signal.
        
        input:
            x: the input signal 
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.

        output:
            the smoothed signal
            
        example:

        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)
        
        see also: 
        
        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
        scipy.signal.lfilter
     
        TODO: the window parameter could be the window itself if an array instead of a string
        NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
        """

        if x.ndim != 1:
            raise ValueError("smooth only accepts 1 dimension arrays.")

        if x.size < window_len:
            raise ValueError("Input vector needs to be bigger than window size.")


        if window_len<3:
            return x


        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


        s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
        #print(len(s))
        if window == 'flat': #moving average
            w=np.ones(window_len,'d')
        else:
            w=eval('np.'+window+'(window_len)')

        y=np.convolve(w/w.sum(),s,mode='valid')
        return y

    def calculate_skin_features(self, frames):
        Rn = frames[0]/np.mean(frames[0])
        Gn = frames[1]/np.mean(frames[1])
        Bn = frames[2]/np.mean(frames[2])

        Xf = []
        Yf = []

        for i in range(len(Rn)):
            X = 3*Rn[i] - 2*Gn[i]
            Y = 1.5*Rn[i] + Gn[i] - 1.5*Bn[i]

            Xf.append(X)
            Yf.append(Y)

        alpha = np.std(Xf)/np.std(Yf)
        S = []

        for i in range(len(Xf)):
          Sn = Xf[i] - alpha*Yf[i]
          S.append(Sn)  

        return S

    def calculate_new_freq(self, raw_fft, freqs):
        bpm_std = np.std(self.peak_freqs)
        bpm_mean = np.mean(self.peak_freqs)
        std_0 = 0.1

        Fw_old = raw_fft
        Fw_new = []

        Xi = 1/((bpm_std + std_0) * math.sqrt(2*math.pi))
        Yi = 2*((bpm_std + std_0)**2)

        # print("bpm_mean: " + str(bpm_mean))
        # print("bpm_std: " + str(bpm_std))
        # print("Xi: " + str(Xi))
        # print("Yi: " + str(Yi))

        for i in range(len(raw_fft)):
            Wi =  freqs[i]
            Pi = Xi * math.exp(-((Wi-bpm_mean)**2)/Yi)

            # print("Wi: " + str(Wi))
            # print("exponent: " + str(-(Wi-bpm_mean)**2))
            # print("Pi " + str(Pi))

            Fi_new = Fw_old[i] * Pi
            Fw_new.append(Fi_new)    
        return Fw_new

    # See http://vipl.ict.ac.cn/uploadfile/upload/2017122111573043.pdf for implementation details/inspiration
    def run_offline_2(self):
        frame, face_frame, ROI1, ROI2, roi1_mean, roi2_mean, status, mask = self.fd.face_detect(self.frame_in, use_skin_detector=True)
        
        self.frame_out = frame
        self.frame_ROI = face_frame
        
        L = len(self.data_buffer_roi_mean[0])

        chin1_width = ROI1.shape[1]
        chin2_width = ROI2.shape[1]
        total_cheek_width = chin1_width + chin2_width

        current_mean = [0, 0, 0]
        for i in range(3):
            if chin1_width / total_cheek_width >= 0.6:
                current_mean[i] = roi1_mean[i]
            elif chin2_width / total_cheek_width >= 0.6:
                current_mean[i] = roi2_mean[i]
            else:
                current_mean[i] = (roi1_mean[i] + roi2_mean[i])/2
        
        for i in range(3):
            self.data_buffer_roi_mean[i].append(current_mean[i])

        #only process in a fixed-size buffer
        if L > self.buffer_size:
            for i in range(3):
                self.data_buffer_roi_mean[i] = self.data_buffer_roi_mean[i][-self.buffer_size:]
            self.bpms = self.bpms[-self.buffer_size//2:]
            L = self.buffer_size
        
        # start calculating after the first 100 frames
        if L > 200:  
            processed = self.calculate_skin_features(self.data_buffer_roi_mean)

            data_graph = Graph("Data Graph", 4)
            data_graph.addSubPlot(title="Signal vs Frames", x_axis_title="Frames", y_axis_title="Signal", x_data=range(len(processed)), y_data=processed)

            processed = self.smooth(np.array(processed), window_len=5, window='hamming')
            data_graph.addSubPlot(title="Signal vs Frames (Gaussian Filter)", x_axis_title="Frames", y_axis_title="Signal", x_data=range(len(processed)), y_data=processed)

            #detrend the signal to avoid interference of light change
            detrended = signal.detrend(processed)
            data_graph.addSubPlot(title="Signal vs Frames (Detrended)", x_axis_title="Frames", y_axis_title="Signal", x_data=range(len(detrended)), y_data=detrended)

            #Normalize Data
            normalized = detrended/np.linalg.norm(detrended)
            data_graph.addSubPlot(title="Signal vs Frames (Normalized)", x_axis_title="Frames", y_axis_title="Signal", x_data=range(len(normalized)), y_data=normalized)
            # plt.show()

            #FFT of the signal
            sig_fft = fftpack.fft(normalized)
            
            #Power of the signal
            power = np.abs(sig_fft)

            #Corresponding frequencies
            sample_freq = fftpack.fftfreq(normalized.size, d=(1/self.fps))

            old_power = []
            if len(self.bpms) > 0:
                old_power = power
                power = np.array(self.calculate_new_freq(power, sample_freq))

            # Find the peak frequency: we can focus on only the positive frequencies
            pos_mask = np.where((sample_freq > 0.9) & (sample_freq < 4))
            freqs = sample_freq[pos_mask]

            # print("\n\n")
            # print(power)
            # print(pos_mask)
            # print(freqs)

            peak_freq = freqs[power[pos_mask].argmax()]

            if len(self.bpms) > 0:
                fft_graph = Graph("FFT Graph", 2)
                fft_graph.addSubPlot(title="Original FFT", y_axis_title="Power", x_axis_title="Frequency [Hz]", x_data=sample_freq[pos_mask], y_data=old_power[pos_mask])

                fft_graph.addSubPlot(title="Modified FFT", y_axis_title="Power", x_axis_title="Frequency [Hz]", x_data=sample_freq[pos_mask], y_data=power[pos_mask])
                plt.show()
                cv2.waitKey()


            self.bpm = peak_freq * 60
            print("\nbpm: " + str(self.bpm))
            self.bpms.append(self.bpm)
            self.peak_freqs.append(peak_freq)
             
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
        frame, face_frame, ROI1, ROI2, status, mask = self.fd.face_detect(self.frame_in)
        
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        