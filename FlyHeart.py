import os
from os.path import normpath, basename
import pandas as pd
import numpy as np
import torch
import pickle
import cv2
from unet import AttU_Net, IntervalsModel


class Heart():
    def __init__(
            self, 
            filepath,
            recording_fps,
            um_to_px,
            save_folder,
            num_frames=None
    ):
        
        self.filepath = filepath
        self.save_folder = save_folder
        self.name = os.path.splitext(basename(normpath(filepath)))[0]
        self.directory = os.path.dirname(filepath)
        self.num_frames  = num_frames
        self.fps = recording_fps
        self.um_to_px = um_to_px

        self.modelFileName = "./checkpoints/54V_HD75_ATT_3c27.pth"
        
        if os.path.exists(f"{self.save_folder}{self.name}/") is False:
            os.mkdir(f"{self.save_folder}{self.name}/")
        
        self.params = {
            "DD": None,
            "SD": None,
            "FS": None,
            "EF": None,
            "DI": None,
            "SI": None,
            "HP": None,
            "HR": None,
            "AI": None
        }
        
        # Read video frames
        cap = cv2.VideoCapture(self.filepath) 
        
        self.frames = []
               
        success, image = cap.read()
            
        while success:

            image = image[:, :, 0]
            padded = np.zeros((256, 512))
            start_h = (padded.shape[0] - image.shape[0]) // 2
            start_w = (padded.shape[1] - image.shape[1]) // 2
            
            padded[start_h:start_h + image.shape[0], start_w:start_w+image.shape[1]] = image
            
            self.frames.append(padded)
            success, image = cap.read()
            
            #check if we have collected num_frames
            if self.num_frames is not None and len(self.frames) == self.num_frames:
                break
               
        self.frames = np.array(self.frames)
        
        self.originalProbs = None
        self.probs = None
        self.predictedMasks = None
        self.roi = None #roi for timeseries
        self.timeseries = None
        self.times = None
        self.degree = 10
        self.intervalWindow = 100
        self.Intervals = None
    
    def get_params(self):
        return self.params, self.timeseries, self.Intervals
    
    def annotate(self):
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = AttU_Net(img_ch=3, output_ch=1)
        model.to(device=device)
        model.load_state_dict(torch.load(self.modelFileName, map_location=device))
        model.eval() 
                
        torch.cuda.empty_cache()
        self.confidence_maps = []
        
        for i in range(self.frames.shape[0]):
            
            img = np.zeros((3, 256, 512))
            img[0, :, :] = self.frames[max(i-4, 0), :, :]
            img[1, :, :] = self.frames[i, :, :] 
            img[2, :, :] = self.frames[min(i+4, self.frames.shape[0]-1), :, :]
    
            img = img / 255
            img = torch.from_numpy(img) 
            img = img.unsqueeze(0) # 1 sample
            img = img.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                preds = model(img)
                preds = torch.sigmoid(preds)
                preds = preds.detach().cpu().numpy()
                preds = preds[0, 0, :, :]
            
            self.confidence_maps.append(preds)
        self.confidence_maps = np.array(self.confidence_maps)
        return self.confidence_maps
            
    def get_timeseries(self, roi, confidence_thresh):
        
        self.roi = roi
        self.threshold = confidence_thresh
  
        masks = self.confidence_maps > confidence_thresh
        
        bool_cnt = np.count_nonzero(masks[:, :, roi[0]:roi[1]], axis = 1)
        areas = np.sum(bool_cnt, axis = 1)
        diameters = areas / np.count_nonzero(bool_cnt, axis = 1)
        
        frames = np.arange(len(diameters))
        times = frames / self.fps
        diameters_um = diameters / self.um_to_px
        areas_um2 = areas / (self.um_to_px ** 2)
        vel_um = np.gradient(diameters_um)
        accel_um = np.gradient(vel_um)
        
        timeseries = {
            'Time(s)': times,
            'Frame': frames,
            'Avg Diameter(px)': diameters,
            'Avg Diameter(um)': diameters_um,
            'Area(px^2)': areas,
            'Area(um^2)': areas_um2,
            'Diam Velocity(um/s)': vel_um,
            'Diam Acceleration(um/s^2)': accel_um
        }
        
        self.timeseries = pd.DataFrame.from_dict(timeseries)        
        return self.timeseries
    
    def calculate_parameters(self):
        
        sampleLen = 1000
        net = IntervalsModel(sampleLen)
        net.load_state_dict(torch.load("./checkpoints/SImodelCP(working).pth", map_location=None))
        
        predDiff = np.array([])
        fps = self.fps
        diamdata = np.array(self.timeseries["Avg Diameter(um)"])

        diamdata = (diamdata - np.min(diamdata))/(np.max(diamdata) - np.min(diamdata))
        n = len(diamdata) // sampleLen
        for i in range(n+1):
            start = i*sampleLen
            if i == n:
                X = np.ones(shape = sampleLen) * diamdata[-1]
                remainder = len(diamdata)-start
                X[:remainder] = diamdata[start:]
            else:
                X = np.array(diamdata[start:start+sampleLen])
            X = (X - np.min(X)) / (np.max(X) - np.min(X))
            X = torch.FloatTensor(X)
            X = torch.reshape(X, (1,sampleLen))
            samplePred = net(X) 
            samplePred = torch.sigmoid(samplePred).detach().numpy()
            samplePred = samplePred.reshape(sampleLen)
            samplePred[np.where(samplePred > 0.5)] = 1
            samplePred[np.where(samplePred != 1)] = 0 
            samplePredDiff = np.array([samplePred[i]-samplePred[i-1] for i in range(1, sampleLen)])
            predDiff = np.concatenate((predDiff, samplePredDiff))

        predDiff = predDiff[:len(diamdata)].astype(int)
        #Search and remove same consecutive events 
        nonzero = np.where(predDiff != 0)[0].astype(int)
        for i in range(len(nonzero)-1):
            idx = nonzero[i]
            nidx = nonzero[i+1]
            if predDiff[idx] == 1: # DIend DIstart DIstart DIend
                if predDiff[nidx] == 1:
                    end1 = 0 if i == 0 else nonzero[i-1]
                    end2 = len(predDiff) if i == len(nonzero)-2 else nonzero[i+2]
                    SI = idx - end1
                    DI = end2 - nidx
                    if SI > DI:
                        predDiff[nidx] = 0
                    else:
                        predDiff[idx] = 0
            elif predDiff[idx] == -1: #DIstart DIend DIend DIstart
                if predDiff[nidx] == -1:
                    start2 = len(predDiff) if i == len(nonzero)-2 else nonzero[i+2]
                    SI = start2 - nidx
                    DI = idx - start1
                    if SI > DI:
                        predDiff[idx] = 0
                    else:
                        predDiff[nidx] = 0

        nonzero = np.where(predDiff != 0)[0].astype(int)
        for i in range(len(nonzero)-1): # remove unrealistically small intervals (less than 0.1 seconds)
            idx = nonzero[i]
            nidx = nonzero[i+1]
            if predDiff[idx] == 1: #during diastol
                if nidx-idx < 0.075*fps:
                    predDiff[idx] = 0
                    predDiff[nidx] = 0 
            elif predDiff[idx] == -1: #during diastol
                if nidx-idx < 0.075*fps:
                    predDiff[idx] = 0
                    predDiff[nidx] = 0
                if np.min(diamdata[idx:nidx]) > 0.3:
                    predDiff[idx] = 0
                    predDiff[nidx] = 0


        dStart = np.where(predDiff == 1)[0].astype(int)
        dEnd = np.where(predDiff == -1)[0].astype(int)

        if dEnd[0]<dStart[0]: # make sure first event is dstart
            dEnd = np.delete(dEnd,0)
        if dStart[-1] > dEnd[-1]: # make sure last event is dend
            dStart = np.delete(dStart, -1)

        dStart = dStart + 7 # makes results slightly better
        diamdata = np.array(self.timeseries["Avg Diameter(um)"])

        distol = []
        systol = []
        peaks= []
        peakIdxs = []
        valleys = []
        valleyIdxs = []

        for i in range(len(dStart)):
            distol.append((dEnd[i]-dStart[i])/fps) 
            peaks.append(np.amax(diamdata[dStart[i]:dEnd[i]]))
            peakIdxs.append(np.argmax(diamdata[dStart[i]:dEnd[i]]) + dStart[i])

        for i in range(len(dStart)-1):
            systol.append((dStart[i+1] - dEnd[i])/fps)
            valleys.append(np.amin(diamdata[dEnd[i]:dStart[i+1]]))
            valleyIdxs.append(np.argmin(diamdata[dEnd[i]:dStart[i+1]]) + dEnd[i])

        for i in range(len(valleyIdxs)):
            if i == 0:
                start = 0
            else:
                start = valleyIdxs[i-1]
            peaks.append(np.amax(diamdata[start:valleyIdxs[i]]))
            peakIdxs.append(np.argmax(diamdata[start:valleyIdxs[i]]) + start)

        if len(valleys) == 0:
            print(h, dStart.shape, dEnd.shape)
            plt.figure(figsize = (18,3), dpi = 150)
            plt.plot(predDiff)
            plt.show()

        rcs = [(peaks[i]-valleys[i])/peaks[i] for i in range(len(valleys))] #Radial Contraction
        efs = [(peaks[i]**2-valleys[i]**2)/peaks[i]**2 for i in range(len(valleys))] # Ejection Fractions

        peaks = np.array(peaks)
        peakIdxs = np.array(peakIdxs)
        valleys = np.array(valleys)
        valleyIdxs = np.array(valleyIdxs)

        self.params["DD"] = {
            "idxs": peakIdxs,
            "times": peakIdxs/fps,
            "vals": peaks,
            "avg": peaks.mean(),
            "std": peaks.std()
        }
        self.params["SD"] = {
            "idxs": valleyIdxs,
            "times": valleyIdxs/fps,
            "vals": valleys,
            "avg": valleys.mean(),
            "std" :valleys.std()
        }
        self.params["FS"] = {
            "vals": rcs,
            "avg": np.array(rcs).mean(),
            "std": np.array(rcs).std()
        }
        self.params["EF"] = {
            "vals": efs,
            "avg": np.array(efs).mean(),
            "std": np.array(efs).std()
        }

        hp = np.array([dStart[i+1]-dStart[i] for i in range(len(dStart)-1)])/fps
        hr = 1/np.mean(hp)
        ai = np.std(hp)/np.median(hp)

        self.intervals = {
            "Diastolic Start": dStart,
            "Diastolic End": dEnd
        }
        self.params["DI"] = {
            "vals": distol,
            "avg": np.array(distol).mean(),
            "std": np.array(distol).std()
        }
        self.params["SI"] = {
            "vals": systol,
            "avg": np.array(systol).mean(),
            "std": np.array(systol).std()
        }
        self.params["HP"] = {
            "vals": hp,
            "avg": np.array(hp).mean(),
            "std": np.array(hp).std()
        }
        self.params["HR"] = {
            "avg": hr,
            "std": self.params["HP"]["std"]/(np.array(hp).mean()**2)
        }
        self.params["AI"] = {
            "val": ai
        }

                                    
    def save(self, folder = None):
        
        if folder is None:
            
            self.timeseries.to_csv(f'{self.save_folder}{self.name}/timeseries.csv', header=True, index=False) # CHECK
            
            path = f'{self.save_folder}{self.name}/'        
            params_str = f"""
                                 fileName = {self.filepath}
                                 name = {self.name}
                                 num_frames = {self.num_frames}
                                 fps = {self.fps}
                                 um_to_px = {self.um_to_px}
                                 roi = {self.roi}
                                 threshold = {self.threshold}
                                 modelName = {self.modelFileName}
                          """
            with open(f'{path}{self.name}_PARAMS.txt', "w") as text_file:
                print(params_str, file=text_file)
            file = open(f'{path}{self.name}.heart', 'wb')
            pickle.dump(self, file)
            file.close()    

        else:
            
            self.save_folder = folder
            if os.path.exists(f"{self.save_folder}{self.name}/") is False:
                os.mkdir(f"{self.save_folder}{self.name}/")
            
            self.save(folder = None)

            
