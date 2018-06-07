import numpy as np
import csv, random

class feature(object):

    def __init__(self, filepath):
        self.__data = []
        self.__label = []
        self.filepath = filepath

    def preprocess(self):

        # Load in data
        with open(self.filepath, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.__label.append(row['label'])
                self.__data.append(np.array(row['feature'].split(" ")))

        # Reshape feature into (#, 48, 48)       
        self.__data = np.reshape(self.__data, (len(self.__data), 48, 48, 1))
        self.__label = np.reshape(self.__label, (len(self.__label), 1))
        return self

    def normalize(self):
        '''
        Normalize feature by dividing by 255
        '''
        self.__data = (self.__data).astype('float32') / 255.0
        return self

    def sample_val(self, size):
        '''
        Random sample validation set of given size from self.
        '''
        seed = random.sample(range(self.__data.shape[0]), size)
        
        val_f = self.__data[seed]
        val_l = self.__label[seed]
        self.__data = np.delete(self.__data, seed, axis=0)
        self.__label = np.delete(self.__label, seed, axis=0)
    
        return val_f, val_l, seed

    def delete(self, seed):
        self.__data = np.delete(self.__data, seed, axis=0)
        self.__label = np.delete(self.__label, seed, axis=0)
        
        return self

    def __getitem__(self, index):
        return self.__data[index, ]
    
    def __len__(self):
        return self.__data.shape[0]
    
    def get_feature(self):
        return self.__data
    
    def get_label(self):
        return self.__label

 
class testfeature(object):
    
    def __init__(self, filepath):
        self.__data = []
        self.__label = []
        self.filepath = filepath

    def preprocess(self):
        
        # Load in data
        with open(self.filepath, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.__label.append(row['id'])
                self.__data.append(np.array(row['feature'].split(" ")))
        # Reshape feature into (#, 48, 48)       
        self.__data = np.reshape(self.__data, (len(self.__data), 48, 48, 1))
        self.__label = np.reshape(self.__label, (len(self.__label), 1))
        return self
    
    def normalize(self):
        self.__data = (self.__data).astype('float32') / 255.0
        return self

    def __getitem__(self, index):
        return self.__data[index, ]
    
    def __len__(self):
        return self.__data.shape[0]
    
    def get_feature(self):
        return self.__data
    
    def get_label(self):
        return self.__label
