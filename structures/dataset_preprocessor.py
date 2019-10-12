#!python3


import pandas as pd
from os import listdir
from os.path import isfile, join


class Dataset_Preprocessor(object):
    """ Class object instance for bioassay dataset preprocessing analysis. """
    def __init__(self):
        """ Initializer method. """
        self.REL_PATH_TO_DATA = "../datasets/external/bioassay-datasets/"
        
    def load_data(self, which="all", delimiter="_"):
        """ 
        Instance method that conditionally loads in directed datasets into tree-based dictionary hierarchy. 
        
        INPUTS: 
            {which}:
                - str(train): Read in training dataset(s).
                - str(test): Read in testing dataset(s).
                - str(full): Read in concatenated training and testing dataset(s).
                - str(all): Read in all three (training, testing, full) dataset(s) separated by sublevel keys. (DEFAULT)
            {delimiter}:
                - str(_): Sets dataset filename delimiter as underscore symbol. (DEFAULT)
                
        OUTPUTS:
            dict: Tree-based dictionary with key-value pairs of bioassay IDs and their associated datasets (pd.DataFrame).
        """
        # Validate conditional data loading arguments
        if which not in ["all", "both", "train", "test"]:
            raise ValueError("ERROR: Inappropriate value passed to argument `which`.\n\nExpected value in range:\n - all\n - both\n - train\n - test\n\nActual:\n - {}".format(which))
        
        if which in ["train", "test", "full"]:
            raise NotImplementedError("ERROR: Value passed to argument is currently not implemented and does not function. Please try again later or alter your keyword argument value submission.")
        
        # Validate data delimiting arguments
        if type(delimiter) is not str:
            raise ValueError("ERROR: Inappropriate data type passed to argument `delimiter`.\n\nExpected type(s):\n - [str]\n\nActual:\n - [{}]".format(str(type(delimiter))))
            
        FILENAMES = list()
        for file in listdir(self.REL_PATH_TO_DATA):
            filename = file.split(delimiter)[0]
            if isfile(join(self.REL_PATH_TO_DATA, file)) and filename not in FILENAMES:
                FILENAMES.append(filename)
                
        if which == "all":
            DATASETS = dict.fromkeys(FILENAMES,
                                     dict.fromkeys(["train", "test", "full"]))
            for filename in FILENAMES:
                files = [file for file in listdir(self.REL_PATH_TO_DATA) if file.startswith(filename)]
                if files[0].endswith("train.csv"):
                    train_data, test_data = files[0], files[1]
                elif files[0].endswith("test.csv"):
                    train_data, test_data = files[1], files[0]
                    
                DATASETS[filename]["train"] = pd.read_csv(self.REL_PATH_TO_DATA + train_data)
                DATASETS[filename]["test"] = pd.read_csv(self.REL_PATH_TO_DATA + test_data)
                DATASETS[filename]["full"] = pd.concat([DATASETS[filename]["train"], DATASETS[filename]["test"]], 
                                                       keys=["train", "test"])
        return DATASETS
    
    def encode_feature(self, dataset_structure, old_feature, new_feature, encoding_map):
        """ 
        Instance method that encodes data from all datasets to new features using a map structure.
        
        INPUTS:
            {dataset_structure}:
                - dict: Input tree-based dictionary structure containing datasets to iterate over for feature encoding. 
            {old_feature}:
                - str: Name of original feature across all input datasets over which to encode.
            {new_feature}:
                - str: Name of new feature to generate across all input datasets.
            {encoding_map}:
                - dict: Accumulation of key-value pairs, where occurrences of keys are replaced with values across observed feature.
        
        OUTPUTS:
            NoneType: Iterative in-place data encoding does not return any new object(s).
        """
        for key, value in dataset_structure.items():
            if isinstance(value, dict):
                self.encode_feature(value, old_feature, new_feature, encoding_map)
            else:
                value[new_feature] = value[old_feature].map(encoding_map)