import os
import shutil
import numpy as np
import argparse

'''
    Splits data into train/validation/test positive/negative folders. The split is done by patients (i.e. if patient X has an image in the test set, all of X's images are found in the test set, there will be no train/test data contamination)
    To run:
        python3 file_mover.py --path_dataset "PATH_PNG_DATASET" --path_to_split "DESTINATION_SPLITS" --train_ratio [0-100] --valid_ratio [0-100] --test_ratio [0-100]
    If ratios are unspecified, the default values will be 60% train, 20% valid, 20% test.
'''

parser = argparse.ArgumentParser()
parser.add_argument("--path_dataset", type=str, default=None, help="Path to .png dataset containing folders both;fatpad;fracture;negative.")
parser.add_argument("--path_to_split", type=str, default=None, help="Path to place the splits.")
parser.add_argument("--train_ratio", type=int, default=60, help="Percentage of train patients.")
parser.add_argument("--valid_ratio", type=int, default=20, help="Percentage of valid patients.")
parser.add_argument("--test_ratio", type=int, default=20, help="Percentage of test patients.")

args = parser.parse_args()
print("List of arguments:")
print(args)

#default ratios: 60%-20%-20% train/valid/test
ratios = [args.train_ratio, args.valid_ratio, args.test_ratio]

assert sum(ratios) <= 100, "Please make sure that the ratios add up to no more than 100."

if sum(ratios) < 100:
    print("Warning! Total ratio is less than 100.")

source_folder = args.path_dataset
source_both = os.path.join(source_folder, "both")
source_fatpad = os.path.join(source_folder, "fatpad")
source_fracture = os.path.join(source_folder, "fracture")
source_negative = os.path.join(source_folder, "negative")

split_folder = args.path_to_split
train = os.path.join(split_folder, "train")
valid = os.path.join(split_folder, "valid")
test = os.path.join(split_folder, "test")

paths_pos = [os.path.join(train, "pos"), os.path.join(valid, "pos"), os.path.join(test, "pos")]
paths_neg = [os.path.join(train, "neg"), os.path.join(valid, "neg"), os.path.join(test, "neg")]

os.makedirs(paths_pos[0], exist_ok=True)
os.makedirs(paths_pos[1], exist_ok=True)
os.makedirs(paths_pos[2], exist_ok=True)
os.makedirs(paths_neg[0], exist_ok=True)
os.makedirs(paths_neg[1], exist_ok=True)
os.makedirs(paths_neg[2], exist_ok=True)

def split_patients(source, dest_paths, id_, ratios):
    '''
        Copies patients from source folder (both/fatpad/fracture/negative) to target split folders (train/valid/test) using target split ratios, and adds prefix id_ to images.
        Parameters:
            source: source path for the patients
            dest_paths: list containing paths to train/valid/test folders
            id_: string;
                "b": both fatpad and fracture positive
                "fp": only fatpad positive
                "f": only fracture positive
                "n": neither fatpad nor fracture positive (negative for both)
            ratios: list of integers representing train_ratio, valid_ratio and test_ratio, values must be between 0 and 100; sum of all 3 must add up to no more than 100.
    '''
    #grab paths for train/valid/test
    dest_train = dest_paths[0]
    dest_valid = dest_paths[1]
    dest_test = dest_paths[2]
    #create list of patient folders
    patients = os.listdir(source)
    #grab amount of patients in folder
    patient_total = len(patients)
    #calculate how many patients to place in each split
    train_target_count = patient_total * (ratios[0]/100) 
    valid_target_count = patient_total * (ratios[1]/100)
    test_target_count = patient_total * (ratios[2]/100)
    
    #sometimes splits aren't perfect ints; calculating how many patients would be lost by rounding down
    leftover = train_target_count - int(train_target_count) + valid_target_count - int(valid_target_count) + test_target_count - int(test_target_count)
    #rounding down the amount of patients to add to each split; adding leftovers to train split
    train_target_count = int(train_target_count) + int(leftover)
    valid_target_count = int(valid_target_count)
    test_target_count = int(test_target_count)
    
    #Note: Following code section could be simplified by writing a function to replace the loops.
    #copying patients to train
    for i in range(train_target_count):
        #generates a random index to pop from patients (list of patients) and copy to destination folder
        patient_to_pop = int(np.random.rand(1) * len(patients))
        patient_to_move = patients.pop(patient_to_pop) #get patient ID to move to train folder
        patient_source = os.path.join(source, patient_to_move)
        copy_images(patient_source, dest_train, patient_to_move, id_)
    #copying patients to valid
    for i in range(valid_target_count):
        patient_to_pop = int(np.random.rand(1) * len(patients))
        patient_to_move = patients.pop(patient_to_pop) #get patient ID to move to valid folder
        patient_source = os.path.join(source, patient_to_move)
        copy_images(patient_source, dest_valid, patient_to_move, id_)

    #copying patients to test
    for i in range(test_target_count):
        patient_to_pop = int(np.random.rand(1) * len(patients))
        patient_to_move = patients.pop(patient_to_pop) #get patient ID to move to test folder
        patient_source = os.path.join(source, patient_to_move)
        copy_images(patient_source, dest_test, patient_to_move, id_)
    
    return 0
    
def copy_images(patient_source, patient_dest, patient_ID, id_):
    '''
        Copies images contained in patient_source folder to id_/patient_dest folder
        Parameters:
            patient_source: path, patient folder to copy files from
            patient_dest: path, destination patient folder to copy files to
            patient_ID: ID of patient
            id_: str, new images will be prefixed by this id_ (e.g. fp2051: fatpad positive patient)
    '''
    images = os.listdir(patient_source)
    for i in images:
        shutil.copy(os.path.join(patient_source, i), os.path.join(patient_dest, id_ + patient_ID + "_" + i))
    return 0

#change the following function calls to describe how you would like the classes split
split_patients(source=source_both, dest_paths=paths_pos, id_="b", ratios=ratios)
split_patients(source=source_fatpad, dest_paths=paths_pos, id_="fp", ratios=ratios)
split_patients(source=source_fracture, dest_paths=paths_neg, id_="f", ratios=ratios)
split_patients(source=source_negative, dest_paths=paths_neg, id_="n", ratios=ratios)

print("Done!")