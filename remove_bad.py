import shutil
import argparse
import os

'''
    Removes patient IDs from given dataset. Text file at --path_to_remove must contain each ID in a separate, newline.
        Optionally moves files to --path_new, if given.
        
    To run:
        python3 remove_bad.py --path_dicom "PATH_TO_DICOM" --path_to_remove "PATH_TO_TXT_IDS" --path_new "[optional] OUTPUT_PATH"
'''


parser = argparse.ArgumentParser()
parser.add_argument("--path_dicom", type=str, default=None, help="Path to dicom dataset.")
parser.add_argument("--path_to_remove", type=str, default=None, help="Path to text file containing patients to remove.")
parser.add_argument("--path_new", type=str, default=None, help="Path where to place removed patients. If unspecified, files will be simply removed.")

args = parser.parse_args()
print("List of arguments:")
print(args)

path_dicom = args.path_dicom
path_to_remove = args.path_to_remove
path_new = args.path_new

with open(path_to_remove, 'r') as f:
    patient_IDs = f.readlines()
    
#remove \n character from each ID    
for i in range(len(patient_IDs)):
    patient_IDs[i] = patient_IDs[i][:-1]
    
    
for patient_ID in patient_IDs:
    curr_patient_path = os.path.join(path_dicom, patient_ID)
    #check if patient ID is not duplicate
    if os.path.exists(curr_patient_path):
        #remove patients if path_new is not given
        if path_new is None:
            shutil.rmtree(curr_patient_path)
        #otherwise move to path_new
        else:
            os.makedirs(path_new, exist_ok=True)
            shutil.move(curr_patient_path, path_new)
            
print("Done!"