import pandas as pd
import cv2
import pydicom
import os
import numpy as np
import argparse

'''
    This script converts .dcm images to .png and does a min-max normalization to a range of [0,255].
        Dicom dataset folder structure:
            >path_dicom_folder
                >patient_
                    >im_ID.dcm
                    >[other images]
                >[other patients]
                    >[other images]

    This script outputs the converted, rescaled images to folders "both", "fatpad", "fracture", "negative".
        "both": patient has fatpad AND fracture
        "fatpad": patient has a fatpad
        "fracture": patient has a fracture
        "negative": patient has neither a fatpad nor a fracture

    The images will be stored under a folder with the patient's ID.
    
    To run:
        python3 dicom_converter.py --path_dicom "PATH_TO_DICOM" --path_excel "PATH_TO_EXCEL" --path_output "OUTPUT_FOLDER"
'''

parser = argparse.ArgumentParser()
parser.add_argument("--path_dicom", type=str, default=None, help="Path to dicom dataset.")
parser.add_argument("--path_excel", type=str, default=None, help="Path to dataset patient information. Expected an excel file.")
parser.add_argument("--path_output", type=str, default=None, help="Path to output converted PNG files.")

args = parser.parse_args()
print("List of arguments:")
print(args)

path_dicom_folder = args.path_dicom
excel_file = args.path_excel
output_folder = args.path_output

os.makedirs(output_folder, exist_ok=True)

data = pd.read_excel(excel_file)

col = ["Patient ID", "fat_pad/effusion", "Fracture"]

folders = os.listdir(path_dicom_folder)

#this part removes the IDs that are missing from processing
#this was necessary, since the original dataset was missing a few IDs
missing_IDs = [] #will contain the IDs that are missing
missing_rows = [] #will contain the rows that are missing

for x in range(len(data[col[0]])):
	if not(str(data[col[0]][x]) in folders):
		missing_IDs.append(data[col[0]][x])
		missing_rows.append(x)

print(missing_IDs)
print(missing_rows)

new_data = data.drop(index=missing_rows)
#hashtable issues, workaround: convert dataframe to a new file and then reload it
new_data.to_csv("temp.csv")
data = pd.read_csv("temp.csv")

entry_count = len(data[col[0]])

#results[0] is the data category
#results[1] will keep track of how many images have been converted in that category; used for naming images
results = [ ["both/", "fatpad/", "fracture/", "negative/"],
			[0,0,0,0] ]

#create files for each data category
for r in results[0]:
	os.makedirs(os.path.join(output_folder, r), exist_ok=True)			
#categories and counts

for row in range(entry_count):
	print("Currently on iteration " + str(row+1) + "/" + str(entry_count))
	curr_path = os.path.join(path_dicom_folder,str(data[col[0]][row]))
	curr_files = os.listdir(curr_path)
	
	
	#curr_result = category in results
	if (data[col[1]][row] == 1) and (data[col[2]][row] == 1):	
		curr_result = 0
	elif (data[col[1]][row] == 1):
		curr_result = 1
	elif (data[col[2]][row] == 1):
		curr_result = 2
	else:
		curr_result = 3
		
	output_patient_path = os.path.join(output_folder, results[0][curr_result], str(data[col[0]][row]))
	os.makedirs(output_patient_path, exist_ok=True) 		
	
	for f in curr_files:
		dicom_file = os.path.join(curr_path,f)
		output = os.path.join(output_patient_path,str(results[1][curr_result]) + ".png")
		results[1][curr_result] += 1 #update count for future file
		img_dicom = pydicom.read_file(dicom_file)
		img = img_dicom.pixel_array
		img_2d = img.astype(float)
        #rescales images to a range [0,255]
		img_2d_scaled = (np.maximum(img_2d,0) / img_2d.max()) * 255.0
		img_2d_scaled = np.uint8(img_2d_scaled)
		cv2.imwrite(output,img_2d_scaled)

print("Done!")