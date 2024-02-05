import os 
import tensorflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import glob
import argparse

'''
    Computes the performance of a model using various metrics (precision, sensitivity, auc etc) on the test dataset.
    To run:
        python3 compute_checkpoint_metrics.py --model_name "NAME" --path_model "PATH_MODELS" ---path_dataset "PATH_TO_TEST_DATA"
'''


def metrics(tn, fp, fn, tp): 
    '''
        Using the confusion matrix, computes various metrics. See return statement.
        Parameters:
            tn: true negatives
            fp: false positives
            fn: false negatives
            tp: true positives
    '''
    precision = tp / (tp +fp)
    #print("Precision = ", precision ,"\n")
    sensitivity = tp/(tp+fn)
    #print("Sensitivity / TPR = ", sensitivity, "\n")
    specificity = tn/(fp+tn)
    #print("Specificity = ", specificity,"\n")
    accuracy = (tp+tn)/(sum([tn, fp, fn, tp]))
    #print("Accuracy = ", accuracy ,"\n")
    f1_score =(2*tp)/(2*tp+fp+fn)
    #print("F1_score = ", f1_score ,"\n")
    fnr = fn/(fn+tp)
    #print("FNR = ", fnr ,"\n")
    return precision, sensitivity, specificity, accuracy, f1_score, fnr

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="001", help="Name of model to be loaded / tested.")
parser.add_argument("--path_model", type=str, default=None, help="Path to main folder where models are saved.")
parser.add_argument("--path_dataset", type=str, default=None, help="Path to dataset.")


args = parser.parse_args()
print("List of arguments:")
print(args)


path_to_models = args.path_model
path_test = os.path.join(args.path_dataset, 'test')

model_name = args.model_name
model_path = os.path.join(path_to_models, model_name)
model_checkpoint_path = model_path + "_checkpoint"

#Don't augment the test dataset!
#load test datagenerator & batches
test_datagen = ImageDataGenerator(rescale=1./255)
test_batches = test_datagen.flow_from_directory(path_test, 
                                                            shuffle=False,
                                                            color_mode='rgb', 
                                                            target_size=(224,224),
                                                            #batch_size=32,
                                                            class_mode='binary')

loaded_model = tensorflow.keras.models.load_model(model_checkpoint_path)
#predict classes
predictions = loaded_model.predict(x = test_batches, verbose=0)
#calculate metrics
cm = confusion_matrix(y_true = test_batches.classes, y_pred = ((predictions>0.5).astype("int32")))
tn, fp, fn, tp = confusion_matrix(y_true = test_batches.classes, y_pred = ((predictions>0.5).astype("int32"))).ravel()
precision, sensitivity, specificity, accuracy, f1_score, fnr = metrics(tn,fp, fn, tp)
fpr, tpr, thresholds = roc_curve(test_batches.classes,predictions)
auc_score = auc(fpr, tpr)

#append the required metrics to the result list
result = [model_name, str(tn), str(fp), str(fn), str(tp), str(precision), str(sensitivity), str(specificity), str(accuracy), str(f1_score), str(auc_score)]

f = open(os.path.join(path_to_models, model_name + "_metrics.csv"), "a")
#write the format
f.write("model_number,tn,fp,fn,tp,precision,sensitivity,specificity,accuracy,f1_score,auc\n")
#convert metrics to string and add to file
result_stringified = ','.join(map(str, result))
f.write(result_stringified + "\n")
f.close()
