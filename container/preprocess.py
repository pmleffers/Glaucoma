#!/usr/bin/env python3
from __future__ import print_function
import datetime
import random
import shutil,sys
import json
import pickle
import os
import subprocess
from subprocess import call
import glob
import numpy as np
from imgaug import augmenters as iaa
import imageio
import imgaug as ia

class preprocess_data(object):
    
    def __init__(self,data_folder):
        self.input_data_folder=data_folder+'images/'
        self.training_directory=self.input_data_folder+'training'
        self.testing_directory=self.input_data_folder+'testing'
        self.validation_directory=self.input_data_folder+'validation'
        self.augmentation_directory=self.input_data_folder+'augmentation'
        
    def dl_url(self,dl_location,url_link):#dl_location: folder to be created where files go
        '''
        This function downloads from a url and extracts the data into a folder 
            and returns a list of the extracted files, removes the zip file, 
            then returns you to your previous directory.
        '''
        input_data_folder=self.input_data_folder
        tmp_dir=dl_location #set tmp dir
        try:
            os.mkdir(tmp_dir) #create tmp dir
            subprocess.call('wget -P '+tmp_dir+' '+url_link,shell=True) #dl url_link
        except:
            subprocess.call('wget -P '+tmp_dir+' '+url_link,shell=True) #dl url_link
        os.chdir(tmp_dir) #move into tmp directory
        dl_files=(glob.glob("*.*")) #get zip file list
        for i in dl_files: 
            subprocess.call('unzip '+i,shell=True)# unzip files
            subprocess.call('rm -r '+i,shell=True)
            
        extracted_list=[os.getcwd()+'/'+i for i in list(np.array(os.listdir())[np.array(os.listdir())!=glob.glob("*.*")[0]])]    
        os.chdir('..')   
        
        #Setup Directories
        training_directory=self.training_directory
        testing_directory=self.testing_directory
        validation_directory=self.validation_directory
        augmentation_directory=self.augmentation_directory
        cases_directory=input_data_folder+'cases'
        controls_directory=input_data_folder+'controls'
        print(training_directory)
        print(testing_directory)
        print(validation_directory)
        print(augmentation_directory)
        print(cases_directory)
        print(controls_directory)

        DirList=[input_data_folder,training_directory,testing_directory,validation_directory,augmentation_directory,cases_directory,controls_directory]
        for i in DirList:
            try:
                # Create target Directory
                os.mkdir(i)
                print("Directory " ,i,  " Created ") 
            except:
                print("Directory " ,i,  " already exists")

        #Move case pictures together in same folder
        subprocess.call('mv '+dl_location+'advanced_glaucoma/* '+dl_location+'early_glaucoma',shell=True)
        subprocess.call('mv '+dl_location+'early_glaucoma/* '+input_data_folder+'cases',shell=True)
        subprocess.call('mv '+dl_location+'normal_control/* '+input_data_folder+'controls',shell=True)
        subprocess.call('rm -r '+dl_location+'early_glaucoma/',shell=True)
        subprocess.call('rm -r '+dl_location+'normal_control/',shell=True)
        subprocess.call('rm -r '+dl_location+'advanced_glaucoma/',shell=True)
        return extracted_list

    def create_train_test_val(input_data_folder):
        training_folder = os.path.sep.join([input_data_folder, "training"])
        validation_folder = os.path.sep.join([input_data_folder, "validation"])
        testing_folder  = os.path.sep.join([input_data_folder, "testing"])
        return training_folder,validation_folder,testing_folder

    def list_images(input_data_folder, contains=None):
        image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        return preprocess_data.list_files(input_data_folder, \
               valid_extensions=image_types, contains=contains)

    def list_files(input_data_folder, valid_extensions=None, contains=None):
        for (root_directory, directory_names, filenames) in os.walk(input_data_folder):
            for filename in filenames:
                if contains is not None and filename.find(contains) == -1:
                    continue

                extension = filename[filename.rfind("."):].lower()
                if valid_extensions is None or extension.endswith(valid_extensions):
                    image_paths = os.path.join(root_directory, filename)
                    yield image_paths

    def images_augment(self,augment_percent,training_paths):
        aug_num=int(len(training_paths) * augment_percent)
        for image_iter in training_paths[:aug_num]:
            filename,extension = (os.path.splitext(os.path.basename(image_iter)))
            path,fullname=os.path.split(image_iter) 
            label=(os.path.basename(os.path.normpath(path)))
            path=self.augmentation_directory
            imread_image = imageio.imread(image_iter)
            random_num=random.randint(0,360)
            rotate = iaa.Affine(rotate=(-random_num, random_num))#<---
            image_aug = rotate.augment_image(imread_image)
            imageio.imwrite(path+'/'+label+'_aug_'+str(random_num)+'.png',image_aug)
        return aug_num

    def setup_folders(self,training_percent=0.8,validation_percent=0.1,augment_percent=0.2,save_location='/opt/program/'):
        input_data_folder=self.input_data_folder
        training_folder,validation_folder,testing_folder=preprocess_data.create_train_test_val(self.input_data_folder)
        image_paths = list(preprocess_data.list_images(input_data_folder))
        random.shuffle(image_paths)
        comparison = int(len(image_paths) * training_percent)
        training_paths = image_paths[:comparison]
        testing_paths = image_paths[comparison:]
        comparison = int(len(training_paths) * validation_percent)
        validation_paths = training_paths[:comparison]
        training_paths = training_paths[comparison:]  
        self.training_paths=training_paths
        
        print('Images selected for training folder: ',len(training_paths))
        print('Images selected for testing folder: ',len(testing_paths))
        print('Images selected for validation folder: ',len(validation_paths))
        datasets = [("training", training_paths, training_folder),
                    ("validation", validation_paths, validation_folder),
                    ("testing", testing_paths, testing_folder)]

        for (data_type, image_paths, output_folder) in datasets:
            print("\nbuilding "+data_type+" collection . . .\n")
            if not os.path.exists(output_folder):
                print("creating "+data_type+" directory . . .")
                os.makedirs(output_folder)

            for path in image_paths:
                filename = path.split(os.path.sep)[-1]
                label = path.split(os.path.sep)[-2]
                label_paths = os.path.sep.join([output_folder, label])
                if not os.path.exists(label_paths):
                    print("creating "+data_type+" directory . . .")
                    os.makedirs(label_paths)
                p = os.path.sep.join([label_paths, filename])
                shutil.copy2(path, p)  

        aug_num=preprocess_data.images_augment(self,augment_percent,training_paths)#<---
        training_total = len(list(preprocess_data.list_images(training_folder)))
        validation_total = len(list(preprocess_data.list_images(validation_folder)))
        testing_total = len(list(preprocess_data.list_images(testing_folder)))
        total_images=training_total+validation_total+testing_total
        print('\nTraining Images: ',training_total,'\nValidation Images: ',
              validation_total,'\nTesting Images: ',testing_total,
              '\nTotal Images Selected: ',total_images,'\nAugmented Images Added: ',aug_num)
        #print('\nTraining Images: ',training_total,'\nValidation Images: ',
        #      validation_total,'\nTesting Images: ',testing_total,
        #      '\nTotal Images Selected: ',total_images)

        train_images = len(list(preprocess_data.list_images(train_process.training_directory)))
        validation_images = len(list(preprocess_data.list_images(train_process.validation_directory)))
        test_images = len(list(preprocess_data.list_images(train_process.testing_directory)))

        save_records={}
        save_records['train_len']=train_images
        save_records['test_len']=test_images
        save_records['val_len']=validation_images
        save_records['train_dir']=self.training_directory
        save_records['test_dir']=self.testing_directory
        save_records['val_dir']=self.validation_directory
        print(save_records)

        with open(save_location+'save_records.pickle', 'wb') as records:
            pickle.dump(save_records, records, protocol=pickle.HIGHEST_PROTOCOL)
        with open(save_location+'image_records.json', 'w') as fp:
            json.dump(save_records, fp)
            
        subprocess.call('rm -r '+self.input_data_folder+'cases/',shell=True)
        subprocess.call('rm -r '+self.input_data_folder+'controls/',shell=True)
        print('The image files are saved to:',train_process.input_data_folder)
        return training_paths

#import argparse
#ap = argparse.ArgumentParser()
#ap.add_argument("-u", "--url", required = True, help = "url path to the data", default='https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/1YRRAC/OGRSQO')
#ap.add_argument("-cp", "--cpath", required = True, help = "path in container to save data", default='tmp')
#ap.add_argument("-sp", "--spath", required = True, help = "path in container to save pickle", default='tmp')

#args = vars(ap.parse_args())

start = datetime.datetime.now()
print(datetime.datetime.now())

#dl_location=args('path')
#url_link=args('url')

url_link='https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/1YRRAC/OGRSQO'
dl_location='tmp/'
save_location='/opt/program/'

train_process=preprocess_data(dl_location)
file_list=train_process.dl_url(dl_location,url_link)
image_directory=train_process.setup_folders()

end = datetime.datetime.now()
print(datetime.datetime.now())
delta = end-start
print('\nMinutes to preprocess data: ',str(delta))



