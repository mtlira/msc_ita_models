import os
import numpy as np
import pandas as pd

'Group individual datasets, split dataset into training and test'

def group_datasets(folder_path):
    '''
    Groups several csv datasets contained in a folder path into a single file.\n
    folder_path: folder path where all the csv's to be grouped are located
    '''
    #https://www.geeksforgeeks.org/how-to-iterate-over-files-in-directory-using-python/

    global_dataset = []

    for subdir, _, files in os.walk(folder_path):
        for file in files:
            if '.csv' in file and 'global' not in subdir:
                dataframe_dir = os.path.join(subdir, file)
                dataframe = pd.read_csv(dataframe_dir, header = None)
                #global_dataset = dataframe if len(dataframe) == 0 else pd.concat((global_dataset, dataframe), axis = 0, ignore_index = True)
                if len(global_dataset) == 0:
                    global_dataset = dataframe
                else:
                    global_dataset = pd.concat((global_dataset, dataframe), axis = 0, ignore_index = True)

    #remover
    # double = global_dataset.astype('float32')
    # print('max', (double - global_dataset).to_numpy().max())
    # print('dffff\n',(global_dataset[179] - double[179]).idxmax())
    # print(global_dataset.iloc[10075,179], double.iloc[10075,179])

    print('Global dataset shape:', np.shape(global_dataset))
    #print('column 50 min max mean', np.min(global_dataset[100]), np.max(global_dataset[100]), np.mean(global_dataset[100]))
    #print('global dataset[98].mean',global_dataset[98].mean())
    #print('global dataset[98].std',global_dataset[98].std())
    
    save_path = folder_path + 'global_dataset/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    #global_dataset.to_csv(save_path + 'global_dataset.csv', header = False, index = False)

    # Normalization
    mean_row = []
    std_row = []
    for column in global_dataset.columns:
        mean = global_dataset[column].mean()
        std = global_dataset[column].std()
        global_dataset[column] = (global_dataset[column] - mean)/std
        if global_dataset[column].isnull().values.any():
            print('mean', mean)
            print('std', std)
            print('NaN column', column)

        mean_row.append(mean)
        std_row.append(std)

        np.savetxt(save_path + 'normalization_data.csv', np.array([mean_row, std_row]), delimiter=",")

    global_dataset.to_csv(save_path + 'global_dataset.csv', header = False, index = False)

def split_dataset(save_folder, dataset_name, train_percentage):
    '''Splits the global dataset into training and test datasets'''
    
    # Read dataframe
    dataframe = pd.read_csv(save_folder + dataset_name, header = None)

    # Shuffling the dataset
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)

    # Partitioning into training and test datasets
    chop_index = int(np.rint(train_percentage*len(dataframe)))
    training_dataset = dataframe[0:chop_index]
    test_dataset = dataframe[chop_index:]
    print('percentage training dataset =',len(training_dataset)/len(dataframe))
    print('percentage test dataset =',len(test_dataset)/len(dataframe))

    # Saving
    training_dataset.to_csv(save_folder + 'train_dataset.csv', sep=',', header = False, index = False)
    test_dataset.to_csv(save_folder + 'test_dataset.csv', sep=',', header = False, index = False)
                                
group_datasets('dataset_canon/canon_N_90_M_10_hover_only/')
split_dataset('dataset_canon/canon_N_90_M_10_hover_only/global_dataset/', 'global_dataset.csv', 0.8)

# Teste de validação
#check = pd.read_csv('dataset_canon/canon_N_50_M_20/global_dataset.csv', header = None)
#print('check global dataset shape',np.shape(check))
#print('column 50 min max mean', np.min(check[100]), np.max(check[100]), np.mean(check[100]))

#check_normalization = pd.read_csv('dataset_canon/canon_N_50_M_20/global/normalization_data.csv', header = None)
#print('check_normalization shape', np.shape(check_normalization))
#print('check_normalization[98].std\n', check_normalization[98][1])