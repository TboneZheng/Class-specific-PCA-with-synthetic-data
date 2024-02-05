
# Usuals
import os
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import itertools
from numpy import arange
import random

# For ALSS
from scipy import sparse
from scipy.sparse.linalg import spsolve

# Scaling
from sklearn.preprocessing import StandardScaler, normalize

# PCA
from sklearn import decomposition

# Models
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

# Model evaluation
from sklearn.metrics import accuracy_score

# Crossvalidation
from sklearn.model_selection import LeaveOneGroupOut

"""
This function takes all the raw CSV data from Edward, 
trim the redundant wavenumbers as adviced 
and combine into one single dataset 
(optional to write to a new csv file, 
known to have extra index column, 
need to process if used in future)
"""
def combine_all_urine_csv(write_to_csv = False, output_path='Urine - Cleaned-Up - v2.csv'):

    csv_files = glob.glob('./Urine_CSV/*.csv')

    print('finished globbing')

    df_raw = pd.DataFrame()
    pnums = []
    labels = []
    for filename in csv_files:
        # Remove redundant wavenumbers
        data_point = pd.read_csv(filename)
        data_point = data_point.T
        data_point = data_point.set_index(0)
        new_header = data_point.iloc[0]
        data_point = data_point[1:]
        data_point.columns = new_header
        if len(data_point.columns) > 10_000:
            data_point = data_point[data_point.columns[::10]]
        df_raw = pd.concat([df_raw, data_point],axis=0)

        # Get patient numbers
        if ' ' in filename:
            pnum = filename[14]+filename[16]+filename[17]
        else:
            pnum = filename[12]+filename[14]+filename[15]
        print(pnum)
        pnums.append(pnum)

        # Adjust labels
        label_rect = {1:2, 2:3, 3:1, 4:0}
        labels.append(label_rect[int(filename[12])])



    print('finished cleaning up')

    df_raw.insert(0,'Patient', pnums)
    df_raw.insert(0,'Label', labels)

    print('Number of patient:',len(set(pnums)))

    if write_to_csv:
        df_raw.to_csv(output_path) 
        print('Wrote to file:', output_path)
    
    return df_raw


def combine_all_filtered_csv(write_to_csv=False, output_path='Filtered_blood Cleaned-up.csv'):
    csv_files = glob.glob('./Filtered_Blood_CSV/*.csv')

    df_raw = pd.DataFrame()
    pnums = []
    labels = []
    for filename in csv_files:
        data_point = pd.read_csv(filename)
        data_point = data_point.drop(data_point.columns[2],axis=1)
        data_point = data_point.T
        data_point = data_point.set_index(0)
        new_header = data_point.iloc[0]
        data_point = data_point[1:]
        # data_point.columns = new_header

        df_raw = pd.concat([df_raw, data_point],axis=0)

        # Get patient numbers
        pnum = filename[25]+filename[26]+filename[27]
        print(pnum)
        pnums.append(pnum)

        # Adjust labels
        label_rect = {'H':0, 'P':1, 'E':2, 'C':3}
        labels.append(label_rect[int(filename[21])])



    print('finished cleaning up')

    df_raw.insert(0,'Patient', pnums)
    df_raw.insert(0,'Label', labels)

    print('Number of patient:',len(set(pnums)))

    if write_to_csv:
        df_raw.to_csv(output_path) 
        print('Wrote to file:', output_path)
    
    return df_raw


def combine_all_saliva_csv(write_to_csv=False, train=False):

    input_path = './Datasets/Saliva_CSV/SalivaTrain/*.csv' if train else './Datasets/Saliva_CSV/SalivaTest/*.csv'
    output_path = './Datasets/Saliva_train.csv' if train else './Datasets/Saliva_test.csv'

    csv_files = glob.glob(input_path)

    df_raw = pd.DataFrame()
    pnums = []
    labels = []
    for file_path in csv_files:
        
        # Remove redundant wavenumbers
        data_point = pd.read_csv(file_path)

        data_point = data_point.T
        new_header = data_point.iloc[0]
        data_point = data_point[1:]
        data_point.columns = new_header
    
        df_raw = pd.concat([df_raw, data_point],axis=0)
        
        file_name = os.path.basename(file_path).split('_')
        # Get patient numbers
        pnum = file_name[1]
        print(pnum)
        pnums.append(pnum)

        # Adjust labels
        label_rect = {'H':0, 'C':1}
        labels.append(label_rect[file_name[0]])



    print('finished cleaning up')

    df_raw.insert(0,'Patient', pnums)
    df_raw.insert(0,'Label', labels)

    print('Number of patient:',len(set(pnums)))

    if write_to_csv:
        df_raw.to_csv(output_path) 
        print('Wrote to file:', output_path)
    
    return df_raw


#        Asymmetric Least Square algorithm.
#        y       : signal
#        lam     : smoothness, often takes values between 10**2 and 10**9
#        p       : asymmetry value, often takes values between 0.001 and 0.1
#        niter   : number of iterations 
#        z       : simulated (smoothed) signal
#        L       : length of y 
#        D       : difference matrix
#        w       : weights
#        W       : diag(w)
            
def baseline_als(y, lam, p, niter=10):
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z


def generate_alss(df, lam, p, is_synth):
    if not is_synth:
        print('Generating ALSS for real data')
        df = (100-df) # Chemistry
    df_arr = df.to_numpy()
    num_rows = df_arr.shape[0]
    num_cols = df_arr.shape[1]
    new_dt_arr = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        data_point = df_arr[i] - baseline_als(df_arr[i], lam, p)
        data_point = data_point / np.average(data_point) 
        new_dt_arr[i,:] = data_point*1
    new_df = pd.DataFrame(new_dt_arr)
    return new_df


# Data augmentation
def generate_augmentation(df, shift=1, add_noise=1, lin_comb=0):
    # Get all three samples from each patient (group)
    groups_generator = df.groupby('Patient')
    groups = [groups_generator.get_group(i) for i in groups_generator.groups]

    result_df = pd.DataFrame()

    # Iterate through groups and augment as specified
    for group in groups:
        patients = group['Patient']
        labels = group['Label']
        spectra = group.iloc[:,2:]

        print('Augmenting data samples from patient',patients.iloc[0])

        if shift:
            for i in range(3):
                patient = patients.iloc[i]
                label = labels.iloc[i]
                spectrum = pd.DataFrame(spectra.iloc[i]).T

                spec_len = len(spectrum.columns)
                if np.random.random() > 0.5:
                    for i in range(spec_len-1):  # loop till last but one column
                        spectrum.iloc[:,i] = spectrum.iloc[:,i+1]   # shifts left by one wavenumber
                    # spectrum.iloc[:,spec_len] = spectrum.iloc[:,spec_len-1]   # not needed as last col stay the same?
                else:
                    for i in range(spec_len-1):  # loop till last but one column
                        spectrum.iloc[:,i] = spectrum.iloc[:,i+1]   # each column gets values of previous column

                spectrum.insert(0,'Patient',patient)
                spectrum.insert(1,'Label',label)
                group = pd.concat([group, spectrum],axis=0).reset_index(drop=True)
        
        # reseting group stuff (6 each group now instead of 3 after shifting)
        patients = group['Patient']
        labels = group['Label']
        spectra = group.iloc[:,2:]

        if add_noise:
            for i in range(6):
                patient = patients.iloc[i]
                label = labels.iloc[i]
                spectrum = pd.DataFrame(spectra.iloc[i]).T            
                
                noise = np.random.normal(0,1.0,3301)
                spectrum = spectrum + noise
                
                spectrum.insert(0,'Patient',patient)
                spectrum.insert(1,'Label',label)
                group = pd.concat([group, spectrum],axis=0).reset_index(drop=True)
        
        result_df = pd.concat([result_df, group], axis=0)

        if lin_comb:
            pass # essentially just jumbo, ignored for now
        
    result_df = result_df.sort_values('Patient')
    result_df = result_df.reset_index(drop=True)
    result_pnums = result_df['Patient']
    result_labels = result_df['Label']
    result_spectra = result_df.drop(columns=['Label', 'Patient'])

    return result_spectra, result_labels, result_pnums


# Get scenario specific data frames, groups and labels
def get_scenario_specs(scenario, _df):

    print('Scenario:',scenario)
    print('Original Shape:',_df.shape)

    df = _df.reset_index(drop=True)
    if scenario=='pvc':
        df = df[df['Label'] != 0]
    elif scenario=='hvc':
        df = df[df['Label'] != 1]
    elif scenario=='eva':
        df = df[df['Label'] > 1]
    df = df.reset_index(drop=True)
    
    groups = df['Patient']
    labels = pd.DataFrame(df['Label'])

    match scenario: 
        case 'pvc':
            labels.loc[labels['Label'] == 1,'Label'] = 0
            labels.loc[labels['Label'] > 1, 'Label'] = 1
        case 'hvc':
            # labels.loc[labels['Label'] == 0,'Label'] = 0
            labels.loc[labels['Label'] > 0, 'Label'] = 1
        case 'avc':
            labels.loc[labels['Label'] == 1,'Label'] = 0
            labels.loc[labels['Label'] > 1, 'Label'] = 1
        case 'eva':
            labels.loc[labels['Label'] == 2,'Label'] = 0
            labels.loc[labels['Label'] > 2, 'Label'] = 1
        case 'mul':
            pass

    df = df.drop(columns=['Patient', 'Label'])

    print('Scenario specific shape:',df.shape)

    return df, groups, labels


# Scaling
def scale(X_train, X_test, X_val=pd.DataFrame()):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    if not X_val.empty:
        X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    if not X_val.empty:
        return pd.DataFrame(X_train_scaled), pd.DataFrame(X_val_scaled), pd.DataFrame(X_test_scaled)
    else:
        return pd.DataFrame(X_train_scaled), pd.DataFrame(X_test_scaled)


# PC
def pca(X_train, X_test, num_components, verbose=1, plot=0):
    pca = decomposition.PCA(n_components = num_components, random_state=1)
    pca.fit(X_train)

    X_train_redu = pd.DataFrame(data = pca.transform(X_train))
    X_test_redu = pd.DataFrame(data = pca.transform(X_test))

    if verbose:
        print('Percentage of variance explained: ', sum(pca.explained_variance_ratio_[:num_components]))

    if plot:
        plot_x = [n+1 for n in range(num_components)]
        plot_y = [sum(pca.explained_variance_ratio_[:s+1]) for s in range(num_components)]
        plt.scatter(plot_x, plot_y)
        plt.ylabel('percentage of explained variance')
        plt.xlabel('principle components')
        plt.show
    
    return X_train_redu, X_test_redu


def pca_show(df, num_components, verbose=1, plot=1):
    pca = decomposition.PCA(n_components = num_components, random_state=1)
    pca.fit(df)

    if verbose:
        print('Percentage of variance explained: ', sum(pca.explained_variance_ratio_[:num_components]))

    if plot:
        plot_x = [n+1 for n in range(num_components)]
        plot_y = [sum(pca.explained_variance_ratio_[:s+1]) for s in range(num_components)]
        plt.scatter(plot_x, plot_y)
        plt.ylabel('percentage of explained variance')
        plt.xlabel('principle components')
        plt.show

def pca_determine_num(df):
    for i in range(10,100):
        pca = decomposition.PCA(n_components = i, random_state=1)
        pca.fit(df)
        if sum(pca.explained_variance_ratio_[:i]) >= 0.995:
            return i
    return 100


# LOOCV
def run_loocv(model, df, labels, split, verbose, hp=[],zero_center=1,do_pca=1):

    print('Running LOOCV')

    if hp == []:
        print('Running test with default model hyperparameters')
        # Best yet:
        if model=='svm':
            _C , _gamma, _kernel = 10, 1, 'linear'
        elif model=='lda':
            _solver, _shrinkage = 'svd', 0.66
    else:
        if model=='svm':
            _C, _gamma, _kernel = hp[0], hp[1], hp[2]
            print('Running test with hyperparameters C:',_C,'gamma:',_gamma,'kernel:',_kernel)
        elif model=='lda':
            _solver, _shrinkage = hp[0], hp[1]

    if do_pca:
        num_components = pca_determine_num(df)
    
    n_classes = len(labels.unique())
    acc_history = []
    fold_count = 1

    # nah.. you need to shuffle the training set
    # for train, test in split:
    #     splits.append((train,test))

    # random.shuffle(splits)

    for train, test in split:
        X_train, X_test, y_train, y_test = df.iloc[train, :], df.iloc[test, :], labels[train], labels[test]

        # shuffle the training data

        X_train = X_train.sample(frac=1,random_state=69)
        y_train = y_train.sample(frac=1,random_state=69)

        print('Completed split and shuffle')

        # X_train, X_test = zero_center(X_train, X_test)


        print(y_test)

        
        if do_pca:
            X_train, X_test = pca(X_train, X_test, num_components, verbose=0, plot=0)
        else:
            num_components=df.shape[1]
            print('Input dimension is:', num_components)

        
        # X_test = X_test[:3]
        # y_test = y_test[:3]

        print('Isolated test set')

        if verbose:
            print('Testing set ground truth labels: ', y_test)
            print('Finished dimensionality reduction, Number of principal components is:',num_components)
        if model=='lda':
            classifier = LDA(solver=_solver) # shrinkage does not apply if using svd solver
        elif model=='rf':
            classifier = RandomForestClassifier()
        elif model=='svm':
            classifier = svm.SVC(C=_C, gamma=_gamma,kernel=_kernel)
        
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        score = accuracy_score(y_test, y_pred) 
        acc_history.append(score)
        if verbose:
            print('Fold:', fold_count, '  Accuracy:', score)

        fold_count+=1

    print('Mean accuracy is ', np.mean(acc_history))
    return np.mean(acc_history)


# Run tests
def test_model(model_name, df, labels, _groups, verbose=0, zero_center=1, do_pca=1):
    split = generate_split(df, labels, _groups)
    result = run_loocv(model_name, df, labels['Label'], split, verbose, hp=[], zero_center=zero_center, do_pca=do_pca)
    return result


def generate_split(df, labels, _groups):
    labels_l = labels['Label']
    logo = LeaveOneGroupOut()
    split = logo.split(df, labels_l, groups=_groups)
    return split


def get_best_hyper_params(model_name, df, labels, _groups, verbose=0):
    
    search_space = get_search_space(model_name)
    print('Total number of hyperparameter sets is:', len(search_space))
    np.random.shuffle(search_space)

    max_score = 0
    for i,hp_combo in enumerate(search_space):
        print('----- Testing hp set',i,'-----')
        if i>=201:
            break # well...
        best_hp = []
        split_inner = generate_split(df, labels, _groups=_groups)
        acc_history = run_loocv(model_name, df, labels['Label'], split_inner, verbose=verbose,hp=hp_combo)
        score = np.mean(acc_history)
        if score >= max_score:
            max_score = score
            print(hp_combo)
            best_hp = hp_combo
    
    if model_name=='svm':
        print('Best hyperparateres are', best_hp, 'with score of:',max_score)
    elif model_name=='lda':
        print('Best configuration is :' ,best_hp) 
    return best_hp, max_score


# Hyperparameter search spaces for different models
def get_search_space(model_name):
    match model_name:
        case 'svm':
            return hp_svm()
        case 'lda':
            return hp_lda()


def hp_lda():
    solver = ['lsqr', 'eigen']
    shrinkage = arange(0, 1, 0.01)
    param_space = list(itertools.product(solver, shrinkage))
    return param_space


def hp_svm():
    C = [0.1, 1, 10]
    gamma = [1, 0.1, 0.01, 0.001, 0.0001]
    kernel = ['linear','rbf']
    param_space = list(itertools.product(C, gamma, kernel))
    return param_space # [C, gamma, kernel]


def split_balanced(data, target, test_size=0.2):

    classes = np.unique(target)
    # can give test_size as fraction of input data size of number of samples
    if test_size<1:
        n_test = np.round(len(target)*test_size)
    else:
        n_test = test_size
    n_train = max(0,len(target)-n_test)
    n_train_per_class = max(1,int(np.floor(n_train/len(classes))))
    n_test_per_class = max(1,int(np.floor(n_test/len(classes))))

    ixs = []
    for cl in classes:
        if (n_train_per_class+n_test_per_class) > np.sum(target==cl):
            # if data has too few samples for this class, do upsampling
            # split the data to training and testing before sampling so data points won't be
            #  shared among training and test data
            splitix = int(np.ceil(n_train_per_class/(n_train_per_class+n_test_per_class)*np.sum(target==cl)))
            ixs.append(np.r_[np.random.choice(np.nonzero(target==cl)[0][:splitix], n_train_per_class),
                np.random.choice(np.nonzero(target==cl)[0][splitix:], n_test_per_class)])
        else:
            ixs.append(np.random.choice(np.nonzero(target==cl)[0], n_train_per_class+n_test_per_class,
                replace=False))

    # take same num of samples from all classes
    ix_train = np.concatenate([x[:n_train_per_class] for x in ixs])
    ix_test = np.concatenate([x[n_train_per_class:(n_train_per_class+n_test_per_class)] for x in ixs])

    X_train = data[ix_train,:]
    X_test = data[ix_test,:]
    y_train = target[ix_train]
    y_test = target[ix_test]

    return X_train, X_test, y_train, y_test

def import_group_spectra(data_path):

    csv_files = glob.glob(data_path+'*.csv')

    print('Finished Globbing', len(csv_files), 'files')

    df_all = pd.DataFrame()
    data_point = pd.DataFrame()

    df_nylon = pd.DataFrame()
    df_cotton = pd.DataFrame()
    df_wool = pd.DataFrame()
    df_silicone = pd.DataFrame()
    df_vinyl_alcohol = pd.DataFrame()

    df_algae = pd.DataFrame()
    df_broodcomb = pd.DataFrame()
    df_chitin = pd.DataFrame()
    df_fur = pd.DataFrame()
    df_polyester = pd.DataFrame()

    df_grass = pd.DataFrame()
    df_coal = pd.DataFrame()
    df_silk = pd.DataFrame()
    df_viscose = pd.DataFrame()
    df_amber = pd.DataFrame()

    df_aramid = pd.DataFrame()
    df_fibre_down = pd.DataFrame()
    df_honeycomb = pd.DataFrame()
    df_flax = pd.DataFrame()
    df_cigarette_filter = pd.DataFrame()

    for filename in csv_files:

        print('Processing', filename)

        colnames=['number', 'value'] 
        data_point = pd.read_csv(filename, names=colnames, header=None)
        data_point = data_point.drop(data_point.columns[0],axis=1)
        data_point = data_point.T
        data_point = data_point.T.reset_index(drop=True).T
        data_point = data_point.iloc[:,:1863]

        data_point = pd.DataFrame(normalize(data_point, axis=1))

        df_all = pd.concat([df_all, data_point],axis=0)

        if filename.find('nylon') != -1:
            df_nylon = pd.concat([df_nylon, data_point], axis=0)
            print(filename.split(),' added to df_nylon')
        
        if filename.find('cotton') != -1:
            df_cotton = pd.concat([df_cotton, data_point], axis=0)
            print(filename.split(),' added to df_cotton')

        if filename.find('wool') != -1:
            df_wool = pd.concat([df_wool, data_point], axis=0)
            print(filename.split(),' added to df_wool')

        if filename.find('silicone') != -1:
            df_silicone = pd.concat([df_silicone, data_point], axis=0)
            print(filename.split(),' added to df_silicone')

        if filename.find('vinyl_alcohol') != -1:
            df_vinyl_alcohol = pd.concat([df_vinyl_alcohol, data_point], axis=0)
            print(filename.split(),' added to df_vinyl_alcohol')

        if filename.find('algae') != -1:
            df_algae = pd.concat([df_algae, data_point], axis=0)
            print(filename.split(),' added to df_algae')

        if filename.find('broodcomb') != -1:
            df_broodcomb = pd.concat([df_broodcomb, data_point], axis=0)
            print(filename.split(),' added to df_broodcomb')

        if filename.find('chitin') != -1:
            df_chitin = pd.concat([df_chitin, data_point], axis=0)
            print(filename.split(),' added to df_chitin')

        if filename.find('fur') != -1:
            df_fur = pd.concat([df_fur, data_point], axis=0)
            print(filename.split(),' added to df_fur')

        if filename.find('polyester') != -1:
            if filename.find('polyesterurethane') == -1:
                df_polyester = pd.concat([df_polyester, data_point], axis=0)
                print(filename.split(),' added to df_polyester')
        
        if filename.find('grass') != -1:
            df_grass = pd.concat([df_grass, data_point], axis=0)
            print(filename.split(),' added to df_grass')
        
        if filename.find('coal') != -1:
            df_coal = pd.concat([df_coal, data_point], axis=0)
            print(filename.split(),' added to df_coal')

        if filename.find('silk') != -1:
            df_silk = pd.concat([df_silk, data_point], axis=0)
            print(filename.split(),' added to df_silk')

        if filename.find('viscose') != -1:
            df_viscose = pd.concat([df_viscose, data_point], axis=0)
            print(filename.split(),' added to df_viscose')

        if filename.find('amber') != -1:
            df_amber = pd.concat([df_amber, data_point], axis=0)
            print(filename.split(),' added to df_amber')

        if filename.find('aramid') != -1:
            df_aramid = pd.concat([df_aramid, data_point], axis=0)
            print(filename.split(),' added to df_aramid')

        if filename.find('down') != -1:
            df_fibre_down = pd.concat([df_fibre_down, data_point], axis=0)
            print(filename.split(),' added to df_fibre_down')

        if filename.find('honeycomb') != -1:
            df_honeycomb = pd.concat([df_honeycomb, data_point], axis=0)
            print(filename.split(),' added to df_honeycomb')

        if filename.find('flax') != -1:
            df_flax = pd.concat([df_flax, data_point], axis=0)
            print(filename.split(),' added to df_flax')
            
        if filename.find('cigarette_filter') != -1:
            df_cigarette_filter = pd.concat([df_cigarette_filter, data_point], axis=0)
            print(filename.split(),' added to df_cigarette_filter')

    # Takes the mean of each group
    #TODO: assign weight to each individual sample instead of taking an average.

    df_avg_nylon = df_nylon.mean(axis=0)
    df_avg_cotton = df_cotton.mean(axis=0)
    df_avg_wool = df_wool.mean(axis=0)
    df_avg_silicone = df_silicone.mean(axis=0)
    df_avg_vinyl_alcohol = df_vinyl_alcohol.mean(axis=0)

    df_avg_algae = df_algae.mean(axis=0)
    df_avg_honeycomb = df_honeycomb.mean(axis=0)
    df_avg_chitin = df_chitin.mean(axis=0)
    df_avg_fur = df_fur.mean(axis=0)
    df_avg_polyester = df_polyester.mean(axis=0)

    df_avg_grass = df_grass.mean(axis=0)
    df_avg_coal = df_coal.mean(axis=0)
    df_avg_silk = df_silk.mean(axis=0)
    df_avg_viscose = df_viscose.mean(axis=0)
    df_avg_amber = df_amber.mean(axis=0)

    df_avg_aramid = df_aramid.mean(axis=0)
    df_avg_down = df_fibre_down.mean(axis=0)
    
    df_avg_broodcomb = df_broodcomb.mean(axis=0)
    df_avg_flax = df_flax.mean(axis=0)
    df_avg_cig_filter = df_cigarette_filter.mean(axis=0)

    all_groups = [df_avg_nylon, df_avg_cotton, df_avg_wool, df_avg_silicone, df_avg_vinyl_alcohol,
                df_avg_algae, df_avg_honeycomb, df_avg_chitin, df_avg_fur, df_avg_polyester, df_avg_grass,
                    df_avg_coal, df_avg_silk, df_avg_viscose, df_avg_amber, df_avg_aramid, df_avg_down,
                     df_avg_broodcomb, df_avg_flax, df_avg_cig_filter]

    all_group_names = ['nylon', 'cotton', 'wool', 'silicone', 'vinyl_alcohol',
                    'algae', 'honeycomb', 'chitin', 'fur', 'polyester', 'grass',
                    'coal', 'silk', 'viscose', 'amber', 'aramid', 'down',
                        'broodcomb', 'flax', 'cigarette_filter']

    return all_groups, all_group_names