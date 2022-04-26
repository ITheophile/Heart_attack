import pandas as pd
import numpy as np
import time
import joblib

# Training Test
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, plot_confusion_matrix, classification_report

# Algorithms
from sklearn.ensemble      import RandomForestClassifier
from sklearn.ensemble      import ExtraTreesClassifier
from sklearn.ensemble      import AdaBoostClassifier
from sklearn.ensemble      import GradientBoostingClassifier
from sklearn.ensemble      import HistGradientBoostingClassifier
from xgboost               import XGBClassifier
from lightgbm              import LGBMClassifier
from catboost              import CatBoostClassifier

# Load and Prepro 

# Load data
def load_and_cleaning(filepath):
    raw_data = pd.read_csv(filepath)
    clean_data = raw_data.drop_duplicates()
    return clean_data

# print('Shape of data after dropping duplicate',data.shape)

# Train test sets

def splitting_data(data):
    X = data.drop(columns='output')
    y = data['output']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    # print('Xtrain shape', X_train.shape)
    # print('ytrain shape', y_train.shape)
    # print('Xtest shape', X_test.shape)
    # print('ytest shape', y_test.shape)
    return X_train, X_test, y_train, y_test



# Preprocessor

def data_preprocessing(numerical_variables):
    # num_var = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
    


    num_prep = ColumnTransformer([('num_prepo', StandardScaler(), numerical_variables)],
                                remainder='passthrough')


    tree_classifiers = {
    "Extra Trees": ExtraTreesClassifier(random_state=0),
    "Random Forest":RandomForestClassifier(random_state=0),
    "AdaBoost": AdaBoostClassifier(random_state=0),
    "Skl GBM": GradientBoostingClassifier(random_state=0),
    "Skl HistGBM": HistGradientBoostingClassifier(random_state=0),
    "XGBoost": XGBClassifier(),
    "LightGBM": LGBMClassifier(random_state=0),
    "CatBoost": CatBoostClassifier(random_state=0)}

    tree_classifiers_pipe = {name: make_pipeline(num_prep, model) for name, model in tree_classifiers.items()}
    return tree_classifiers_pipe


# Benchmark accuracy

def model_perfromance(classifiers,X_train,y_train,X_test,y_test):   

    results = pd.DataFrame({'Model': [], 'Accuracy': [], 'Time': []})

    for model_name, model in classifiers.items():
        start_time = time.time()
        
        # Training
        model.fit(X_train, y_train)
        
        # Prediction
        pred = model.predict(X_test)

        total_time = time.time() - start_time # Time taken to fit and predict

        model_results = pd.DataFrame({"Model":    [model_name],
                                "Accuracy": [accuracy_score(y_test, pred)*100],
                                "Time":     [total_time]})
        results = pd.concat([results, model_results])
        
        

    results_ord = results.sort_values(by=['Accuracy'], ascending=False, ignore_index=True)
    print('Accuracy and timing of models \n',results_ord)


"""
           Model   Accuracy      Time
0       AdaBoost  91.803279  0.110465
1    Extra Trees  88.524590  0.195881
2  Random Forest  86.885246  0.213867
3       CatBoost  86.885246  4.362975
4        Skl GBM  81.967213  0.113930
5        XGBoost  81.967213  0.142911
6       LightGBM  80.327869  0.065960
7    Skl HistGBM  78.688525  0.346785

"""


def generate_data(data):
    np.random.seed(0)
    gen_data = data.copy()
    sep_on = 'sex'
    divide_std_by = 10
    
    for value in data[sep_on].unique():
        sub_data =  gen_data[gen_data[sep_on] == value]

        age_std = sub_data['age'].std()
        
        trtbps_std = sub_data['trtbps'].std()
       
        chol_std = sub_data['chol'].std()
      
        thalachh_std = sub_data['thalachh'].std()
    
        oldpeak_std = sub_data['oldpeak'].std()
     
        
        for i in gen_data[gen_data[sep_on] == value].index:
            if np.random.randint(2) == 1:
                gen_data.loc[i,'age'] += age_std/divide_std_by
                gen_data.loc[i,'trtbps'] += trtbps_std/divide_std_by
                gen_data.loc[i,'chol'] += chol_std/divide_std_by
                gen_data.loc[i,'thalachh'] += thalachh_std/divide_std_by
                gen_data.loc[i,'oldpeak'] += oldpeak_std/divide_std_by

            else:
                gen_data.loc[i,'age'] -= age_std/divide_std_by
                gen_data.loc[i,'trtbps'] -= trtbps_std/divide_std_by
                gen_data.loc[i,'chol'] -= chol_std/divide_std_by
                gen_data.loc[i,'thalachh'] -= thalachh_std/divide_std_by
                gen_data.loc[i,'oldpeak'] -= oldpeak_std/divide_std_by
                
    return gen_data      

 

def augmented_data(generated_data,X_train,y_train):
    np.random.seed(0)
    gen = generated_data
    extra_data = gen.sample(gen.shape[0] // 5)
    X_train_aug = pd.concat([X_train, extra_data.drop(['output'], axis=1) ])
    y_train_aug = pd.concat([y_train, extra_data['output'] ])
    # print(f'Augmented X_train by {((len(X_train_aug) - len(X_train)) / len(X_train)) * 100 }%')
    return X_train_aug, y_train_aug

   
     

  
def final_model(classifier,model_name,X_train_aug, y_train_aug):

    final_model = classifier[model_name]
    final_model.fit(X_train_aug,y_train_aug)
    return final_model 


    


    
    

    

    