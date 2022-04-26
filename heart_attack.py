from ast import Or
from multiprocessing import Pipe
import numpy as np
import pandas as pd
from scipy import rand
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, plot_confusion_matrix,classification_report
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble      import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

# Load and Prepro 

# Load data
data = pd.read_csv('data/heart.csv')

# remove duplicates
data = data.drop_duplicates()
# print('Shape of data after dropping duplicate',data.shape)

# Train test sets
X = data.drop(columns='output')
y = data['output']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
print('Xtrain shape', X_train.shape)
print('ytrain shape', y_train.shape)
print('Xtest shape', X_test.shape)
print('ytest shape', y_test.shape)

# Preprocessor
num_var = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']


num_prep = ColumnTransformer([('num_prepo', StandardScaler(), num_var)],
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

# Benchmark accuracy
# results = pd.DataFrame({'Model': [], 'Accuracy': [], 'Time': []})

# for model_name, model in tree_classifiers_pipe.items():
#     start_time = time.time()
    
#     # Training
#     model.fit(X_train, y_train)
    
#     # Prediction
#     pred = model.predict(X_test)

#     total_time = time.time() - start_time # Time taken to fit and predict

#     model_results = pd.DataFrame({"Model":    [model_name],
#                               "Accuracy": [accuracy_score(y_test, pred)*100],
#                               "Time":     [total_time]})
#     results = pd.concat([results, model_results])
    
    

# results_ord = results.sort_values(by=['Accuracy'], ascending=False, ignore_index=True)
# print('Accuracy and timing of models \n',results_ord)


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


def data_enhancement(data):
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

 

def Augmented_data():
        
    gen = data_enhancement(data)
    np.random.seed(0)
    extra_data = gen.sample(gen.shape[0] // 5)
    X_train_aug = pd.concat([X_train, extra_data.drop(['output'], axis=1) ])
    y_train_aug = pd.concat([y_train, extra_data['output'] ])
    # print(f'Augmented X_train by {((len(X_train_aug) - len(X_train)) / len(X_train)) * 100 }%')
    return X_train_aug, y_train_aug

def new_model():
    Xtrain_aug, ytrain_aug = Augmented_data()
    tree_classifiers_pipe_augg = {names: make_pipeline(num_prep, models) for names, models in tree_classifiers.items()}
    results_aug = pd.DataFrame({'Models': [], 'Accuracies': [], 'Timing': []})

    for model_names1, models1 in tree_classifiers_pipe_augg.items():
        start_time = time.time()
        
        # Training
        models1.fit(Xtrain_aug, ytrain_aug)
        
        # Prediction
        pred_aug = models1.predict(X_test)

        total_time = time.time() - start_time # Time taken to fit and predict

        model_results_aug = pd.DataFrame({"Models":    [model_names1],
                                "Accuracies": [accuracy_score(y_test, pred_aug)*100],
                                "Timing":     [total_time]})


        results_aug = pd.concat([results_aug, model_results_aug])
        results_ord_aug = results_aug.sort_values(by=['Accuracies'], ascending=False, ignore_index=True)
    print(results_ord_aug)
    return tree_classifiers_pipe_augg   
            

  
def final_model():
    final_tree_classifiers_pipe_augg = new_model()
    Xtrain_final, ytrain_final= Augmented_data()
    final_mod = final_tree_classifiers_pipe_augg['Extra Trees']

    final_mod.fit(Xtrain_final, ytrain_final)

    pred_final = final_mod.predict(X_test)
    # print(classification_report(y_test, pred_final))
    plot_confusion_matrix(final_mod, X_test, y_test)


    
final_model()   

    
    

    

    