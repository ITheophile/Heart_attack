from msilib.schema import Class
import heart_attack as HR
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

#Loading the data
filepath = 'data/heart.csv'
# data = HR.load_and_cleaning(filepath)
# print(data.head())
data = HR.load_and_cleaning(filepath)
dropping =  data.drop(columns=['chol','fbs','restecg'])
# print(dropping.head())
#Spltting the dataset
X_train, X_test, y_train, y_test = HR.splitting_data(dropping)
# print('Xtrain shape', X_train.shape)
# print('ytrain shape', y_train.shape)
# print('Xtest shape', X_test.shape)
# print('ytest shape', y_test.shape)

# #Getting the classifies from preprocessing 
num_var = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
Classifiers = HR.data_preprocessing(num_var)
# # print(Classifiers.keys())

# #Checking model perfomance of the classifiers

HR.model_perfromance(Classifiers,X_train,y_train,X_test, y_test)

# #Generating new data
# gen_data = HR.generate_data(data)

# #Data enhancement
# aug_xtrain, aug_ytrain= HR.augmented_data(gen_data, X_train, y_train)

# #Check the model performance of augmented data
# # HR.model_perfromance(Classifiers,aug_xtrain,aug_ytrain,X_test,y_test)

# """
#             Model    Accuracy       Time
# 0    Extra Trees  100.000000   0.280731
# 1       CatBoost   98.360656  15.228880
# 2        XGBoost   96.721311   0.231857
# 3       LightGBM   96.721311   0.104447
# 4  Random Forest   95.081967   0.337612
# 5    Skl HistGBM   95.081967   0.536668
# 6        Skl GBM   93.442623   0.169894
# 7       AdaBoost   90.163934   0.183887
# """

# #Getting the final model
# fin_model = HR.final_model(Classifiers,'LightGBM',aug_xtrain, aug_ytrain)
# print(fin_model.predict(X_test))
# plot_confusion_matrix(fin_model, X_test, y_test)
# tick_marks = [0,1]
# plt.xticks(tick_marks,labels=['No heart attack','Heart attack'] )
# plt.yticks(tick_marks,labels=['No heart attack','Heart attack'] )
# plt.show()







