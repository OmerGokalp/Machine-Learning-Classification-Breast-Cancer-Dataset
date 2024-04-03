# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 12:42:57 2024

@author: omer
"""
#Breast-Cancer-Classification

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#veri yükleme
veriler = pd.read_excel('Breast.xlsx')
print(veriler)


#Nan verilerin ortalamasının alınarak numerik hale getirilmesi
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

yas = veriler.iloc[:,0:2].values
imputer = imputer.fit(yas[:,0:2])
yas[:,0:2] = imputer.transform(yas[:,0:2])
print(yas)


#Label Encodinge hazırlamak için etiketli verilerin kolon olarak çekilmesi
Tissue_composition = veriler.iloc[:,2:3].values
print(Tissue_composition)

Yas = yas
print(Yas)

Signs = veriler.iloc[:,3:4].values
print(Signs)

Symptoms = veriler.iloc[:,4:5].values
print(Symptoms)

Shape = veriler.iloc[:,5:6].values
print(Shape)

Margin = veriler.iloc[:,6:7].values
print(Margin)

Echogenicity = veriler.iloc[:,7:8].values
print(Echogenicity)

Posterior_features = veriler.iloc[:,8:9].values
print(Posterior_features)

Halo = veriler.iloc[:,9:10].values
print(Halo)

Calcifications = veriler.iloc[:,10:11].values
print(Calcifications)

Skin_thickening = veriler.iloc[:,11:12].values
print(Skin_thickening)

Interpretation = veriler.iloc[:,12:13].values
print(Interpretation)

BIRADS = veriler.iloc[:,13:14].values
print(BIRADS)

Verification = veriler.iloc[:,14:15].values
print(Verification)

Diagnosis = veriler.iloc[:,15:16].values
print(Diagnosis)

#Numerik olmayan verilerin 0-1 haline getirilmesi
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

#Tissue_composition
Tissue_composition[:,0] = le.fit_transform(veriler.iloc[:,2])
print(Tissue_composition)

ohe = preprocessing.OneHotEncoder()
Tissue_composition = ohe.fit_transform(Tissue_composition).toarray()
print(Tissue_composition)

#Signs
Signs[:,0] = le.fit_transform(veriler.iloc[:,3])
print(Signs)

ohe = preprocessing.OneHotEncoder()
Signs = ohe.fit_transform(Signs).toarray()
print(Signs)

#Symptoms
Symptoms[:,0] = le.fit_transform(veriler.iloc[:,4])
print(Symptoms)

ohe = preprocessing.OneHotEncoder()
Symptoms = ohe.fit_transform(Symptoms).toarray()
print(Symptoms)

#Shape
Shape[:,0] = le.fit_transform(veriler.iloc[:,5])
print(Shape)

ohe = preprocessing.OneHotEncoder()
Shape = ohe.fit_transform(Shape).toarray()
print(Shape)

#Margin
Margin[:,0] = le.fit_transform(veriler.iloc[:,6])
print(Margin)

ohe = preprocessing.OneHotEncoder()
Margin = ohe.fit_transform(Margin).toarray()
print(Margin)

#Echogenicity
Echogenicity[:,0] = le.fit_transform(veriler.iloc[:,7])
print(Echogenicity)

ohe = preprocessing.OneHotEncoder()
Echogenicity = ohe.fit_transform(Echogenicity).toarray()
print(Echogenicity)

#Posterior_features
Posterior_features[:,0] = le.fit_transform(veriler.iloc[:,8])
print(Posterior_features)

ohe = preprocessing.OneHotEncoder()
Posterior_features = ohe.fit_transform(Posterior_features).toarray()
print(Posterior_features)

#Halo
Halo[:,0] = le.fit_transform(veriler.iloc[:,9])
print(Halo)

ohe = preprocessing.OneHotEncoder()
Halo = ohe.fit_transform(Halo).toarray()
print(Halo)

#Calcifications
Calcifications[:,0] = le.fit_transform(veriler.iloc[:,10])
print(Calcifications)

ohe = preprocessing.OneHotEncoder()
Calcifications = ohe.fit_transform(Calcifications).toarray()
print(Calcifications)

#Skin_thickening
Skin_thickening[:,0] = le.fit_transform(veriler.iloc[:,11])
print(Skin_thickening)

ohe = preprocessing.OneHotEncoder()
Skin_thickening = ohe.fit_transform(Skin_thickening).toarray()
print(Skin_thickening)

#Interpretation
Interpretation[:,0] = le.fit_transform(veriler.iloc[:,12])
print(Interpretation)

ohe = preprocessing.OneHotEncoder()
Interpretation = ohe.fit_transform(Interpretation).toarray()
print(Interpretation)

#Verification
Verification[:,0] = le.fit_transform(veriler.iloc[:,14])
print(Verification)

ohe = preprocessing.OneHotEncoder()
Verification = ohe.fit_transform(Verification).toarray()
print(Verification)

# #Diagnosis
Diagnosis[:,0] = le.fit_transform(veriler.iloc[:,15])
print(Diagnosis)

ohe = preprocessing.OneHotEncoder()
Diagnosis = ohe.fit_transform(Diagnosis).toarray()
print(Diagnosis)



#Numerik olmayan verilerin etiketlenmesi
sonuc = pd.DataFrame(data=Tissue_composition, index = range (256),columns = ['heterogeneous:predominantly fat','homogeneous:fat','homogeneous:fibroglandular','heterogeneous:predominantly fibroglandular','not available','lactating','lactating&homogeneous:fibroglandular','lactating&heterogeneous:predominantly fibroglandular','lactating&heterogeneous:predominantly fat'])
print(sonuc)

sonuc2 = pd.DataFrame(data = Yas, index = range  (256), columns=['Pixel_size','Age'])
print(sonuc2)

sonuc3 = pd.DataFrame(data = Signs, index = range (256), columns = ['breast scar','not available','no','nipple retraction&palpable','palpable','breast scar&skin retraction','nipple retraction','peau dorange&palpable','warmth&palpable','redness&warmth&palpable','skin retraction&palpable','redness&warmth','palpable&breast scar'])
print(sonuc3)

sonuc4 = pd.DataFrame(data = Symptoms, index = range(256), columns = ['family history of breast/ovarian cancer','not available','nipple discharge','no','HRT/hormonal contraception','personal history of breast cancer&family history of breast/ovarian cancer','family history of breast/ovarian cancer&HRT/hormonal contraception','personal history of breast cancer','nipple discharge&family history of breast/ovarian cancer','breast injury'])
print(sonuc4)

sonuc5 = pd.DataFrame(data = Shape, index = range(256), columns = ['irregular','not applicable','oval','round'])
print(sonuc5)

sonuc6 = pd.DataFrame(data = Margin, index = range(256), columns = ['circumscribed','not applicable','not circumscribed - angular','not circumscribed - angular&indistinct','not circumscribed - angular&microlobulated','not circumscribed - angular&microlobulated&indistinct','not circumscribed - indistinct','not circumscribed - microlobulated','not circumscribed - microlobulated&indistinct','not circumscribed - spiculated','not circumscribed - spiculated&angular','not circumscribed - spiculated&angular&indistinct','not circumscribed - spiculated&angular&microlobulated&indistinct','not circumscribed - spiculated&indistinct','not circumscribed - spiculated&microlobulated&indistinct'])
print(sonuc6)

sonuc7 = pd.DataFrame(data = Echogenicity, index = range(256), columns = ['anechoic','complex cystic/solid','heterogeneous','hyperechoic','hypoechoic','isoechoic','not applicable'])
print(sonuc7)

sonuc8 = pd.DataFrame(data = Posterior_features, index = range(256), columns = ['combined','enhancement','no','not applicable','shadowing'])
print(sonuc8)

sonuc9 = pd.DataFrame(data = Halo, index = range(256), columns = ['no','not applicable','yes'])
print(sonuc9)

sonuc10 = pd.DataFrame(data = Calcifications, index = range(256), columns = ['in a mass','indefinable','intraductal','no','not applicable'])
print(sonuc10)

sonuc11 = pd.DataFrame(data = Skin_thickening, index = range(256), columns=['no','yes'])
print(sonuc11)

sonuc12 = pd.DataFrame(data = Interpretation, index = range(256), columns = ['Breast scar (surgery)','Breast scar (surgery)&Breast scar (radiotherapy)','Complex cyst / Non-simple cyst','Complex cyst / Non-simple cyst&Cyst filled with thick fluid','Complex cyst / Non-simple cyst&Cyst filled with thick','Complex cyst / Non-simple cyst&Cyst filled with thick','Complex cyst / Non-simple cyst&Cyst filled with thick','Complex cyst / Non-simple cyst&Cyst filled with thick','Complex cyst / Non-simple cyst&Cyst filled with thick','Complex cyst / Non-simple cyst&Cyst filled with thick','Complex cyst / Non-simple cyst&Dysplasia&Hamartoma','Complex cyst / Non-simple cyst&Dysplasia&Mammary duct ectasia','Complex cyst / Non-simple cyst&Fibroadenoma','Cyst filled with thick fluid','Cyst filled with thick fluid&Dysplasia&Duct filled with thick fluid&Intraductal papilloma','Cyst filled with thick fluid&Dysplasia&Duct filled with thick fluid&Mammary duct ectasia','Cyst filled with thick fluid&Dysplasia&Fibroadenoma','Cyst filled with thick','Cyst filled with thick fluid&Fibroadenoma','Cyst filled with thick fluid&Fibroadenoma&Silicone implant','Cyst filled with thick fluid&Intramammary lymph node','Dysplasia&Fibroadenoma','Duct filled with thick fluid&Mammary duct ectasia','Duct filled with thick fluid&Mammary duct ectasia&Intraductal papilloma','Dysplasia','Dysplasia&Fibroadenoma','Dysplasia&Fibroadenoma&Adenosis','Dysplasia&Fibroadenoma&Duct filled with thick fluid&Intraductal papilloma','Dysplasia&Fibroadenoma&Hamartoma','Dysplasia&Fibroadenoma&Intraductal papilloma','Dysplasia&Lipoma','Dysplasia&Mammary duct ectasia','Fat necrosis&Breast scar (surgery)','Fat necrosis&Hematoma','Fibroadenoma','Fibroadenoma&Cyst filled with thick fluid','Fibroadenoma&Duct filled with thick fluid','Fibroadenoma&Hamartoma','Fibroadenoma&Hamartoma&Adenosis','Fibroadenoma&Intraductal papilloma','Fibroadenoma&Isolated calcifications','Fibroadenoma&Lactating adenoma','Fibroadenoma&Lacteal cyst','Hamartoma','Hematoma&Breast scar (surgery)','Implant rupture&Silicone implant','Intramammary lymph node','Lipoma','Lipoma&Hemangioma','Mammary duct ectasia','Mastitis','Seroma','Simple cyst','Simple cyst&Complex cyst / Non-simple cyst','Suspicion of malignancy','Suspicion of malignancy&Complex cyst / Non-simple cyst','Suspicion of malignancy&Complex cyst / Non-simple cyst&Dysplasia','Suspicion of malignancy&Complex cyst / Non-simple cyst&Dysplasia&Fibroadenoma&Mammary duct ectasia&Intraductal papilloma','Suspicion of malignancy&Cyst filled with thick fluid&Dysplasia&Fibroadenoma','Suspicion of malignancy&Dysplasia','Suspicion of malignancy&Dysplasia&Breast scar (surgery)','Suspicion of malignancy&Dysplasia&Duct filled with thick fluid&Lacteal cyst','Suspicion of malignancy&Dysplasia&Duct filled with thick fluid&Mammary duct ectasia&Intraductal papilloma','Suspicion of malignancy&Dysplasia&Fibroadenoma','Suspicion of malignancy&Dysplasia&Fibroadenoma&Intraductal papilloma','Suspicion of malignancy&Dysplasia&Fibroadenoma&Intraductal papilloma&Hamartoma','Suspicion of malignancy&Dysplasia&Fibroadenoma&Intraductal papilloma&Phyllodes tumor','Suspicion of malignancy&Dysplasia&Intraductal papilloma','Suspicion of malignancy&Dysplasia&Intraductal papilloma&Hematoma','Suspicion of malignancy&Dysplasia&Mammary duct ectasia','Suspicion of malignancy&Dysplasia&Silicone implant','Suspicion of malignancy&Fibroadenoma','Suspicion of malignancy&Fibroadenoma&Hamartoma','Suspicion of malignancy&Fibroadenoma&Intraductal papilloma','Suspicion of malignancy&Fibroadenoma&Phyllodes tumor','Suspicion of malignancy&Intraductal papilloma','Suspicion of malignancy&Intraductal papilloma&Mastitis','Suspicion of malignancy&Mammary duct ectasia&Intraductal papilloma','Suspicion of malignancy&Mastitis','Suspicion of malignancy&Mastitis&Abscess&Hematoma&Breast scar (surgery)','not applicable'])
print(sonuc12)

sonuc13 = pd.DataFrame(data = BIRADS, index = range(256), columns = ['BIRADS'])
print(sonuc13)

sonuc14 = pd.DataFrame(data = Verification, index = range(256), columns = ['confirmed by biopsy','confirmed by follow-up care','not applicable'])
print(sonuc14)

sonuc15 = pd.DataFrame(data = Diagnosis, index = range(256), columns = ['Atypical lobular hyperplasia (ALH)','Benign mammary dysplasia','Benign mammary dysplasia&Usual ductal hyperplasia (UDH)','Complex sclerosing lesion','Cribriform carcinoma','Ductal carcinoma in situ (DCIS)','Ductal carcinoma in situ (DCIS)&Invasive carcinoma of no special type (NST)','Ductal carcinoma in situ (DCIS)&Solid papillary carcinoma in situ','Encapsulated papillary carcinoma&Ductal carcinoma in situ (DCIS)','Fat necrosis','Fibroadenoma','Fibroadenoma&Phyllodes tumor','Fibroadenosis','Fibrocystic change','Fibrosclerosis','Hamartoma','Intraductal papilloma','Intraductal papilloma&Usual ductal hyperplasia (UDH)&Adenosis','Intramammary lymph node','Invasive carcinoma of no special type (NST)','Invasive carcinoma of no special type (NST)&Apocrine carcinoma','Invasive carcinoma of no special type (NST)&Ductal carcinoma in situ (DCIS)','Invasive carcinoma of no special type (NST)&Invasive micropapillary carcinoma','Invasive carcinoma of no special type (NST)&Mucinous carcinoma','Invasive carcinoma of no special type (NST)&Sebaceous carcinoma','Invasive lobular carcinoma','Invasive lobular carcinoma&Lobular carcinoma in situ (LCIS)','Invasive micropapillary carcinoma','Invasive papillary carcinoma&Encapsulated papillary carcinoma','Lactating adenoma','Lobular carcinoma in situ (LCIS)','Lymphoma','Mastitis','Metaplastic carcinoma','Mucinous carcinoma','Phyllodes tumor','Pseudoangiomatous stromal hyperplasia (PASH)','Simple cyst','Tubular carcinoma','Tubular carcinoma&Cribriform carcinoma','Usual ductal hyperplasia (UDH)','Usual ductal hyperplasia (UDH)&Fibrocystic change','Usual ductal hyperplasia (UDH)&Fibrosclerosis','Usual ductal hyperplasia (UDH)&Pseudoangiomatous stromal hyperplasia (PASH)','not applicable'])
print(sonuc15)


#Frame olarak ayrılmış verilerin bir araya getirilmesi
s1 = pd.concat([sonuc2,sonuc], axis = 1)
print(s1)

s2 = pd.concat([s1,sonuc3], axis = 1)
print(s2)

s3 = pd.concat([s2,sonuc4], axis = 1)
print(s3)

s4 = pd.concat([s3,sonuc5], axis = 1)
print(s4)

s5 = pd.concat([s4,sonuc6], axis = 1)
print(s5)

s6 = pd.concat([s5,sonuc7], axis = 1)
print(s6)

s7 = pd.concat([s6,sonuc8], axis = 1)
print(s7)

s8 = pd.concat([s7,sonuc9], axis = 1)
print(s8)

s9 = pd.concat([s8,sonuc10], axis = 1)
print(s9)

s10 = pd.concat([s9,sonuc11], axis = 1)
print(s10)

s11 = pd.concat([s10,sonuc12], axis = 1)
print(s11)

s12 = pd.concat([s11,sonuc13], axis = 1)
print(s12)

s13 = pd.concat([s12,sonuc14], axis = 1)
print(s13)

fixed_data = pd.concat([s13,sonuc15], axis = 1)
print(fixed_data)


x = fixed_data.iloc[:,0:250].values #bağımsız değişkenler 
y = veriler.iloc[:,-1].values  #bağımlı değişlenler

#Verilerin train ve test olarak ayrılması
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.33, random_state=0)


#Verilerin train ve teste fit edilmesi
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#Precision and Recall import
from sklearn.metrics import precision_score, recall_score

#Accuracy Score
from sklearn.metrics import accuracy_score

#F-1 Score
from sklearn.metrics import f1_score

#G-Mean Score
from imblearn.metrics import geometric_mean_score

#TAHMİNLER


#LOGISTIC REGRESSION PREDICTION
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train, y_train)

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('Logistic Regression Confusion Matrix')
print(cm)

# Logistic Regression için precision ve recall hesaplama
precision_lr = precision_score(y_test, y_pred, average='weighted')
recall_lr = recall_score(y_test, y_pred, average='weighted')
print("Logistic Regression Precision:", precision_lr)
print("Logistic Regression Recall:", recall_lr)

# Logistic Regression için accuracy hesaplama
accuracy_lr = accuracy_score(y_test, y_pred)
print("Logistic Regression Accuracy:", accuracy_lr)

# Logistic Regression için F1-score hesaplama
f1_lr = f1_score(y_test, y_pred, average='weighted')
print("Logistic Regression F1-score:", f1_lr)

# Logistic Regression için G-Mean skoru hesaplama
gmean_lr = geometric_mean_score(y_test, y_pred, average='weighted')
print("Logistic Regression G-Mean Score:", gmean_lr)

#K-Nearest Neighborhood
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7, metric='minkowski')
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('KNN Confusion Matrix')
print(cm)

# KNN için precision ve recall hesaplama
precision_lr = precision_score(y_test, y_pred, average='weighted')
recall_lr = recall_score(y_test, y_pred, average='weighted')
print("KNN Precision:", precision_lr)
print("KNN Recall:", recall_lr)

# KNN için accuracy hesaplama
accuracy_knn = accuracy_score(y_test, y_pred)
print("KNN Accuracy:", accuracy_knn)

# KNN için F1-score hesaplama
f1_knn = f1_score(y_test, y_pred, average='weighted')
print("KNN F1-score:", f1_knn)

# KNN için G-Mean skoru hesaplama
gmean_knn = geometric_mean_score(y_test, y_pred, average='weighted')
print("KNN G-Mean Score:", gmean_knn)

#Support Vector Machine
from sklearn.svm import SVC
svc= SVC(kernel='linear')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('SVM linear Confusion Matrix')
print(cm)

# SVM için precision ve recall hesaplama
precision_lr = precision_score(y_test, y_pred, average='weighted')
recall_lr = recall_score(y_test, y_pred, average='weighted')
print("SVM Precision:", precision_lr)
print("SVM Recall:", recall_lr)

# SVM linear için accuracy hesaplama
accuracy_svm_linear = accuracy_score(y_test, y_pred)
print("SVM linear Accuracy:", accuracy_svm_linear)

# SVM linear için F1-score hesaplama
f1_svm_linear = f1_score(y_test, y_pred, average='weighted')
print("SVM linear F1-score:", f1_svm_linear)

# SVM linear için G-Mean skoru hesaplama
gmean_svm_linear = geometric_mean_score(y_test, y_pred, average='weighted')
print("SVM linear G-Mean Score:", gmean_svm_linear)

#Linear yerine farkli metrikler denemek
svc= SVC(kernel='poly')
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('SVM polynomial Confusion Matrix')
print(cm)

# SVM POLY için precision ve recall hesaplama
precision_lr = precision_score(y_test, y_pred, average='weighted')
recall_lr = recall_score(y_test, y_pred, average='weighted')
print("SVM POLY Precision:", precision_lr)
print("SVM POLY Recall:", recall_lr)

# SVM polynomial için accuracy hesaplama
accuracy_svm_poly = accuracy_score(y_test, y_pred)
print("SVM polynomial Accuracy:", accuracy_svm_poly)

# SVM polynomial için F1-score hesaplama
f1_svm_poly = f1_score(y_test, y_pred, average='weighted')
print("SVM polynomial F1-score:", f1_svm_poly)

# SVM polynomial için G-Mean skoru hesaplama
gmean_svm_poly = geometric_mean_score(y_test, y_pred, average='weighted')
print("SVM polynomial G-Mean Score:", gmean_svm_poly)

svc= SVC(kernel='rbf')
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('SVM rbf Confusion Matrix')
print(cm)

# SVM RBF için precision ve recall hesaplama
precision_lr = precision_score(y_test, y_pred, average='weighted')
recall_lr = recall_score(y_test, y_pred, average='weighted')
print("SVM RBF Precision:", precision_lr)
print("SVM RBF:", recall_lr)

# SVM RBF için accuracy hesaplama
accuracy_svm_rbf = accuracy_score(y_test, y_pred)
print("SVM RBF Accuracy:", accuracy_svm_rbf)

# SVM RBF için F1-score hesaplama
f1_svm_rbf = f1_score(y_test, y_pred, average='weighted')
print("SVM RBF F1-score:", f1_svm_rbf)

# SVM RBF için G-Mean skoru hesaplama
gmean_svm_rbf = geometric_mean_score(y_test, y_pred, average='weighted')
print("SVM RBF G-Mean Score:", gmean_svm_rbf)

#Naive Bayes 
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('GNB Confusion Matrix')
print(cm)

# Naive Bayes için precision ve recall hesaplama
precision_lr = precision_score(y_test, y_pred, average='weighted')
recall_lr = recall_score(y_test, y_pred, average='weighted')
print("Naive Bayes Precision:", precision_lr)
print("Naive Bayes Recall:", recall_lr)

# Naive Bayes için accuracy hesaplama
accuracy_nb = accuracy_score(y_test, y_pred)
print("Naive Bayes Accuracy:", accuracy_nb)

# Naive Bayes için F1-score hesaplama
f1_nb = f1_score(y_test, y_pred, average='weighted')
print("Naive Bayes F1-score:", f1_nb)

# Naive Bayes için G-Mean skoru hesaplama
gmean_nb = geometric_mean_score(y_test, y_pred, average='weighted')
print("Naive Bayes G-Mean Score:", gmean_nb)

#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('Decision Tree Confusion Matrix')
print(cm)

# Decision Tree için precision ve recall hesaplama
precision_lr = precision_score(y_test, y_pred, average='weighted')
recall_lr = recall_score(y_test, y_pred, average='weighted')
print("Decision Tree Precision:", precision_lr)
print("Decision Tree Recall:", recall_lr)

# Decision Tree için accuracy hesaplama
accuracy_dt = accuracy_score(y_test, y_pred)
print("Decision Tree Accuracy:", accuracy_dt)

# Decision Tree için F1-score hesaplama
f1_dt = f1_score(y_test, y_pred, average='weighted')
print("Decision Tree F1-score:", f1_dt)

# Decision Tree için G-Mean skoru hesaplama
gmean_dt = geometric_mean_score(y_test, y_pred, average='weighted')
print("Decision Tree G-Mean Score:", gmean_dt)

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion='entropy')
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('Random Forest Confusion Matrix')
print(cm)

# Random Forest için precision ve recall hesaplama
precision_lr = precision_score(y_test, y_pred, average='weighted')
recall_lr = recall_score(y_test, y_pred, average='weighted')
print("Random Forest Precision:", precision_lr)
print("Random Forest Recall:", recall_lr)

# Random Forest için accuracy hesaplama
accuracy_rf = accuracy_score(y_test, y_pred)
print("Random Forest Accuracy:", accuracy_rf)

# Random Forest için F1-score hesaplama
f1_rf = f1_score(y_test, y_pred, average='weighted')
print("Random Forest F1-score:", f1_rf)

# Random Forest için G-Mean skoru hesaplama
gmean_rf = geometric_mean_score(y_test, y_pred, average='weighted')
print("Random Forest G-Mean Score:", gmean_rf)






