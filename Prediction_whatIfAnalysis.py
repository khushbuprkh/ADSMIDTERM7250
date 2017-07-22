
# coding: utf-8

# In[39]:

# !pip install urllib
import requests
from bs4 import BeautifulSoup
import re
import os 
import time
from io import BytesIO
from zipfile import ZipFile
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv
import math
import h2o
from sklearn import linear_model
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression 
from sklearn.svm import LinearSVC
from sklearn.exceptions import NotFittedError
from sklearn.svm import SVR
from itertools import chain, combinations
from sklearn.cross_validation import cross_val_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from sklearn.preprocessing import MinMaxScaler


# In[40]:

fileDir = os.path.dirname(os.path.realpath('__file__'))


# In[41]:

baseUrl='https://freddiemac.embs.com/FLoan/'
postUrl='Data/download.php'


# In[42]:

def createCredentialData(user, passwd):
    creds={'username': user,'password': passwd}
    return creds

def getFilesFromFreddieMac(cred):
    ## We are using inside WITH BLock so that session is closed ASAP with BLock is exited 
    with requests.Session() as s:
        ## Step 1 routing to auth.php Site with the proper crentials 
        urlOne = s.post(baseUrl+"secure/auth.php", data=cred) 
        if "Please log in" in urlOne.text:
        ## IF CREDENTIALS are not valid Throw Alert 
            print("Alert: Invalid Credentials, Please try again or sign up on below site \n https://freddiemac.embs.com/FLoan/Bin/loginrequest.php")
        else:
            print("Step1: Logged in")
        ## Sterp 2 Preparing the data for to Accept terms and Conditions 
            pay2={'accept': 'Yes','acceptSubmit':'Continue','action':'acceptTandC'}
            finalUrl=s.post(baseUrl +"Data/download.php",pay2)
            if "Loan-Level Dataset" in finalUrl.text:
                      print("Step2 : Terms and Conditions Accepted")
                      soup = BeautifulSoup(finalUrl.content, "html.parser")   
                      links_list = soup.findAll('a')
                      print("Step3: Filtered the Sample Files with Condition== 2007/20008/2009/1999/2013")
                      print("Status::::::::::")
                      for ele in links_list:
        ## Filtering the ZIp files >= 2005 
                         if 'historical' in ele.get_text():
                            if(ele.get_text()[-8:-4] == '2007' or ele.get_text()[-8:-4] == '2008' or ele.get_text()[-8:-4] == '2009' or ele.get_text()[-8:-4] == '2010' or ele.get_text()[-8:-4] == '2013' or ele.get_text()[-8:-4] == '1999'):
                                    print(ele.get_text()[-8:-4])
                                    tempUrl = baseUrl+"Data/"+ele.get('href')                         
                                    b =time.time()
                                    downloadUrl=s.post(tempUrl) ## return type = Response
                                    e=time.time()
                                    print(tempUrl + " took "+ str(e-b)+" sec")
                                    with ZipFile(BytesIO(downloadUrl.content)) as zfile:
                                          zfile.extractall(os.path.join(fileDir, 'adsDataRepo/'+'Historical_data_'+ele.get_text()[-8:-4]+'/'))
                                          print("File "+ ele.get_text()+" Downloaded")
    
            else:
                print("Alert: Please Check the rerouting action suffix")
        


# In[43]:

def preProcessData(inputQuater,inputYear,inputQuaterTwo,inputYearTwo):
    cleandataOne= ""
    cleandataTwo= ""
    print("pre-process data")
    if(os.path.exists(fileDir+'/adsDataRepo/')):
        trainingDataFile = glob.glob(fileDir+'/adsDataRepo/'+'Historical_data_'+inputYear+'/historical_data1_'+inputQuater+inputYear+'.txt')
        testingDataFile = glob.glob(fileDir+'/adsDataRepo/'+'Historical_data_'+inputYearTwo+'/historical_data1_'+inputQuaterTwo+inputYearTwo+'.txt')
        headerNames = ['CreditScore','FirstPaymentDate','FirstTimeHomeBuyerFlag','MaturityDate','MSA','MIP','NumberOfUnits',
                         'OccupancyStatus','OCLTV','DTI','OriginalUPB','OLTV','OriginalInterestRate','Channel','PrepaymentPenaltyFlag',
                         'ProductType','PropertyState','PropertyType','PostalCode','LoanSequenceNumber','LoanPurpose',
                         'OriginalLoanTerm','NumberOfBorrowers','SellerName','ServicerName','SuperConformingFlag']
        with open(trainingDataFile[0]) as f:
            dataf = pd.read_table(f, sep='|', low_memory=False, header=None,lineterminator='\n', names= headerNames)
            cleandataOne = originationDatacleaning(dataf)
            cleandataOne.to_csv("Origination_Clean_"+inputQuater+inputYear+".csv",index=False)
            print("training data cleaned, CSV Created")
       
        with open(testingDataFile[0]) as f:
            dataf = pd.read_table(f, sep='|', low_memory=False, header=None,lineterminator='\n', names= headerNames)
            cleandataTwo = originationDatacleaning(dataf)
            cleandataTwo.to_csv("Origination_Clean_"+inputQuaterTwo+inputYearTwo+".csv",index=False)
            print("testing data cleaned, CSV Created")

    return cleandataOne,cleandataTwo


# In[44]:

def originationDatacleaning(dataf):
    dataf['CreditScore'].replace('   ',301,inplace=True)
    dataf['CreditScore'].fillna(301,inplace=True)
    dataf['FirstTimeHomeBuyerFlag'].fillna('X',inplace=True) 
    dataf['MSA'].replace('   ',0,inplace=True)
    dataf['MSA'].fillna(0, inplace=True) 
    dataf['MIP'].replace('   ',0,inplace=True)
    dataf['MIP'].fillna(0, inplace=True)
    dataf['NumberOfUnits'].fillna(0,inplace=True)
    dataf['OccupancyStatus'].fillna('X',inplace=True)
    dataf['OCLTV'].replace('   ',0,inplace=True)
    dataf['OCLTV'].fillna(0,inplace=True)
    dataf['DTI'].replace('   ',0,inplace=True)
    dataf['DTI'].fillna(0,inplace=True)
    dataf['OriginalUPB'].replace('   ',0,inplace=True)
    dataf['OriginalUPB'].fillna(0,inplace=True)
    dataf['OLTV'].replace('   ',0,inplace=True)
    dataf['OLTV'].fillna(0,inplace=True)
    dataf['OriginalInterestRate'].fillna(0,inplace=True)
    dataf['Channel'].fillna('X',inplace=True)
    dataf['PrepaymentPenaltyFlag'].fillna('X',inplace=True)
    dataf['ProductType'].fillna('XXXXX',inplace=True)
    dataf['PropertyState'].fillna('XX',inplace=True)
    dataf['PropertyType'].fillna('XX',inplace=True)
    dataf['PostalCode'].fillna(0,inplace=True)
    dataf['LoanSequenceNumber'].replace('', np.NaN).fillna(0,inplace=True)
    dataf['LoanPurpose'].fillna('X',inplace=True)
    dataf['OriginalLoanTerm'].replace('', np.NaN).fillna(0,inplace=True)
    dataf['NumberOfBorrowers'].fillna('01',inplace=True)
    dataf['SellerName'].fillna('X',inplace=True)
    dataf['ServicerName'].fillna('X',inplace=True)
    dataf['SuperConformingFlag'].fillna('X',inplace=True)
    
    #factorizing data 
    factorizeCategoricalColumn(dataf)
    
    #assingning datatype
    dataf[['PropertyState','LoanSequenceNumber']]=dataf[['PropertyState','LoanSequenceNumber']].astype('str')
    dataf[['FirstTimeHomeBuyerFlag','OccupancyStatus','Channel','PrepaymentPenaltyFlag','ProductType','PropertyType','CreditScore','LoanPurpose','SellerName','ServicerName','MSA','MIP','NumberOfUnits','DTI','OCLTV','OLTV','PostalCode','NumberOfBorrowers']]=dataf[['FirstTimeHomeBuyerFlag','OccupancyStatus','Channel','PrepaymentPenaltyFlag','ProductType','PropertyType','CreditScore','LoanPurpose','SellerName','ServicerName','MSA','MIP','NumberOfUnits','DTI','OCLTV','OLTV','PostalCode','NumberOfBorrowers']].astype('int64')
    
    #missinganalysis(dataf)
    
    return dataf
    '''As we can see we have the below Null Values presnt in the Data for all the Years (Only varying the Counts )
                       MSA           
    FirstTimeHomeBuyerFlag           
     PrepaymentPenaltyFlag          
         NumberOfBorrowers 
    We can ignore''' 


# In[45]:

def factorizeCategoricalColumn(cleanperfTrain):
        print('_________________________________________________________')
        print('Factorizing the Categorical Columns .....................')
        print('_________________________________________________________')

        cleanperfTrain['FirstTimeHomeBuyerFlag'] = pd.factorize(cleanperfTrain['FirstTimeHomeBuyerFlag'])[0]
        cleanperfTrain['OccupancyStatus'] = pd.factorize(cleanperfTrain['OccupancyStatus'])[0]
        cleanperfTrain['Channel'] = pd.factorize(cleanperfTrain['Channel'])[0]
        cleanperfTrain['ProductType'] = pd.factorize(cleanperfTrain['ProductType'])[0]
        cleanperfTrain['PropertyType'] = pd.factorize(cleanperfTrain['PropertyType'])[0]
        cleanperfTrain['LoanPurpose'] = pd.factorize(cleanperfTrain['LoanPurpose'])[0]
        cleanperfTrain['SellerName'] = pd.factorize(cleanperfTrain['SellerName'])[0]
        cleanperfTrain['ServicerName'] = pd.factorize(cleanperfTrain['ServicerName'])[0]
        cleanperfTrain['PrepaymentPenaltyFlag'] = pd.factorize(cleanperfTrain['PrepaymentPenaltyFlag'])[0]
        
        return cleanperfTrain


# In[46]:

def dropColumns(file):
    file.drop("FirstPaymentDate",axis=1,inplace=True)
    file.drop("MaturityDate",axis=1,inplace=True)
    file.drop("PostalCode",axis=1,inplace=True)


# In[47]:

def computeMae(model_mae,y,x):
    model= model_mae
    pred=model.predict(x)
    mae=mean_absolute_error(y,pred);
    print("MAE:"+str(mae))


# In[48]:

def randomForestRegressionAlgorithm(datadfTraining,datadfTesting):
    label=datadfTraining.OriginalInterestRate
    datadfTraining.drop('OriginalInterestRate',axis=1,inplace=True)
    features=datadfTraining
    labelTesting=datadfTesting.OriginalInterestRate
    datadfTesting.drop('OriginalInterestRate',axis=1,inplace=True)
    featuresTesting=datadfTesting
    print("Training Data")
    rForest=RandomForestRegressor(max_depth=8)
    rForest.fit(features,label)
    computeMae(rForest,label,features)
    computeRMSE(rForest,label,features)
    computeMape(rForest,label,features)
    print("Testing Data")
    computeMae(rForest,labelTesting,featuresTesting)
    computeRMSE(rForest,labelTesting,featuresTesting)
    computeMape(rForest,labelTesting,featuresTesting)
    plt.scatter(rForest.predict(features),rForest.predict(features)-label,c='r',s=40,alpha=0.5)
    plt.scatter(rForest.predict(featuresTesting),rForest.predict(featuresTesting)-labelTesting,c="b",s=40)
    plt.hlines(y=0,xmin=2,xmax=10)
    plt.title('Residual plot using training(blue) and test(green) data')
    plt.ylabel('Residuals')
    #plt.show()


# In[49]:

def computeRMSE(model_rmse,y,x):
    model= model_rmse
    pred=model.predict(x)
    rmse=math.sqrt(mean_squared_error(y,pred))
    print("RMSE:"+str(rmse))


# In[50]:

def computeMape(model_mape,y,x):
    model= model_mape
    pred=model.predict(x)
    mape=np.mean(np.abs((y - pred) / y)) * 100
    print( "MAPE:"+str(mape))


# In[ ]:

def main():
    creds=createCredentialData("parekh.kh@husky.neu.edu","UkQqsHbV")
    getFilesFromFreddieMac(creds)
    print("2007 Analysis")
    files=preProcessData("Q1","2007","Q2","2007")
    dropColumns(files[0])
    dropColumns(files[1])
    randomForestRegressionAlgorithm(files[0]._get_numeric_data(),files[1]._get_numeric_data())
    filesOne=preProcessData("Q2","2007","Q3","2007")
    dropColumns(filesOne[0])
    dropColumns(filesOne[1])
    randomForestRegressionAlgorithm(filesOne[0]._get_numeric_data(),filesOne[1]._get_numeric_data())
    filesTwo=preProcessData("Q3","2007","Q4","2007")
    dropColumns(filesTwo[0])
    dropColumns(filesTwo[1])
    randomForestRegressionAlgorithm(filesTwo[0]._get_numeric_data(),filesTwo[1]._get_numeric_data())
    filesThree=preProcessData("Q4","2007","Q1","2007")
    dropColumns(filesThree[0])
    dropColumns(filesThree[1])
    randomForestRegressionAlgorithm(filesThree[0]._get_numeric_data(),filesThree[1]._get_numeric_data())
    
    print("2009 Analysis")
    filesFour=preProcessData("Q1","2009","Q2","2009")
    dropColumns(filesFour[0])
    dropColumns(filesFour[1])
    randomForestRegressionAlgorithm(filesFour[0]._get_numeric_data(),filesFour[1]._get_numeric_data())
    filesFive=preProcessData("Q2","2009","Q1","2010")
    dropColumns(filesFive[0])
    dropColumns(filesFive[1])
    randomForestRegressionAlgorithm(filesFive[0]._get_numeric_data(),filesFive[1]._get_numeric_data())

    
if __name__ == '__main__':
    main() 


# In[ ]:



