
# coding: utf-8

# # Part 3 Classification

# ## Import of the Packages, Libraries 

# In[1]:

# !pip install urllib
# !pip install pandas
# !pip install -U scikit-learn
# !sudo pip install Pillow
# !sudo pip install fabulous
# !sudo pip install Pillow
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
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn import svm
from sklearn import metrics
from sklearn import neural_network
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.datasets.classification import ClassificationDataSet
from pybrain.tools.validation import Validator
get_ipython().magic(u'matplotlib inline')


# ## Defining Functions 

# ## 1. Making Credentials 

# In[2]:

def createCredentialData(user, passwd):
    print('_________________________________________________________')
    print('Creating Credentials ....')
    creds={'username': user,'password': passwd}
    return creds


# ## 2.Getting Files from the Freddie Mac

# In[3]:

def getFilesFromFreddieMac(cred,quater,year,quaterTwo,yearTwo,baseUrl,postUrl,fileDir):
    print('_________________________________________________________')
    print('GettingFile from Freddie Mac ......')
    c=cred
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
                      print("Step3: Filtered the Sample Files with Condition =" + year)
                      print("Status::::::::::")
                      for ele in links_list:
        ## Filtering the ZIp files = 2005 
                         if 'historical' in ele.get_text():
                          
                            while(ele.get_text()[-8:-4] < yearTwo):
                                    print(ele.get_text()[-8:-4])
#                                     if(ele.get_text()[-10:-8] <= quaterTwo):
#                                         print(ele.get_text()[-10:-8])
                                        
                                        tempUrl = baseUrl+"Data/"+ele.get('href')                         
                                        b =time.time()
                                        downloadUrl=s.post(tempUrl) ## return type = Response
                                        e=time.time()
                                        print(tempUrl + " took "+ str(e-b)+" sec")
                                        with ZipFile(BytesIO(downloadUrl.content)) as zfile:
                                              zfile.extractall(os.path.join(fileDir, 'adsDataRepo/'+'Historical_data_'+ele.get_text()[-8:-4]+'/'))
                                              print("File "+ ele.get_text()+" Downloaded")
                                        
                            while(ele.get_text()[-8:-4] == yearTwo):
                                if(ele.get_text()[-10:-8] <= quaterTwo):
                                    tempUrl = baseUrl+"Data/"+ele.get('href')                         
                                    b =time.time()
                                    downloadUrl=s.post(tempUrl) ## return type = Response
                                    e=time.time()
                                    print(tempUrl + " took "+ str(e-b)+" sec")
                                    with ZipFile(BytesIO(downloadUrl.content)) as zfile:
                                             zfile.extractall(os.path.join(fileDir, 'adsDataRepo/'+'Historical_data_'+ele.get_text()[-8:-4]+'/'))
                                             print("File "+ ele.get_text()+" Downloaded")
                                
                                
#                                         if(quarter>quaterTwo and year > yearTwo):
#                                             print("")
#                                             year=99
#                                             quarter=99
#                                         if(quarter==4):
#                                             year=int(year)+1
#                                             quarter=1
#                                         quarter=int(quarter)+1
                                        
                                        
                            
#                             if(ele.get_text()[-8:-4] == yearTwo):
# #                                     print(ele.get_text()[-8:-4])
#                                     if(ele.get_text()[-10:-8] == quaterTwo):
# #                                         print(ele.get_text()[-10:-8])
                                        
#                                         tempUrl = baseUrl+"Data/"+ele.get('href')                         
#                                         b =time.time()
#                                         downloadUrl=s.post(tempUrl) ## return type = Response
#                                         e=time.time()
#                                         print(tempUrl + " took "+ str(e-b)+" sec")
#                                         with ZipFile(BytesIO(downloadUrl.content)) as zfile:
#                                               zfile.extractall(os.path.join(fileDir, 'adsDataRepo/'+'Historical_data_'+ele.get_text()[-8:-4]+'/'))
#                                               print("File "+ ele.get_text()+" Downloaded")
            else:
                print("Alert: Please Check the rerouting action suffix")
        
        ##To scrape the data from the Site finalUrl.       
            


# ## 3.Data  Cleaning 

# In[4]:

def performanceDatacleaning(dataf):
#     print(len(dataf))
#     df1=dataf.isnull().sum().reset_index()
#     df1.columns = ['column_name', 'missing_count']
#     df1 = df1.loc[df1['missing_count']>0]
#     c=df1.sort_values(by='missing_count',ascending=False)
#     c['missing_count']=c['missing_count']/len(dataf)*100
#     print(c)

#   If null Forward Fill or Backward Fill -  MONTHLY_REPORTING_PERIOD, as its usually apprears in sequence 
    print('_________________________________________________________')
    print('Performance Data Cleaning Started .......')
    dataf['MonthlyReportingPeriod'].fillna(method='ffill',inplace=True)
    dataf['MonthlyReportingPeriod'].fillna(method='bfill',inplace=True)
    dataf['MiRecoveries'].fillna(0,inplace=True)
    dataf['NonMiRecoveries'].fillna(0,inplace=True)
    dataf['ActualLossCalculation'].fillna(0,inplace=True)
    dataf['DueDateOfLastPaidInstallment'].fillna('NA',inplace=True)
    dataf['ZeroBalanceCode'].fillna(-1,inplace=True)
    dataf['ZeroBalanceEffectiveDate'].fillna('NA',inplace=True)
    dataf['RepurchaseFlag'].fillna('NA',inplace=True)
    dataf['Modification Cost'].fillna(0,inplace=True)
    dataf['MiscellaneousExpenses'].fillna(0,inplace=True)
    dataf['TaxesAndInsurance'].fillna(0,inplace=True)
    dataf['MaintenanceAndPreservationCosts'].fillna(0,inplace=True)
    dataf['LegalCosts'].fillna(0,inplace=True)
    dataf['Expenses'].fillna(0,inplace=True)
    
    dataf['ModificationFlag'].fillna('N',inplace=True)
    
    dataf['NetSalesProceeds'].fillna('U',inplace=True)
    
    ## Interpolation of few columns where the missing values are present 
    dataf['LoanAge']=dataf['LoanAge'].interpolate()
    dataf['RemainingMonthsToLegalMaturity']=dataf['RemainingMonthsToLegalMaturity'].interpolate()
    dataf['CurrentInterestRate']=dataf['CurrentInterestRate'].interpolate()
    
    
    
    dataf['CurrentLoadDelinquencyStatus'].replace('R',-1,inplace=True)
    dataf['CurrentLoadDelinquencyStatus'].replace('XX',-2,inplace=True)
    dataf['CurrentLoadDelinquencyStatus'].replace("   ",0,inplace=True)
    dataf['CurrentLoadDelinquencyStatus'].replace('',0,inplace=True)
    dataf.CurrentLoadDelinquencyStatus=dataf.CurrentLoadDelinquencyStatus.astype(int)
    
    print('_____________________________________________________________')
    print("Data Cleaning of the Performance file Summary  :")
   
#     print("Data Cleaning Original file info :")
#     df1=dataf.isnull().sum().reset_index()
#     df1.columns = ['column_name', 'missing_count']
#     df1 = df1.loc[df1['missing_count']>0]
#     c=df1.sort_values(by='missing_count',ascending=False)
#     c['missing_count%']=c['missing_count']/len(dataf)*100
#     print(c)

    print("Total Null Values Present in Column LoanSequenceNumber "+ str(dataf['LoanSequenceNumber'].isnull().sum()))
    print("Total Null Values Present in Column MonthlyReportingPeriod "+ str(dataf['MonthlyReportingPeriod'].isnull().sum()))
    print("Total Null Values Present in Column CurrentActualUpb "+ str(dataf['CurrentActualUpb'].isnull().sum()))
    print("Total Null Values Present in Column CurrentLoadDelinquencyStatus "+ str(dataf['CurrentLoadDelinquencyStatus'].isnull().sum()))
    print("Total Null Values Present in Column LoanAge "+ str(dataf['LoanAge'].isnull().sum()))
    print("Total Null Values Present in Column RemainingMonthsToLegalMaturity "+ str(dataf['RemainingMonthsToLegalMaturity'].isnull().sum()))
    print("Total Null Values Present in Column RepurchaseFlag "+ str(dataf['RepurchaseFlag'].isnull().sum()))
    print("Total Null Values Present in Column ModificationFlag "+ str(dataf['ModificationFlag'].isnull().sum()))
    print("Total Null Values Present in Column ZeroBalanceCode "+ str(dataf['ZeroBalanceCode'].isnull().sum()))
    print("Total Null Values Present in Column ZeroBalanceEffectiveDate "+ str(dataf['ZeroBalanceEffectiveDate'].isnull().sum()))
    print("Total Null Values Present in Column CurrentInterestRate "+ str(dataf['CurrentInterestRate'].isnull().sum()))
    print("Total Null Values Present in Column CurrentDeferredUpb "+ str(dataf['CurrentDeferredUpb'].isnull().sum()))
    print("Total Null Values Present in Column DueDateOfLastPaidInstallment "+ str(dataf['DueDateOfLastPaidInstallment'].isnull().sum()))
    print("Total Null Values Present in Column MiRecoveries "+ str(dataf['MiRecoveries'].isnull().sum()))
    print("Total Null Values Present in Column NetSalesProceeds "+ str(dataf['NetSalesProceeds'].isnull().sum()))
    print("Total Null Values Present in Column NonMiRecoveries "+ str(dataf['NonMiRecoveries'].isnull().sum()))
    print("Total Null Values Present in Column Expenses "+ str(dataf['Expenses'].isnull().sum()))
    print("Total Null Values Present in Column LegalCosts "+ str(dataf['LegalCosts'].isnull().sum()))
    print("Total Null Values Present in Column MaintenanceAndPreservationCosts "+ str(dataf['MaintenanceAndPreservationCosts'].isnull().sum()))
    print("Total Null Values Present in Column TaxesAndInsurance "+ str(dataf['TaxesAndInsurance'].isnull().sum()))
    print("Total Null Values Present in Column MiscellaneousExpenses "+ str(dataf['MiscellaneousExpenses'].isnull().sum()))
    print("Total Null Values Present in Column ActualLossCalculation "+ str(dataf['ActualLossCalculation'].isnull().sum()))
    print("Total Null Values Present in Column Modification Cost "+ str(dataf['Modification Cost'].isnull().sum()))
    print('_____________________________________________________________')
    
    
    return dataf
    
    
    


# ## 4. Data PreProcessing 

# In[5]:

def preProcessData(inputQuater,inputYear,inputQuaterTwo,inputYearTwo,fileDir):
    print('_________________________________________________________')
    print("Pre-processing the data........")
    if(os.path.exists(fileDir+'/adsDataRepo/')):
#         trainingDataFile = glob.glob(fileDir+'/adsDataRepo/'+'Historical_data_'+inputYear+'/historical_data1_'+inputQuater+inputYear+'.txt')
#         testingDataFile = glob.glob(fileDir+'/adsDataRepo/'+'Historical_data_'+inputYearTwo+'/historical_data1_'+inputQuaterTwo+inputYearTwo+'.txt')
        trainingPerformanceDataFile = glob.glob(fileDir+'/adsDataRepo/'+'Historical_data_'+inputYear+'/historical_data1_time_'+inputQuater+inputYear+'.txt')
        print('Imported Dataset1 ')
        testPerformanceDataFile = glob.glob(fileDir+'/adsDataRepo/'+'Historical_data_'+inputYearTwo+'/historical_data1_time_'+inputQuaterTwo+inputYearTwo+'.txt')
        print('Imported Dataset2 ')

        print(len(trainingPerformanceDataFile))
#         headerNames = ['CreditScore','FirstPaymentDate','FirstTimeHomeBuyerFlag','MaturityDate','MSA','MIP','NumberOfUnits',
#                          'OccupancyStatus','OCLTV','DTI','OriginalUPB','OLTV','OriginalInterestRate','Channel','PrepaymentPenaltyFlag',
#                          'ProductType','PropertyState','PropertyType','PostalCode','LoanSequenceNumber','LoanPurpose',
#                          'OriginalLoanTerm','NumberOfBorrowers','SellerName','ServicerName','SuperConformingFlag']
        headerNames2 = ['LoanSequenceNumber','MonthlyReportingPeriod','CurrentActualUpb','CurrentLoadDelinquencyStatus',
                            'LoanAge','RemainingMonthsToLegalMaturity','RepurchaseFlag','ModificationFlag','ZeroBalanceCode',
                            'ZeroBalanceEffectiveDate','CurrentInterestRate','CurrentDeferredUpb','DueDateOfLastPaidInstallment',
                            'MiRecoveries','NetSalesProceeds','NonMiRecoveries','Expenses','LegalCosts',
                            'MaintenanceAndPreservationCosts','TaxesAndInsurance','MiscellaneousExpenses','ActualLossCalculation',
                            'Modification Cost']
        
#         with open(trainingDataFile[0]) as f:
#             dataf = pd.read_table(f, sep='|', low_memory=False, header=None,lineterminator='\n', names= headerNames)
#             cleandataOne = originationDatacleaning(dataf)
#             cleandataOne.to_csv(fileDir+'/adsDataRepo/Historical_data_'+inputYear+"/Origination_Classification_"+inputQuater+inputYear+".csv",index=False)
#             print("training original data cleaned, CSV Created")
       
#         with open(testingDataFile[0]) as f:
#             dataf = pd.read_table(f, sep='|', low_memory=False, header=None,lineterminator='\n', names= headerNames)
#             cleandataTwo = originationDatacleaning(dataf)
#             cleandataTwo.to_csv(fileDir+'/adsDataRepo/Historical_data_'+inputYear+"/Origination_Classification_"+inputQuaterTwo+inputYearTwo+".csv",index=False)
#             print("testing original data cleaned, CSV Created")
            
        with open(trainingPerformanceDataFile[0]) as f:
            dataf2 = pd.read_table(f, sep='|', low_memory=False,header=None,lineterminator='\n',names= headerNames2,
                                     dtype={'ZeroBalanceCode':str, 'CurrentLoadDelinquencyStatus':str, 
                                                 'ModificationFlag':str,'NetSalesProceeds':str, 'LegalCosts':str, 
                                                 'MaintenanceAndPreservationCosts':str, 'TaxesAndInsurance':str, 
                                                 'Expenses':str, 'MiscellaneousExpenses':str })
            cleanperfTrain = performanceDatacleaning(dataf2)
            cleanperfTrain.to_csv(fileDir+'/adsDataRepo/Historical_data_'+inputYear+"/Performance_Classification_"+inputQuater+inputYear+".csv",index=False)
            print("training performance data cleaned, CSV Created")
            
        with open(testPerformanceDataFile[0]) as f:
            dataft2 = pd.read_table(f, sep='|', low_memory=False,header=None,lineterminator='\n',names= headerNames2,
                                     dtype={'ZeroBalanceCode':str, 'CurrentLoadDelinquencyStatus':str, 
                                                 'ModificationFlag':str,'NetSalesProceeds':str, 'LegalCosts':str, 
                                                 'MaintenanceAndPreservationCosts':str, 'TaxesAndInsurance':str, 
                                                 'Expenses':str, 'MiscellaneousExpenses':str })
            cleanperfTest = performanceDatacleaning(dataft2)
            cleanperfTest.to_csv(fileDir+'/adsDataRepo/Historical_data_'+inputYear+"/Performance_Classification_"+inputQuaterTwo+inputYearTwo+".csv",index=False)
            print("testing performance data cleaned, CSV Created")
            
        return cleanperfTrain,cleanperfTest


# ## 5.Main Function 

# In[6]:

def main():
    
        
    print('_________________________________________________________')
    print("Inside MainFunction .....................")
    print('_________________________________________________________')
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    print("Current Working Directory::::")
    print(fileDir)
    baseUrl='https://freddiemac.embs.com/FLoan/'
    postUrl='Data/download.php'
    username= raw_input('Please enter your Username')
    password= raw_input('Please enter your Password')
    creds=createCredentialData(username,password)
#     creds=createCredentialData("maiti.t@husky.neu.edu","3J\G\{K4")
    baseUrl='https://freddiemac.embs.com/FLoan/'
    postUrl='Data/download.php'
    inputQuater='Q1'
    inputyear='2000'
    inputTestYear='2000'
    ## Making the Directory , if not exits before 
    if(not os.path.exists(fileDir+'/adsDataRepo/')):
        os.makedirs(fileDir+'/adsDataRepo/')
    inputQuater= raw_input('Enter Quarter ')
    inputYear= raw_input('Enter year ')
    count=0
#     print(inputQuater[1:2])
    if(len(inputQuater)!=2 and len(inputYear)!=4):
        print("Please Enter Valid Input Format ")
        inputQuater= raw_input('Enter Start Quarter: ')
        inputYear= raw_input('Enter Startyear: ')
        inputQuater2= raw_input('Enter End Quarter: ')
        inputYear2= raw_input('Enter End year: ')
        count+=1
        

    
    try:
#             print(type(int(inputYear)))
        if(int(inputQuater[1:2])<=4 and int(inputQuater[1:2]) >=1 and int(inputYear) <= 2016 and int(inputYear) >=1999):
             if(int(inputQuater[1:2])<=4 and int(inputQuater[1:2]) >=1 and int(inputYear) <= 2016 and int(inputYear) >=1999):
        

#             temp = int(inputQuater[1:2])
#             if(temp==4):
#                 temp=1
#                 inputTestYear=str(int(inputYear)+1)
#             else:
#                 temp +=1
#                 inputTestYear=str(int(inputYear))
#             t=str(temp)

                    inputTestQuater=0
                    inputTestYear=0
                   
                
#                     jobs = []
#                     for ele in links_list:
#                         p = multiprocessing.Process(target=downloadFiles(ele, logger, fileDir, httpSession))
#                         jobs.append(p)
#                         p.start()
                    


                    getFilesFromFreddieMac(creds,inputQuater,inputYear,inputQuater2,inputYear2,baseUrl,postUrl,fileDir)
                    y1=inputYear
                    q1=inputQuater
                    y2=1999
                    q2=1
            
                    while(y1<inputYear2):
                            if(q1==4):
                                y2=int(y1)+1
                                q2=str(1)
                            else:
                                q2=str(int(q1)+1)
                            
                            
                            cleanperfTrain,cleanperfTest=preProcessData(q1,y1,q2,y2,fileDir)
                            datainsight(cleanperfTrain,cleanperfTest,q2,y2)
                            
                            if(q1==4):
                                y1=str(int(y1)+1)
                                q1=str(int(q1)+1)
                   
                    if(y1==inputYear2):
                            while(q1<=inputQuater2):
                                q2=str(int(q1)+1)
                                y2=str(y1)
                                cleanperfTrain,cleanperfTest=preProcessData(q1,y1,q2,y2,fileDir)
                                datainsight(cleanperfTrain,cleanperfTest,q2,y2)
                    

                            


        else:
            print("Alert: Please Enter Valid Quater Time : < = 4 and input Year 1999-2016 ")
#             year = int(inputYear)+1
            inputQuater= raw_input('Enter Quarter')
            inputYear= raw_input('Enter year')
            inputQuater2= raw_input('Enter End Quarter: ')
            inputYear2= raw_input('Enter End year: ')
            count+=1

        if(count ==5):
            print("Please restart the Script.")
            count=0
    except:
        print("Please Retry with Valid Input Format ")
    






# ## 6.Unique Value Present in the Delinquency Column 

# In[7]:

def uniqueValesDelinquency(cleanperfTrain):
    print('_________________________________________________________')
    print('Unique Values Present in the Training Dataset ')
    print('_________________________________________________________')
    print(cleanperfTrain.CurrentLoadDelinquencyStatus.unique())
    print('_________________________________________________________')


# ## 7. Transforming the Delinquency Column 

# In[8]:

def transformDelinqColumn(cleanperfTrain,cleanperfTest):
    print('_________________________________________________________')
    print("Tranforming Delinquency Columns .......")
    print('_________________________________________________________')
    cleanperfTrain['DelinquencyStatus'] = cleanperfTrain.CurrentLoadDelinquencyStatus.map(lambda x: 1 if x > 0 else 0 )
    cleanperfTest['DelinquencyStatus'] = cleanperfTest.CurrentLoadDelinquencyStatus.map(lambda x: 1 if x > 0 else 0 )
    
#     print(type(cleanperfTrain['DelinquencyStatus']))
    return cleanperfTrain,cleanperfTest


# ## 8. Factorize Categorical Data

# In[9]:

def factorizeCategoricalColumn(cleanperfTrain,cleanperfTest):
        print('_________________________________________________________')
        print('Factorizing the Categorical Columns .....................')
        print('_________________________________________________________')

        cleanperfTrain['RepurchaseFlag_Fact'] = pd.factorize(cleanperfTrain['RepurchaseFlag'])[0]
        cleanperfTrain['ModificationFlag_Fact'] = pd.factorize(cleanperfTrain['ModificationFlag'])[0]
        cleanperfTrain['ZeroBalanceCode_Fact'] = pd.factorize(cleanperfTrain['ZeroBalanceCode'])[0]
        cleanperfTrain['NetSalesProceeds_Fact'] = pd.factorize(cleanperfTrain['NetSalesProceeds'])[0]
        cleanperfTest['RepurchaseFlag_Fact'] = pd.factorize(cleanperfTest['RepurchaseFlag'])[0]
        cleanperfTest['ModificationFlag_Fact'] = pd.factorize(cleanperfTest['ModificationFlag'])[0]
        cleanperfTest['ZeroBalanceCode_Fact'] = pd.factorize(cleanperfTest['ZeroBalanceCode'])[0]
        cleanperfTest['NetSalesProceeds_Fact'] = pd.factorize(cleanperfTest['NetSalesProceeds'])[0]
        
        return cleanperfTrain,cleanperfTest


# In[10]:

# x_train = pd.DataFrame()
# x_train = cleanperfTrain[['CurrentActualUpb','LoanAge','RemainingMonthsToLegalMaturity','CurrentInterestRate',
#                           'CurrentDeferredUpb','MiRecoveries','NonMiRecoveries','Expenses','LegalCosts',
#                             'MaintenanceAndPreservationCosts','TaxesAndInsurance',
#                             'MiscellaneousExpenses','ActualLossCalculation',
#                             'Modification Cost','RepurchaseFlag_Fact','ModificationFlag_Fact',
#                           'ZeroBalanceCode_Fact','NetSalesProceeds_Fact']]
# y_train = pd.DataFrame()
# y_train['traindelinq'] = cleanperfTrain['DelinquencyStatus']


# # 9. Feature Selection 

# ## A. Recursive feature elimination with cross-validation

# <p>
# A recursive feature elimination example with automatic tuning of the number of features selected with cross-validation.
# </p>
# <p> AS we can see after Count number of 5 features the Cross Validation Score Decreases Drastically.  </p>
# <b> Note </b><p> We have taken all the columns to analyse for us in this method </p>

# In[11]:

def featureSelection(cleanperfTrain,cleanperfTest):
    print('_________________________________________________________')
    print('Feature Selection Started ...')
    print('_________________________________________________________')
    print('Currenlty Commented Out RFEDV Method  as it takes >5 hours ')
#     rfecv(cleanperfTrain)

    x_train = pd.DataFrame()
    x_train = cleanperfTrain[['CurrentActualUpb','LoanAge','RemainingMonthsToLegalMaturity','CurrentInterestRate',
                          'CurrentDeferredUpb','MiRecoveries','NonMiRecoveries','Expenses','LegalCosts',
                            'MaintenanceAndPreservationCosts','TaxesAndInsurance',
                            'MiscellaneousExpenses','ActualLossCalculation',
                            'Modification Cost','RepurchaseFlag_Fact','ModificationFlag_Fact',
                          'ZeroBalanceCode_Fact','NetSalesProceeds_Fact']]
    y_train = pd.DataFrame()
    y_train['traindelinq'] = cleanperfTrain['DelinquencyStatus']
    ranks = rfe(x_train,y_train)
    
    trainData,testData,temp=finalfeatureSelection(ranks,cleanperfTrain,cleanperfTest)
    
    logitnalysis(cleanperfTrain,trainData)
    
    return temp,trainData,testData
    
    
    
    
# svc = SVC(kernel="linear")
# # The "accuracy" scoring is proportional to the number of correct
# # classifications
# rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
#               scoring='accuracy')
# rfecv.fit(x, y)

# print("Optimal number of features : %d" % rfecv.n_features_)

# # Plot number of features VS. cross-validation scores
# plt.figure()
# plt.xlabel("Number of features selected")
# plt.ylabel("Cross validation score (nb of correct classifications)")
# plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
# plt.show()


# In[12]:

def rfecv(cleanperfTrain):
    print('_________________________________________________________')
    print('Starting Recursive feature elimination with cross-validation')
    print('_________________________________________________________')
    print('It may take a take time to do the analysis ')
    b =time.time()
    x_train = pd.DataFrame()
    sample=cleanperfTrain.sample(100)
    x_train = sample[['CurrentActualUpb','LoanAge','RemainingMonthsToLegalMaturity','CurrentInterestRate',
                          'CurrentDeferredUpb','MiRecoveries','NonMiRecoveries','Expenses','LegalCosts',
                            'MaintenanceAndPreservationCosts','TaxesAndInsurance',
                            'MiscellaneousExpenses','ActualLossCalculation',
                            'Modification Cost','RepurchaseFlag_Fact','ModificationFlag_Fact',
                          'ZeroBalanceCode_Fact','NetSalesProceeds_Fact']]
    y_train = pd.DataFrame()
    y_train['traindelinq'] = cleanperfTrain['DelinquencyStatus']
    
    y=y_train.head(100)
    y = np.ravel(y)  
    x = np.asarray(x_train.head(100))
    
    svc = SVC(kernel="linear")
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),scoring='accuracy')
    print(type(x))
    print('gggg')
    print(y)
    rfecv.fit(x,y)

    e=time.time()
    print('Total Time Taken : '+str(e-b)+ "sec")
    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    
    rfecv.ranking_
    
    ranking_out = zip(map(lambda x: round(x, 8), rfecv.ranking_), x_train.columns)
    
    print(sorted(ranking_out))
    


# ### Optimal number of features : 5  Note: Run Time : 4 hours Approximately 

# ## Result for the best Top 5 Features:
# <p>Current Interest Rate </p>
# <p> Loan Age </p>
# <p> Remaining Months TO Legal Maturity </p>
# <p> Repurchase Flag </p>
# <p>Zero Balance Code </p>
# <p>Current Actual UPb </p>
# <b> Note </b><p> Loan Age and Remaining Months TO Legal Maturity Columns represent the same notion in the data </p>

# ## B. Recursive feature elimination

# ### A recursive feature elimination example showing the relevance of pixels in a digit classification task.
# 

# In[13]:

def rfe(x_train,y_train):
    print('_________________________________________________________')
    print('Starting Recursive Feature Elimination ....................')
    print('_________________________________________________________')
    b=time.time()
    lr = LogisticRegression()
    y_train2=np.ravel(y_train)   
    rfe = RFE(estimator=lr, n_features_to_select=5, step=2)
    rfe.fit(x_train.head(1000000),y_train2[0:1000000])
    ranking = rfe.ranking_
    e=time.time()
    print("Time in Processing : "+str(e-b)+"sec")
    rfe.ranking_
    ranks = zip(map(lambda x: round(x, 4), rfe.ranking_), x_train.columns)
    ranks = sorted(ranks)
    print("Result :::")
    for v in ranks:
        print(v)
    return ranks
    
    


# In[14]:

def finalfeatureSelection(ranks,cleanperfTrain,cleanperfTest):
    print('_________________________________________________________')
    print('Final Feature Selection Process Started ................')
    print('_________________________________________________________')
    temp=[]
    for v in ranks:
        if(v[0]==1.0):
            temp.append(v[1])
    temp.append('CurrentActualUpb') ## got from Recursive feature elimination with cross-validation
    trainData = cleanperfTrain[temp]
    testData = cleanperfTest[temp]
    return trainData,testData,temp


# In[15]:

# # Create the RFE object and rank each pixel
# svc = SVC(kernel="linear", C=1)
# rfe1 = RFE(estimator=svc, n_features_to_select=1, step=2)
# rfe1.fit(x_train.head(100),y_train2[0:100])
# # ranking = rfe.ranking_.reshape(digits.images[0].shape)

# # Plot pixel ranking
# plt.matshow(ranking, cmap=plt.cm.Blues)
# plt.colorbar()
# plt.title("Ranking of pixels with RFE")
# plt.show()


# ## C.Logit Function - Analysis 

# ### We have not taken in account of the below Columns : 
# <p>As we know Actual Loss = (Default UPB – Net Sale_Proceeds) + DelinquentAccrued Interest 
#                                                               - Expenses – MI Recoveries – Non MIRecoveries</p>
# <p> And Delinquent Accrued Interest = (Default_Upb – Non Interestbearing UPB)* 
#                                                             (Current Interest rate – 0.35) *
#                                                             ( Months between Last Principal & 
#                                                               Interest paid to date and zero balance date ) *30/360/100</p>
# <p>So We have all the DAta information related to Loss that are included in the Actual Loss Calculation Column. SO we are won't be considering the below Columns:</p>
#     
# <p>Expenses = Sum(Legal Costs, Maintenance and Preservation Costs, Taxes and                                                      Insurance,Miscellaneous Expenses)</p>
# <p>    NetSalesProceeds</p>
# <p>    MiRecoveries</p>
#  <p>   NonMiRecoveries</p>
#  <p>   Legal Costs</p>
#  <p>   Maintenance and Preservation Costs</p>
#  <p>   Taxes and Insurance</p>
# <p>   MiscellaneousExpenses
#     </p>
# 

# ## Logit - Function of  Logs of Odds 

# In[16]:

def logitnalysis(cleanperfTrain,trainData):
        print('_________________________________________________________')
        print("Starting Logit Analysis ............")
        print('_________________________________________________________')
        b=time.time()
        logit = sm.Logit(np.asarray(cleanperfTrain['DelinquencyStatus'].head(1000000)),np.asarray(trainData.head(1000000)))
        logitodd = logit.fit()
        print(logitodd.summary2())
        e=time.time()
        print("Time in Processing : "+str(e-b)+"sec")
        print("The estimated coefficients are the log odds. By exponentiating these values, we can calculate the odds, which are easier to interpret.")
        print('_________________________________________________________')
        print(np.exp(logitodd.params))
        


# ## The estimated coefficients are the log odds. By exponentiating these values, we can calculate the odds, which are easier to interpret.

# ## Result : Below are are the features whose Standard Error are CLose to Zero : 
# <p> CurrentActualUpb</p>
# <p> LoanAge</p>
# <p> RemainingMonthsToLegalMaturity</p>
# <p> ActualLossCalculation</p>
# <p>Modification Cost</p>

# ## 10. Regression 

# ### We are doing regression on Dependent Variable Binary(Delinquency Status ) with the below features :
# <p>CurrentInterestRate</p>
# <p>LoanAge</p>
# <p>NetSalesProceeds_Fact</p>
# <p>RepurchaseFlag_Fact</p>
# <p>ZeroBalanceCode_Fact</p>
# <p>CurrentActualUpb</p>
# <p>ActualLossCalculation</p>
# 

# In[17]:

def regression(temp,trainData,testData,cleanperfTrain,cleanperfTest,q2,y2):
        print('_________________________________________________________')
        print("Starting Classification .............")
        print('_________________________________________________________')
        y_train2=np.ravel(cleanperfTrain['DelinquencyStatus'])
        deliq_test=np.ravel(cleanperfTest['DelinquencyStatus'])
        datainfo=float(np.unique(y_train2,return_counts=True)[1][1])/float(len(y_train2))*100
        print("Percentage of Delinquency Status Flag present in the Test Dataset "+
              str(datainfo)+str("%"))
        tempCA=[]
        tempActDelinq=[]
        tempPredDelinq=[]
        tempTotalRec=[]
        tempDelinqProp=[]
        tempNonDelinq=[]
        
        reportVar=[tempCA,tempActDelinq,tempPredDelinq,tempTotalRec,tempDelinqProp,tempNonDelinq]
        if(datainfo<25):
            print("Caution: Dataset is Imbalanced:::Predictive Modelling may not be efficient for the Delinqency Status Flag")
            bestScore=balancedDataClassification(cleanperfTrain,cleanperfTest,temp,reportVar,q2,y2)
            print("Original Dataset Clssification")
            bestScore=originalDataClassification(trainData,testData,cleanperfTrain,cleanperfTest,reportVar,q2,y2)
            
        else:
            print("Original Dataset Clssification")
            bestScore=originalDataClassification(trainData,testData,cleanperfTrain,cleanperfTest,reportVar,q2,y2)
        
        return bestScore
    
    


# ## 11. Balancing the Data

# In[18]:

def balancedDataClassification(cleanperfTrain,cleanperfTest,temp,reportVar,q2,y2):
        print('_________________________________________________________')
        print("Balancing the data")
        print('_________________________________________________________')
        databalancetemp=temp
        databalancetemp.append('DelinquencyStatus')
        y_train2=np.ravel(cleanperfTrain['DelinquencyStatus'])
        trainBalanceData = cleanperfTrain[databalancetemp]
        trainBalanceData1=trainBalanceData.loc[trainBalanceData['DelinquencyStatus'] == 0]
        trainBalanceData1=trainBalanceData1.sample(trainBalanceData['DelinquencyStatus'].sum()*3)
        
        trainBalanceData2=trainBalanceData.loc[trainBalanceData['DelinquencyStatus'] == 1]
        frame1=[trainBalanceData1,trainBalanceData2]
        a=len(trainBalanceData1)
        b=len(trainBalanceData2)
        
        print("Current Dataset insight")
        print("Non Delinquent present Percentage"+str(a/(a+b)*100))
        print("Delinquent present Percentage"+str(b/(a+b)*100))
        finaltrainBalaneData=pd.concat(frame1)
        x_trainData=finaltrainBalaneData[temp]
        y_trainData=np.ravel(finaltrainBalaneData['DelinquencyStatus'])

        testBalanceData = cleanperfTest[databalancetemp]
        testBalanceData1=testBalanceData.loc[testBalanceData['DelinquencyStatus'] == 0]
        testBalanceData1=testBalanceData1.sample(testBalanceData['DelinquencyStatus'].sum())
        testBalanceData2=testBalanceData.loc[testBalanceData['DelinquencyStatus'] == 1]
        frame2=[testBalanceData1,testBalanceData2]
        finaltestBalaneData=pd.concat(frame2)
        x_testData=finaltestBalaneData[temp]
        y_testData=np.ravel(finaltestBalaneData['DelinquencyStatus'])
        
        print("##################################################")
        print('Balanced DataSet Classification')
        tempScore=[]
        tempScore,reportVar=LogisticRegressionClassification(x_trainData,y_trainData,x_testData,y_testData,reportVar,tempScore)
        tempScore,reportVar=randomForesClassifier(x_trainData,y_trainData,x_testData,y_testData,reportVar,tempScore)
        tempScore,reportVar=neuralClssification(x_trainData,y_trainData,x_testData,y_testData,reportVar,tempScore)
        bestScore=evaluateBestModel(tempScore)
        
        createReport(reportVar,q2,y2)
        
        reportVar=[]
        tempScore=[]
        return bestScore

        


# ## 12. Original Data Classification 

# In[19]:

def originalDataClassification(trainData,testData,cleanperfTrain,cleanperfTest,reportVar,q2,y2):
    print("Starting Original  Data Classification")
    print('_________________________________________________________')
    y_train2=np.ravel(cleanperfTrain['DelinquencyStatus'])
    deliq_test=np.ravel(cleanperfTest['DelinquencyStatus'])
    tempScore=[]
    tempScore,reportVar=LogisticRegressionClassification(trainData,y_train2,testData,deliq_test,reportVar,tempScore)
    tempScore,reportVar=randomForesClassifier(trainData,y_train2,testData,deliq_test,reportVar,tempScore)
    tempScore,reportVar=neuralClssification(trainData,y_train2,testData,deliq_test,reportVar,tempScore)
    
    bestScore = evaluateBestModel(tempScore)
    createReport(reportVar,q2,y2)
    tempScore=[]
    reportVar=[]
    return bestScore
    


# ## 13. Logistic Regression Classification

# In[20]:

def LogisticRegressionClassification(trainData,y_train2,testData,deliq_test,reportVar,tempScore):
        b=time.time()
        print('Starting Logistic Regession Classification')
        print('_________________________________________________________')
        modellr = LogisticRegression()
        modellr.fit(trainData,y_train2)
        e=time.time()
        print("Time in Processing : "+str(e-b)+"sec")
        #Check the accuracy of the model
        print("Accuracy of the logistic Regression Classification Model Training Dataset: "+ str(modellr.score(trainData,y_train2)))
        
        ##Prediction og regression
#         tempScore=[]
        b=time.time()
        resultpred = modellr.predict(testData)
        probs = modellr.predict_proba(testData)[:, 1]
        # generate evaluation metrics
        scoreLogisitic = metrics.accuracy_score(deliq_test, resultpred)
        print('Accuracy of the logistic Regression Classification Model Test Dataset: ' + str(scoreLogisitic))
        e=time.time()
        print("Time in Processing : "+str(e-b)+"sec")
        
        
        ## Generating Confusion Matrix :::::::::::::::
        b=time.time()
        cm = metrics.confusion_matrix(deliq_test, resultpred)
        print('Confusion Matrix:')
        print(cm)
        e=time.time()
        print("Time in Processing : "+str(e-b)+"sec")
        
        p = cm[0][0]
        t = np.unique(deliq_test,return_counts=True)[1][0]
        r=float(p)/float(t)*100
        print("Percentage of Predicting right Non - Delinquency Status Flag   " +str(r)+str("%"))
        d=cm[1][1]
        t=np.unique(deliq_test,return_counts=True)[1][1]
        r=float(d)/float(t)*100
        print("Percentage of Predicting right Delinquency Status Flag   " +str(r)+str("%"))
        tempScore.append(r)
        
        ## Vizualisation of the Confusion Matrix 
        fix, ax = plt.subplots(figsize=(8, 6))
        plt.suptitle('Confusion Matrix  on Data Set')
        matrix = cm
        plt.title('Logistic Regression Classification on Test Data');
        sns.heatmap(matrix, annot=True,  fmt='');
        
        deliq_test = deliq_test.astype(np.float)
        fpr, tpr, _ = metrics.roc_curve(deliq_test, probs)
        ## Plotting ROC Curve ------------
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('ROC curve')
        plt.legend(loc="lower right")
        plt.show()
        
        
        reportVar[0].append('Logistic Regression') ##Classification_Algorithm
        reportVar[1].append(np.unique(deliq_test,return_counts=True)[1][1]) ## Number_of_Actual_Delinquents
        reportVar[2].append(cm[0][1]+cm[1][1])##Number_of_predicted_delinquents
        reportVar[3].append(len(deliq_test))##Number_of_records_in_dataset
        reportVar[4].append(cm[1][1])##Number_of_delinquents_properly_classified
        reportVar[5].append(cm[1][0])##Number_of_non_delinquents_improperly_classified_as_delinquents
        
        return tempScore,reportVar
        


# #### Compute confusion matrix to evaluate the accuracy of a classification
# #### By definition a confusion matrix C is such that C_{i, j} is equal to the 
# #### number of observations known to be in group i but predicted to be in group j.
# #### Thus in binary classification, the count of true negatives is C_{0,0}, false negatives is C_{1,0}, true positives is C_{1,1} and false positives is C_{0,1}.

# ## 14. Random Forest Classification

# In[21]:

def randomForesClassifier(trainData,y_train2,testData,deliq_test,reportVar,tempScore):
    print('_________________________________________________________')
    print('Random Forest Classification Started ...........')
    print('_________________________________________________________')
    b=time.time()
    model = RandomForestClassifier(n_jobs=2)
    model = model.fit(trainData,y_train2)
    e=time.time()
    print("Time in Processing : "+str(e-b)+"sec")
    
    print("Accuracy of the Random Forest Classification Model on Training Dataset : "+ str(model.score(trainData,y_train2)))
    ##Prediction 
    b=time.time()
    resultpred = model.predict(testData)
    probs = model.predict_proba(testData)[:, 1]
    # generate evaluation metrics
    scorerf = metrics.accuracy_score(deliq_test, resultpred)
    print('"Accuracy of the Random Forest Classification Model on Test Dataset:' + str(scorerf))
    e=time.time()
    print("Time in Processing : "+str(e-b)+"sec")
#     tempScore.append(scorerf)

    ## Creating Confusion Matrix 
    b=time.time()
    cm = metrics.confusion_matrix(deliq_test, resultpred)
    print('Confusion Matrix:')
    print(cm)
    e=time.time()
    print("Time in Processing : "+str(e-b)+"sec")
    
    ## visualizing the confusion Matrix 
    fix, ax = plt.subplots(figsize=(8, 6))
    plt.suptitle('Confusion Matrix  on Data Set')
    matrix = cm
    plt.title('Random Forest on Test Data');
    sns.heatmap(matrix, annot=True,  fmt='');
    plt.show()
    
    ##ROC 
    deliq_test = deliq_test.astype(np.float)
    fpr, tpr, _ = metrics.roc_curve(deliq_test, probs)
    #Plot ROC curve
    get_ipython().magic(u'matplotlib inline')
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()
    


    p = cm[0][0]
    t = np.unique(deliq_test,return_counts=True)[1][0]
    r=float(p)/float(t)*100
    print("Percentage of Predicting right Non - Delinquency Status Flag   " +str(r)+str("%"))
    d=cm[1][1]
    t=np.unique(deliq_test,return_counts=True)[1][1]
    r=float(d)/float(t)*100
    print("Percentage of Predicting right Delinquency Status Flag   " +str(r)+str("%"))
    tempScore.append(r)

    reportVar[0].append('Random Forest Classification') ##Classification_Algorithm
    reportVar[1].append(np.unique(deliq_test,return_counts=True)[1][1]) ## Number_of_Actual_Delinquents
    reportVar[2].append(cm[0][1]+cm[1][1])##Number_of_predicted_delinquents
    reportVar[3].append(len(deliq_test))##Number_of_records_in_dataset
    reportVar[4].append(cm[1][1])##Number_of_delinquents_properly_classified
    reportVar[5].append(cm[1][0])##Number_of_non_delinquents_improperly_classified_as_delinquents
    
    return tempScore,reportVar



# ## 15. Neural Network Classification 

# In[22]:

def neuralClssification(trainData,y_train2,testData,deliq_test,reportVar,tempScore):
    print('_________________________________________________________')
    print('Neural Network Classfication Started ')
    print('_________________________________________________________')
    b=time.time()
    model = neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    model = model.fit(trainData, y_train2)
    e=time.time()
    print("Time in Processing : "+str(e-b)+"sec")
    print("Accuracy of the Neural Network Classification Model on Training Dataset : "+ str(model.score(trainData,y_train2)))
    b=time.time()
    result = model.predict(testData)
    probs = model.predict_proba(testData)[:, 1]
    # generate evaluation metrics
    scorenn = metrics.accuracy_score(deliq_test, result)
    print('"Accuracy of the  Neural Network Classification Model on Test Dataset :' + str(scorenn))
    e=time.time()
    print("Time in Processing : "+str(e-b)+"sec")
#     tempScore.append(scorenn)

    b=time.time()
    cm = metrics.confusion_matrix(deliq_test, result)
    print('Confusion Matrix:')
    print(cm)
    e=time.time()
    print("Time in Processing : "+str(e-b)+"sec")

    
    fix, ax = plt.subplots(figsize=(8, 6))
    plt.suptitle('Confusion Matrix  on Data Set')
    matrix = cm
    plt.title('Neural Network on Test Data');
    sns.heatmap(matrix, annot=True,  fmt='');
    plt.show()
    
    
    deliq_test = deliq_test.astype(np.float)
    fpr, tpr, _ = metrics.roc_curve(deliq_test, probs)


    #Plot ROC curve
    get_ipython().magic(u'matplotlib inline')
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()

    p = cm[0][0]
    t = np.unique(deliq_test,return_counts=True)[1][0]
    r=float(p)/float(t)*100
    print("Percentage of Predicting right Non - Delinquency Status Flag   " +str(r)+str("%"))
    d=cm[1][1]
    t=np.unique(deliq_test,return_counts=True)[1][1]
    r=float(d)/float(t)*100
    print("Percentage of Predicting right Delinquency Status Flag   " +str(r)+str("%"))
    tempScore.append(r)

    
    reportVar[0].append('Neural Network Classification') ##Classification_Algorithm
    reportVar[1].append(np.unique(deliq_test,return_counts=True)[1][1]) ## Number_of_Actual_Delinquents
    reportVar[2].append(cm[0][1]+cm[1][1])##Number_of_predicted_delinquents
    reportVar[3].append(len(deliq_test))##Number_of_records_in_dataset
    reportVar[4].append(cm[1][1])##Number_of_delinquents_properly_classified
    reportVar[5].append(cm[1][0])##Number_of_non_delinquents_improperly_classified_as_delinquents
    
    return tempScore,reportVar


# ### 16. Evaluating the Best Model 

# In[23]:

def evaluateBestModel(tempScore):
        print('_________________________________________________________')
        print('Calulating Best Model ...')
        print('_________________________________________________________')
        bestScore=pd.DataFrame(tempScore)
        bestScore.columns=['Score']
        bestScore.index=['LogisticRegression','RandomForest','NeuralNetWork']
        bestScore=bestScore.sort_values(by='Score',ascending =False)
        print("Best Classification Algorithm : "+str(bestScore.index[0])+"   Accuracy "+str(bestScore.Score[0]))
        print("Result in Nutshell : ")
        print(bestScore)
        return bestScore
        


# ## 17.Creating report

# In[24]:

def createReport(reportVar,q2,y2):
        print('_________________________________________________________')
        print('Creating Report .....')
        print('_________________________________________________________')
        bestClassification = pd.DataFrame()
        bestClassification['Classification_Algorithm']=reportVar[0]
        bestClassification['Number_of_Actual_Delinquents']=reportVar[1]
        bestClassification['Number_of_predicted_delinquents']=reportVar[2]
        bestClassification['Number_of_records_in_dataset']=reportVar[3]
        bestClassification['Number_of_delinquents_properly_classified']=reportVar[4]
        bestClassification['Number_of_non_delinquents_improperly_classified_as_delinquents']=reportVar[5]
        print(bestClassification)
        pd.to_csv("Report"+str(q2)+str(y1))
        print('_________________________________________________________')


# ## 18. Data Insight Function Calling all the defined functions 

# In[ ]:

def datainsight(cleanperfTrain,cleanperfTest,q2,y2):
    
    print('_________________________________________________________')
    print("Shape of Training Data :"+ str(cleanperfTrain.shape))
    print('_________________________________________________________')
    print("Description of Training Data")
    print(cleanperfTrain.describe())

    uniqueValesDelinquency(cleanperfTrain)

    cleanperfTrain,cleanperfTest=transformDelinqColumn(cleanperfTrain,cleanperfTest)

    print(cleanperfTrain.DelinquencyStatus.unique())
    print('_________________________________________________________')
    print("Total Training Delinquency Flag"+str(cleanperfTrain.DelinquencyStatus.sum()))
    print('_________________________________________________________')
    print("Total TEst Delinquency Flag "+str(cleanperfTest.DelinquencyStatus.sum()))

    cleanperfTrain,cleanperfTest=factorizeCategoricalColumn(cleanperfTrain,cleanperfTest)

    # print("Columns  ")
    # print(cleanperfTrain.columns)

    temp,trainData,testData =featureSelection(cleanperfTrain,cleanperfTest)


    bestScore =regression(temp,trainData,testData,cleanperfTrain,cleanperfTest,q2,y2)
    return bestScore


# ## 19. The Driver Function 

# In[ ]:

if __name__ == '__main__':
    
    cleanperfTrain,cleanperfTest = main()  
#     bestScore=datainsight(cleanperfTrain,cleanperfTest)
#     inputuser= raw_input('Do want to Remodel Again with newer Dataset ')
#     alltrain=pd.DataFrame()
#     alltrain=pd.concat([cleanperfTrain,cleanperfTest])
#     if "Yes" or "Y" or "y" in inputuser:
#         cleanperfTrain1,cleanperfTest2 = main() 
#         alltrain=pd.concat([alltrain,cleanperfTrain1])
#         alltest=cleanperfTest2
#         bestScore1=datainsight(alltrain,alltest)
#         print(bestScore1)
#         print(bestScore)
#         inputuser= raw_input('Do want to Remodel Again with newer Dataset,which helps our Model to improve its accuracy')
#     elif "No" or "n" or "N" in inputuser:
#         print("Data Classification finished.")
    


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



