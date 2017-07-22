
# coding: utf-8

# In[ ]:

## Importing the Libraries, Package 


# In[99]:

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
import sys
print('Step1. Import Done ')


# In[100]:

fileDir = os.path.dirname(os.path.realpath('__file__'))
print('Step2. Local Path defined')
print(fileDir)


# In[101]:

baseUrl='https://freddiemac.embs.com/FLoan/'
postUrl='Data/download.php'

def createCredentialData(user, passwd):
    creds={'username': user,'password': passwd}
    return creds

def getFiles(cred):
    ## We are using inside WITH BLock so that session is closed ASAP with BLock is exited 
    with requests.Session() as s:
        ## Step 1 routing to auth.php Site with the proper crentials 
        urlOne = s.post(baseUrl+"secure/auth.php", data=cred) 
        if "Please log in" in urlOne.text:
        ## IF CREDENTIALS are not valid Throw Alert 
            print("Alert: Invalid Credentials, Please try again or sign up on below site \n https://freddiemac.embs.com/FLoan/Bin/loginrequest.php")
        else:
            print("Step1: Logged in")
        ## Step 2 Preparing the data for to Accept terms and Conditions 
            pay2={'accept': 'Yes','acceptSubmit':'Continue','action':'acceptTandC'}
            finalUrl=s.post(baseUrl +"Data/download.php",pay2)
            if "Loan-Level Dataset" in finalUrl.text:
                      print("Step2 : Terms and Conditions Accepted")
                      soup = BeautifulSoup(finalUrl.content, "html.parser")   
                      links_list = soup.findAll('a')
                      print("Step3: Filtered the Sample Files with Condition > 2005")
                      print("Status::::::::::")
                      for ele in links_list:
        ## Filtering the ZIp files >= 2005 
                         if 'sample' in ele.get_text():
                            if(ele.get_text()[-8:-4] >= '2005'):
                                    print(ele.get_text()[-8:-4])
                                    tempUrl = baseUrl+"Data/"+ele.get('href')                         
                                    b =time.time()
                                    downloadUrl=s.post(tempUrl) ## return type = Response
                                    e=time.time()
                                    print(tempUrl + " took "+ str(e-b)+" sec")
                                    with ZipFile(BytesIO(downloadUrl.content)) as zfile:
                                          zfile.extractall(os.path.join(fileDir, 'adsDataRepo/'+'Sample_data_'+ele.get_text()[-8:-4]+'/'))
                                          print("File "+ ele.get_text()+" Downloaded")
    
            else:
                print("Alert: Please Check the rerouting action suffix")
        
        ##To scrape the data from the Site finalUrl.       


    
    
 


# In[102]:

def originalDatacleaning(dataf):

    df1=dataf.isnull().sum().reset_index()
    df1.columns = ['column_name', 'missing_count']
    df1 = df1.loc[df1['missing_count']>0]
    c=df1.sort_values(by='missing_count',ascending=False)
    c['missing_count%']=c['missing_count']/len(dataf)*100
    print(c)
    ## As of now in  the dataset we didn't see any value that are blank ,but we would fill the null value with the lowest Score ,
    ## in case it shows up 
    dataf['CreditScore'].replace('',600,inplace=True)
    dataf['CreditScore'].replace('   ',600,inplace=True)
    dataf['CreditScore']=dataf['CreditScore'].astype(int)
    
    print("Total Null Values Present in Column Credit Score "+ str(dataf['CreditScore'].isnull().sum()))
    
    ## There are no case of missing Values in FirstPayment Date
    print("Total Null Values Present in Column FirstPayment Date "+ str(dataf['FirstPaymentDate'].isnull().sum()))
    
    ## Plotting the graph to vizualise 
    missinganalysis(dataf)
    
    upbFF = {'dataa':[]}
    for i,v in dataf.FirstTimeHomeBuyerFlag.iteritems():
        if(pd.isnull(dataf.FirstTimeHomeBuyerFlag[i])):
            upbFF['dataa'].append(dataf.OriginalUPB[i])
    vf=pd.DataFrame(upbFF)
    print('**************************************************************************')
    print('Statistics of the Corresponding Original UPB for missing First Timers :')
    print(vf['dataa'].describe())
    print('****************************************************************************')
    
    
    for i,v in dataf.FirstTimeHomeBuyerFlag.iteritems():
        if(pd.isnull(dataf.FirstTimeHomeBuyerFlag[i])):
              if(dataf.OriginalUPB[i]>vf.quantile(.75)[0]):
                    dataf.FirstTimeHomeBuyerFlag.fillna('N',inplace=True)
              else:
                    dataf.FirstTimeHomeBuyerFlag.fillna('Y',inplace=True)
                    
    print("Total Null Values Present in Column First Time Home Buyer Flag "+ str(dataf['FirstTimeHomeBuyerFlag'].isnull().sum()))
    
    dataf['MSA']=dataf['MSA'].fillna(9999)
  
    print("Total Null Values Present in Column MSA "+ str(dataf['FirstTimeHomeBuyerFlag'].isnull().sum()))
    ##ORIGINAL DEBT-TO-INCOME (DTI) RATIO - Disclosure of the debt to income ratio
    dataf['DTI'].replace('',0,inplace=True)
    dataf['DTI'].replace('   ',0,inplace=True)
    dataf['DTI']=dataf['DTI'].astype(int)
    dataf['DTI']=dataf['DTI'].fillna(66)
    
    print("Total Null Values Present in Column DTI "+ str(dataf['DTI'].isnull().sum()))
    
    dataf['NumberOfBorrowers'].fillna(dataf['NumberOfBorrowers'].mean(),inplace=True)
    
    dataf['OriginalInterestRate'].fillna(dataf['OriginalInterestRate'].cummin(skipna=True))
    
    dataf['OCLTV'].fillna(201,inplace=True)
    print("Total Null Values Present in Column OCLTV "+ str(dataf['OCLTV'].isnull().sum()))
    
    dataf['OLTV'].fillna(106,inplace=True)
    
    dataf['LoanPurpose'].replace('',C,inplace=True)
    dataf['LoanPurpose'].replace('   ',C,inplace=True)
    
    dataf['PrepaymentPenaltyFlag'].fillna('N',inplace=True)
    dataf['SuperConformingFlag'].fillna('N',inplace=True)
    pc=dataf[['PostalCode']].groupby(dataf['PostalCode']).count().sort_values(by='PostalCode',ascending=False).head(1)['PostalCode'].index.values.tolist()[0]
    dataf['PostalCode'].fillna(pc,inplace=True)
    print("Total Null Values Present in Column OLTV "+ str(dataf['OLTV'].isnull().sum()))

#     print(dataf['DTI'].head())
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print("Data Cleaning Original file info :")
    print(dataf.info())
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    
    return dataf
    


# In[103]:

def missinganalysis(dataf):
    data = pd.concat([dataf['FirstTimeHomeBuyerFlag'], dataf['OriginalUPB']], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x='FirstTimeHomeBuyerFlag', y="OriginalUPB", data=data)
    fig.axis();
    plt.savefig(os.path.join(fileDir, 'adsDataRepo/UPBvsFirstTimer.png'))
    
    


# In[104]:

def performanceDatacleaning(dataf):
#     print(len(dataf))
#     df1=dataf.isnull().sum().reset_index()
#     df1.columns = ['column_name', 'missing_count']
#     df1 = df1.loc[df1['missing_count']>0]
#     c=df1.sort_values(by='missing_count',ascending=False)
#     c['missing_count']=c['missing_count']/len(dataf)*100
#     print(c)
    '''As we can see we have the below Null Values presnt in the Data for all the Years (Only varying the Counts )
                    MiRecoveries        3571428
                 NonMiRecoveries        3571428
           ActualLossCalculation        3571428
    DueDateOfLastPaidInstallment        3569842
        ZeroBalanceEffectiveDate        3530685
                  RepurchaseFlag        3530661
               Modification Cost        3524035'''
    
    
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
    
    
    dataf['CurrentLoadDelinquencyStatus'].replace('R',-1,inplace=True)
    dataf['CurrentLoadDelinquencyStatus'].replace('XX',-2,inplace=True)
    dataf.CurrentLoadDelinquencyStatus=dataf.CurrentLoadDelinquencyStatus.astype(int)
    
    
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print("Data Cleaning of the Performance file info :")
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
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
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    
    return dataf
    
    
    


# In[118]:

syears = [syear for syear in os.listdir(fileDir+'/adsDataRepo/')]
len(syears)
for syear in syears:
    print(syear)


# In[124]:

def dataFiltering():
    syears = [syear for syear in os.listdir(fileDir+'/adsDataRepo/')]          
    for syear in syears:
        ## Checking only the files we want , have to discard .ipynb files 
        if ('Sample_data' in syear):
            print('#####################################################################################################')
            print(syear)
            originalf = glob.glob(fileDir+'/adsDataRepo/'+syear+'/*_orig_*.txt')
            performancef= glob.glob(fileDir+'/adsDataRepo/'+syear+'/*_svcg_*.txt')
            
            headerNames1 = ['CreditScore','FirstPaymentDate','FirstTimeHomeBuyerFlag','MaturityDate','MSA','MIP','NumberOfUnits',
                           'OccupancyStatus','OCLTV','DTI','OriginalUPB','OLTV','OriginalInterestRate','Channel','PrepaymentPenaltyFlag',
                           'ProductType','PropertyState','PropertyType','PostalCode','LoanSequenceNumber','LoanPurpose',
                           'OriginalLoanTerm','NumberOfBorrowers','SellerName','ServicerName','SuperConformingFlag']
            headerNames2 = ['LoanSequenceNumber','MonthlyReportingPeriod','CurrentActualUpb','CurrentLoadDelinquencyStatus',
                            'LoanAge','RemainingMonthsToLegalMaturity','RepurchaseFlag','ModificationFlag','ZeroBalanceCode',
                            'ZeroBalanceEffectiveDate','CurrentInterestRate','CurrentDeferredUpb','DueDateOfLastPaidInstallment',
                            'MiRecoveries','NetSalesProceeds','NonMiRecoveries','Expenses','LegalCosts',
                            'MaintenanceAndPreservationCosts','TaxesAndInsurance','MiscellaneousExpenses','ActualLossCalculation',
                            'Modification Cost']
            
            
            with open(originalf[0]) as f:
                  ## Reading the data from .txt file 
                  dataf1 = pd.read_table(f, sep='|', low_memory=False, header=None,lineterminator='\n', names= headerNames1,
                                         dtype={'CreditScore':int,'OCLTV': str,'OLTV': str,'DTI': str,'CreditScore': str, 'PostalCode': str,
                                                'SuperConformingFlag' : str})
                  ## Calling Functions to preprocess it   
                  cleandata1 = originalDatacleaning(dataf1)
                  ## Saving the clean file in the csv format
                  cleandata1.to_csv(fileDir+'/adsDataRepo/'+syear+"/Original_Clean_"+syear+".csv")
                  print("Clean DOriginal Data CSV Created")
                    
            with open(performancef[0]) as f:
                ## Reading the data from .txt file 
                  dataf2 = pd.read_table(f, sep='|', low_memory=False,header=None,lineterminator='\n',names= headerNames2,
                                                 dtype={'ZeroBalanceCode':str, 'CurrentLoadDelinquencyStatus':str, 
                                                             'ModificationFlag':str,'NetSalesProceeds':str, 'LegalCosts':str, 
                                                             'MaintenanceAndPreservationCosts':str, 'TaxesAndInsurance':str, 
                                                             'Expenses':str, 'MiscellaneousExpenses':str })
                  ## Calling Functions to preprocess it 
                  cleanperf1 = performanceDatacleaning(dataf2)
                  ## Saving the clean file in the csv format
                  cleanperf1.to_csv(fileDir+'/adsDataRepo/'+syear+"/Performance_Clean_"+syear+".csv")
                  print("Clean Performance Data CSV Created")
            print('#####################################################################################################')
            print('#####################################################################################################')
                  
 


# In[125]:

print("Step3. All Functions Defined")


# In[126]:

def main():
    
    creds=createCredentialData("parekh.kh@husky.neu.edu","UkQqsHbV")
    if len(sys.argv) == 4 and sys.argv[3] in ['0','1']:
        username = sys.argv[1]
        password = sys.argv[2]
        creds=createCredentialData(username,password)
        getFiles(creds)
        dataFiltering()
    
if __name__ == '__main__':
    print("Step1.Calling Main function")
        ## Making sure that adsDataRepo be present in the local System , where the data gets downloaded 
    if(not os.path.exists(fileDir+'/adsDataRepo/')):
        os.makedirs(fileDir+'/adsDataRepo/')
    main()   


# In[ ]:



