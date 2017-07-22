
# coding: utf-8

# # Summarization and EDA 

# ## Importing the Libraries, Package 

# In[1]:

import pandas as pd
import numpy as np
import os 
import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt


# ### Defining the functions 

# In[10]:

def analysePerformanceData(dataf):
#     dataf['CurrentLoadDelinquencyStatus'].replace('R',-1,inplace=True)
#     dataf['CurrentLoadDelinquencyStatus'].replace('XX',-2,inplace=True)
    
# #     print(dataf.CurrentLoadDelinquencyStatus.unique())
       
#     dataf.CurrentLoadDelinquencyStatus=dataf.CurrentLoadDelinquencyStatus.astype(int)
    
    dlq=dataf.groupby('LoanSequenceNumber')['CurrentLoadDelinquencyStatus'].max()
    ir=dataf.groupby('LoanSequenceNumber')['CurrentInterestRate'].mean()
    loanDuration=dataf.groupby('LoanSequenceNumber')['LoanAge'].max()
    loss=dataf.groupby('LoanSequenceNumber')['ActualLossCalculation'].sum()
    UPBstart=dataf.groupby('LoanSequenceNumber')['CurrentActualUpb'].max()
    UPBend=dataf.groupby('LoanSequenceNumber')['CurrentActualUpb'].min()
    modCost=dataf.groupby('LoanSequenceNumber')['Modification Cost'].sum()
    remaininMonth=dataf.groupby('LoanSequenceNumber')['RemainingMonthsToLegalMaturity'].min()
    reportstart=dataf.groupby('LoanSequenceNumber')['MonthlyReportingPeriod'].min()
    reportend=dataf.groupby('LoanSequenceNumber')['MonthlyReportingPeriod'].max()
    zeroCode=dataf.groupby('LoanSequenceNumber')['ZeroBalanceCode'].max()
    
    
    summaryPerformance=pd.DataFrame({'LoanSequenceNumber':dlq.index,'DelinquencyCount':dlq.values,'InterestRate':ir.values,
                                    'ActualLossCalculation':loss.values,'loanDuration':loanDuration.values,
                                    'UPBstart':UPBstart.values,'UPBend':UPBend.values,'zeroCode':zeroCode.values,
                                    'RemainingMonthsToLegalMaturity':remaininMonth.values,'Modification Cost':modCost.values,
                                    'reportstart':reportstart.values,'reportend':reportend.values})
#     print(summaryPerformance.head())
    
    return summaryPerformance
#     DO not use ###################
#     if(dataf.LoanSequenceNumber == df.LoanSequenceNumber.shift()):
#         if(dataf.CurrentLoadDelinquencyStatus )
######################################
# ''' We have not taken in account of the below Columns : 
#     as we know Actual Loss = (Default UPB – Net Sale_Proceeds) + DelinquentAccrued Interest 
#                                                                - Expenses – MI Recoveries – Non MIRecoveries
#     ANd Delinquent Accrued Interest = (Default_Upb – Non Interestbearing UPB)* (Current Interest rate – 0.35) * 
#                                       ( Months betweenLast Principal & Interest paid to date and zero balance date ) *30/360/100
#     So We have all the DAta information realted to Loss that are included in the ActualLossCalculation Column. 
#     SO we are wontbe considering the below Columns:
    
#     Expenses = Sum(Legal Costs, Maintenance and Preservation Costs, Taxes and Insurance,MiscellaneousExpenses)
    
#     NetSalesProceeds
#     MiRecoveries
#     NonMiRecoveries
#     Legal Costs
#     Maintenance and Preservation Costs
#     Taxes and Insurance
#     MiscellaneousExpenses
    
#     WE are not accounting the information for the MonthlyReportingPeriod Column as we think
#     Starting date of Zero Balance Date is same data as that of ending MonthlyReportingPeriod , 
#     as when the loan ends and no records are kept after that
    
#     ZeroBalanceEffectiveDate
    
#     CurrentDeferredUpb - This is the Non-interest bearing UPB , which is used to caclulate DAI , and inturn the Actual Loss 
    
#     DueDateOfLastPaidInstallment - WE are already taking consideration by taking Column RemainingMonthsToLegalMaturity, 
#                                    in our case both represent the same entity
    
    
#     RepurchaseFlag - AS this will be indicated on the Zero FLag , how the Loan has ended by Code 6 
#     ModificationFlag - This is related to the mortgages with loan modifications,indicates that the loan has been modified 
#                        and it happens only when the loan has been repurchased , which have taken care of as mentioned before.
                
    

# '''


# ########3 Not working optimally #################       
#     for i,v in dataf.CurrentLoadDelinquencyStatus.iteritems():
#         if (i==0):
#             if(dataf['CurrentLoadDelinquencyStatus'][i] > 0):
#                 count=count+1
#             continue
#         if(dataf.LoanSequenceNumber[i]==dataf.LoanSequenceNumber[i-1]):
#             if(dataf['CurrentLoadDelinquencyStatus'][i] > 0):
#                 count=count+1
#         elif(dataf.LoanSequenceNumber[i]!=dataf.LoanSequenceNumber[i-1]):
#              dataf['DeliquencyCount'][i-1]=count
#     print(dataf.head(590))
######################################################
    
                        
        
        


# ### Call the defined function 

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
# 
# 
# <p> WE are not accounting the information for the ZeroBalanceEffectiveDate Column as we saw
#     Starting date of Zero Balance Effective Date is same data as that of ending   MonthlyReportingPeriod ,  as when the loan ends and no records are kept after that .</p>
#     
#  <p>   CurrentDeferredUpb - This is the Non-interest bearing UPB , which is used to caclulate DAI ,   and inturn the Actual Loss 
#      DueDateOfLastPaidInstallment - WE are already taking consideration by taking Column RemainingMonthsToLegalMaturity, in our case both represent the same entity</p>
#   <p>  RepurchaseFlag - AS this will be indicated on the Zero FLag , how the Loan has ended by Code 6 </p>
#    <p> ModificationFlag - This is related to the mortgages with loan modifications,indicates that the loan has been modified and it happens only when the loan has been repurchased , which have taken care of as mentioned before.
#     </p>

# In[15]:

fileDir=os.path.dirname(os.path.realpath('__file__'))
print("Current Directory : "+str(fileDir))
syears = [syear for syear in os.listdir(fileDir+'/adsDataRepo/')]
print("Number of Files Present in the adsDataRepo"+str(len(syears)))
temp=[]
tempscore={'mean':[],"min":[],"max":[]}

frametempFico=[]

 
frames=[]
data200789=[]
count=0
for syear in syears:
        ## Checking only the files we want , have to discard .ipynb files 
        
    if('Sample'in syear):
          print("#####################################################3")
          print(syear)
          print("Original File ") 
          originalf = pd.read_csv(fileDir+"/adsDataRepo/"+syear+"/Original_Clean_"+syear+".csv",low_memory=False)
          print("Imported CSV")
          if("2007" in syear):
            print("Hello"+syear)
            data200789.append(originalf)
          if("2008" in syear):
            print("Hello"+syear)
            data200789.append(originalf)
          if("2009" in syear):
            print("Hello"+syear)
            data200789.append(originalf)
          
          temp.append(originalf['CreditScore'])
          tempscore['mean'].append(originalf['CreditScore'].mean())
          tempscore['min'].append(originalf['CreditScore'].min())
          tempscore['max'].append(originalf['CreditScore'].max())
        
        ## Binning the data according to the FICO Score 
          bins = [300,580, 670, 740, 800,900]
          group_names = ['POOR','BLW', 'MED', 'VG', 'EXL']
          originalf['FicoScore'] = pd.cut(originalf['CreditScore'], bins, labels=group_names)
#           for i in originalf['Year']:
#                 print(str(i)[:-2]+" "+str(i)[-2:])

          originalf['Year']=originalf['FirstPaymentDate']/100
          originalf['Year']=originalf['Year'].astype(int)
          originalf['Month']=originalf['FirstPaymentDate']%100
          originalf['Quarter']=pd.to_datetime(originalf['Month'],format= '%m')
          originalf['Quarter']=originalf['Quarter'].dt.quarter
        
          
          
#           fico= originalf.groupby('Year')['FicoScore'].sum()
          frametempFico.append(originalf)
          
          print("-------------------------------------------------------------------------------")
          print("Performance File ") 
          var1 = syear[-4:]
          ## Reading the file 
          performancef = pd.read_csv(fileDir+"/adsDataRepo/"+syear+"/Performance_Clean_"+syear+".csv",low_memory=False)
            # Passing the file to take the important parameters 
          data2 = analysePerformanceData(performancef)
           # Evaluating the  Data reduced % 
          data2['Year']=var1
          len1 = float(len(performancef)-len(data2))
          len2 = float(len(performancef))
          result = len1/len2*100
          print("Compressing the Performance Dataset by "+str(result)+"%")
          frames.append(data2)
           
print("End of Summarising ")


# ## Interest Rate - Quarterly Distribution - 2007-2008-2009

# In[16]:

origdata789 = pd.concat(data200789)


# In[17]:

data = pd.concat([origdata789['Quarter'], origdata789['OriginalInterestRate']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='Quarter', y="OriginalInterestRate", data=data)
fig.axis()
plt.show()


# ## Total Number of Loan SActioned - Quarterly 2007-2008-2009

# In[18]:

origdata789.groupby('Quarter')['LoanSequenceNumber'].count()


# ## First Time Buyer - Quartely 

# In[19]:

origdata789.groupby(['Quarter','FirstTimeHomeBuyerFlag'])['LoanSequenceNumber'].count()


# ## Quartely Channel of Mortgage Loans Sanctioned 
# <p>R = Retail</p>
# <p>B = Broker</p>
# <p>C = Correspondent</p>
# <p>T = TPO Not Specified</p>
# 

# In[20]:

ax=origdata789.groupby(['Quarter','Channel'])['LoanSequenceNumber'].count().reset_index(name="Count")


# In[21]:

frames1=[]
tempdic={"B":[]}
tempdic2={"C":[]}
tempdic3={"R":[]}
tempdic4={"T":[]}

for i,v in ax.Count.iteritems():
            if(ax.Channel[i]=='B'):
                tempdic["B"].append(ax['Count'][i])
            elif(ax.Channel[i]=='C'):
                tempdic2["C"].append(ax['Count'][i])
            elif(ax.Channel[i]=='R'):
                tempdic3["R"].append(ax['Count'][i])
            elif(ax.Channel[i]=='T'):
                tempdic4["T"].append(ax['Count'][i])
tempdicdf=pd.DataFrame(tempdic)
tempdicdf2=pd.DataFrame(tempdic2)
tempdicdf3=pd.DataFrame(tempdic3)
tempdicdf4=pd.DataFrame(tempdic4)


# In[22]:

a=pd.DataFrame({'B':tempdicdf['B'],"C":tempdicdf2["C"],"R":tempdicdf3["R"],"T":tempdicdf4["T"]})
a.index=['Q1','Q2','Q3','Q4']


# In[23]:

get_ipython().magic(u'matplotlib inline')
plt.rcParams['figure.figsize'] = (20,8)
a.plot.bar();
plt.show()


# In[24]:

fico = pd.concat(frametempFico)


# ## Summary of the Different Type of Loan Purpose 

# In[25]:

fico.columns


# In[26]:

zone = pd.DataFrame({"PostalCode":fico['PostalCode'],"State":fico['PropertyState'],"LoanSeqNo":fico['LoanSequenceNumber']})


# In[27]:

groupedZone = zone.groupby('State')['LoanSeqNo'].count()
groupZdf = pd.DataFrame(groupedZone)
groupZdf.sort_values(by="LoanSeqNo",ascending=False).head()


# In[28]:

groupZdf['LoanSeqNo']=groupZdf['LoanSeqNo'].astype(float)


# In[29]:

# !pip install plotly


# In[30]:

import plotly 
plotly.tools.set_credentials_file(username='maiti.t', api_key='elAkJU6TNeOR4Qcx3Wfd')


# In[31]:

import plotly.plotly as py
import pandas as pd

# scl = [[0, 'zmin'],[20000, 'zmin'],[30000, 'zmin'],
#             [40000, 'zmin'],[50000, 'zmin'],[100000, 'zmin']]
data = [ dict(
        type='choropleth',
        colorscale = 'Jet',
        reversescale='True',
        locations = groupZdf.index,
        z = groupZdf['LoanSeqNo'],
        locationmode = 'USA-states',
#         text = df['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Millions USD")
        ) ]

layout = dict(title = 'Freedie Mac - Mortagage Loan distribution - 2005 -2016',
              geo = dict(scope='usa',projection=dict( type='albers usa' )))
    
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='d3-cloropleth-map' )


# In[32]:

groupZdf.columns


# In[33]:

### ORIGINAL COMBINED LOAN-TO-VALUE (CLTV),ORIGINAL DEBT-TO-INCOME (DTI) RATIO -
### in  Time Series 


# In[34]:

fico['DTI'].replace('',0,inplace=True)
fico['DTI'].replace('   ',0,inplace=True)
fico['DTI']=fico['DTI'].astype(int)


# In[35]:

dc= fico.groupby('Year')['OCLTV'].mean()
dco= fico.groupby('Year')['OLTV'].mean()
dcdti= fico.groupby('Year')['DTI'].mean()
from bokeh.io import output_notebook
from bokeh.plotting  import figure,show
p=figure(width=500,height=500)
p.line(dc.index,dc.values,color="red",alpha=0.5,line_width=2,legend="OCLTV")
p.line(dco.index,dco.values,color="orange",alpha=0.5,line_width=2,legend="OLTV")
p.line(dcdti.index,dcdti.values,color="blue",alpha=0.5,line_width=2,legend="DTI")
p.xaxis.axis_label="TimeLine"
p.yaxis.axis_label="Mortagage OCLTV, OLTV,DTI "

output_notebook()
show(p)


# ## Summary according to the Property Type 
# <p> CO = Condo</p>
# <p> LH = Leasehold</p>
# <p>PU = PUD</p>
# <p> MH = Manufactured Housing</p>
# <p>SF = 1-4 Fee Simple</p>
# <p> CP = Co-op</p>
# 

# In[36]:

fico.groupby('PropertyType').describe()


# In[37]:

lp = fico.groupby(['Year','LoanPurpose'])['LoanSequenceNumber'].count().reset_index(name='Count')
lp.index=lp.Year
del lp['Year']


# ## Count Varinace of Different Types of the Loan Purpose 

# <p>P = Purchase</p>
# <p>C = Cash-outRefinance</p>
# <p>N = No Cash-outRefinance</p>

# In[38]:

data = pd.concat([lp['LoanPurpose'], lp['Count']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='LoanPurpose', y="Count", data=data)
fig.axis()
plt.show()


# In[39]:

fico2 = fico.groupby(['Year','FicoScore'])['LoanSequenceNumber'].count().reset_index(name='Count')


# ### Converting the Dataframe to Plot 

# In[40]:

frames1=[]
tempdic={"BLW":[]}
tempdic2={"EXL":[]}
tempdic3={"MED":[]}
tempdic4={"POOR":[]}
tempdic5={"VG":[]}


count=1
blwflag=True
exlflag=True
medflag=True
poorflag=True
vgflag=True
for i,v in fico2.Count.iteritems():
    try:
        if(fico2.Year[i]==fico2.Year[i+1]):
            if(fico2.FicoScore[i]=='BLW'):
                tempdic["BLW"].append(fico2['Count'][i])
                blwflag=False
            elif(fico2.FicoScore[i]=='EXL'):
                tempdic2["EXL"].append(fico2['Count'][i])
                exlflag=False
            elif(fico2.FicoScore[i]=='MED'):
                tempdic3["MED"].append(fico2['Count'][i])
                medflag=False
            elif(fico2.FicoScore[i]=='POOR'):
                tempdic4["POOR"].append(fico2['Count'][i])
                poorflag=False
            elif(fico2.FicoScore[i]=='VG'):
                tempdic5["VG"].append(fico2['Count'][i])
                vgflag=False

        elif(fico2.Year[i]!=fico2.Year[i+1]):
            if(fico2.FicoScore[i]=='BLW'):
                tempdic["BLW"].append(fico2['Count'][i])
                blwflag=False
            elif(fico2.FicoScore[i]=='EXL'):
                tempdic2["EXL"].append(fico2['Count'][i])
                exlflag=False
            elif(fico2.FicoScore[i]=='MED'):
                tempdic3["MED"].append(fico2['Count'][i])
                medflag=False
            elif(fico2.FicoScore[i]=='POOR'):
                tempdic4["POOR"].append(fico2['Count'][i])
                poorflag=False
            elif(fico2.FicoScore[i]=='VG'):
                tempdic5["VG"].append(fico2['Count'][i])
                vgflag=False
            
            if(blwflag):
                tempdic["BLW"].append(0)
            if(exlflag):
                tempdic2["EXL"].append(0)
            if(medflag):
                tempdic3["MED"].append(0)
            if(poorflag):
                tempdic4["POOR"].append(0)
            if(vgflag):
                tempdic5["VG"].append(0)


        elif(i==58):
                if(fico2.FicoScore[i]=='BLW'):
                    tempdic["BLW"].append(fico2['Count'][i])
                    blwflag=False
                elif(fico2.FicoScore[i]=='EXL'):
                    tempdic2["EXL"].append(fico2['Count'][i])
                    exlflag=False
                elif(fico2.FicoScore[i]=='MED'):
                    tempdic3["MED"].append(fico2['Count'][i])
                    medflag=False
                elif(fico2.FicoScore[i]=='POOR'):
                    tempdic4["POOR"].append(fico2['Count'][i])
                    poorflag=False
                elif(fico2.FicoScore[i]=='VG'):
                    tempdic5["VG"].append(fico2['Count'][i])
                    vgflag=False


                if(blwflag):
                    tempdic["BLW"].append("0")
                if(exlflag):
                    tempdic2["EXL"].append("0")
                if(medflag):
                    tempdic3["MED"].append(0)
                if(poorflag):
                    tempdic4["POOR"].append("0")
                if(vgflag):
                    tempdic5["VG"].append("0")
    except:
        print("")
#         countdf = pd.DataFrame(tempdic)
#         print(countdf)
#         frames1.append(countdf)
#         tempdic=[]
       
    
        
    


# In[41]:

tempdicdf=pd.DataFrame(tempdic)
tempdicdf2=pd.DataFrame(tempdic2)
tempdicdf3=pd.DataFrame(tempdic3)
tempdicdf4=pd.DataFrame(tempdic4)
tempdicdf5=pd.DataFrame(tempdic5)


# In[42]:

a= pd.DataFrame({'BLW':tempdicdf.BLW,"EXL":tempdicdf2.EXL,"MED":tempdicdf3.MED,"POOR":tempdicdf4.POOR,"VG":tempdicdf5.VG})
a.index=['2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017']


# In[43]:

get_ipython().magic(u'matplotlib inline')
plt.rcParams['figure.figsize'] = (20,8)
a.plot.bar();
plt.show()


# In[44]:

fico1 = fico.groupby('ServicerName').count()


# In[45]:

fico1= fico1.sort_values(by='Year',ascending=False)


# In[46]:

df2=pd.DataFrame({'ServiceName':fico1.index,'Count':fico1.Year})


# In[47]:

get_ipython().magic(u'matplotlib inline')
plt.rcParams['figure.figsize'] = (20,8)
df2.plot.bar();
plt.show()


# In[48]:

fico['LoanSequenceNumber'].describe()


# ## Summary of Original data 

# ### 1. Plot of Credit Score variance with the Time Series 

# In[49]:

plt.figure(figsize=(20, 6))
sns.distplot(temp[0],kde_kws={"label":"2005","alpha":0.5,"lw": 2,"color":"y"});
sns.distplot(temp[1],kde_kws={"label":"2006","alpha":0.5,"lw": 2,"color":"m"});
sns.distplot(temp[2],kde_kws={"label":"2007","alpha":0.5,"lw": 2,"color":"c"});
sns.distplot(temp[3],kde_kws={"label":"2008","alpha":0.5,"lw": 2,"color":"r"});
sns.distplot(temp[4],kde_kws={"label":"2009","alpha":0.5,"lw": 2,"color":"g"});
sns.distplot(temp[5],kde_kws={"label":"2010","alpha":0.5,"lw": 2,"color":"b"});
sns.distplot(temp[6],kde_kws={"label":"2011","alpha":0.5,"lw": 2,"color":"pink"});
sns.distplot(temp[7],kde_kws={"label":"2012","alpha":0.5,"lw": 2,"color":"brown"});
sns.distplot(temp[8],kde_kws={"label":"2013","alpha":0.5,"lw": 2,"color":"cyan"});
sns.distplot(temp[9],kde_kws={"label":"2014","alpha":0.5,"lw": 2,"color":"magenta"});
sns.distplot(temp[10],kde_kws={"label":"2015","alpha":0.5,"lw": 2,"color":"purple"});
sns.distplot(temp[11],kde_kws={"label":"2016","alpha":0.5,"lw": 2,"color":"#111111"});

plt.show()
plt.savefig(os.path.join(fileDir, 'adsDataRepo/CreditScoreVariance.png'))


# # Summary of Performance Data

# In[50]:

tempPerformance=pd.concat(frames)


# In[51]:

tempPerformance.head()


# In[52]:

tempPerformance['ActualLossCalculation'].describe()


# In[53]:

dc= tempPerformance.groupby('Year')['ActualLossCalculation'].sum()


# In[54]:

dc


# In[55]:

from bokeh.io import output_notebook
from bokeh.plotting  import figure,show
p=figure(width=500,height=500)
p.line(dc.index,dc.values,color="red",alpha=0.5,line_width=2)
p.xaxis.axis_label="TimeLine"
p.yaxis.axis_label="Mortagage Actual Loss  "
output_notebook()
show(p)


# ## Loan Duration with Time Series 

# In[56]:

data = pd.concat([tempPerformance['loanDuration'], tempPerformance['Year']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='Year', y="loanDuration", data=data)
fig.axis()
plt.show()


# In[57]:

tempPerformance.DelinquencyCount.unique()


# ## Loan Average Loan AMount in TIme Series Variance 

# In[58]:

dc= tempPerformance.groupby('Year')['UPBstart'].mean()


# In[59]:

from bokeh.io import output_notebook
from bokeh.plotting  import figure,show
p=figure(width=500,height=500)
p.line(dc.index,dc.values,color="red",alpha=0.5,line_width=2)
 
 
p.xaxis.axis_label="TimeLine"
p.yaxis.axis_label="Mortagage Average Loan AMount  "
output_notebook()
show(p)


# ## Interest Delinquency Count VS Time Series 

# In[60]:

tempPerformance.DelinquencyCount = tempPerformance.DelinquencyCount.map(lambda x: 1 if x > 0 else 0 )


# In[61]:

dc= tempPerformance.groupby('Year')['DelinquencyCount'].sum()
di= tempPerformance.groupby('Year')['InterestRate'].mean()


# In[62]:

from bokeh.io import output_notebook
from bokeh.plotting  import figure,show
p=figure(width=500,height=500)
p.line(dc.index,dc.values,color="red",alpha=0.5,line_width=2,legend="Delinquency Count")
p.line(di.index,di.values*1000,color="blue",alpha=0.5,line_width=2,legend="Interest Rate")
# p.line(a["DATE"],a["HOURLYDewPointTempC"],color="Green",alpha=0.5)
p.xaxis.axis_label="TimeLine"
p.yaxis.axis_label="Mortagage Delinquency Count and Interest Rate  "
output_notebook()
show(p)


# ## Interest Rate Variance - TIme Series 

# In[ ]:

framesInsight2 = []
d2= tempPerformance.groupby('Year')['DelinquencyCount'].sum()
InterestInsight=pd.DataFrame({'Year':d2.index,'Min Interest':d2.values,'Mean Interest':d1.values,'Max Interest':d3.values})
InterestInsight


# In[337]:

framesInsight1 = []
d2= tempPerformance.groupby('Year')['InterestRate'].min()
d1= tempPerformance.groupby('Year')['InterestRate'].mean()
d3= tempPerformance.groupby('Year')['InterestRate'].max()
InterestInsight=pd.DataFrame({'Year':d2.index,'Min Interest':d2.values,'Mean Interest':d1.values,'Max Interest':d3.values})
InterestInsight


# In[348]:

from bokeh.io import output_notebook
from bokeh.plotting  import figure,show
p=figure(width=500,height=500)
p.line(InterestInsight["Year"],InterestInsight["Min Interest"],color="red",alpha=0.5,line_width=2)
p.line(InterestInsight["Year"],InterestInsight["Mean Interest"],color="blue",alpha=0.5,line_width=2)
p.line(InterestInsight["Year"],InterestInsight["Max Interest"],color="green",alpha=0.5,line_width=2)
# p.line(a["DATE"],a["HOURLYDewPointTempC"],color="Green",alpha=0.5)
p.xaxis.axis_label="TimeLine"
p.yaxis.axis_label="Mortagage Interest Rate"
output_notebook()
show(p)


# In[350]:

mi = InterestInsight['Min Interest']-InterestInsight['Min Interest'].shift()
ma = InterestInsight['Max Interest']-InterestInsight['Max Interest'].shift()
me = InterestInsight['Mean Interest']-InterestInsight['Mean Interest'].shift()
InterestInsightPercentage=pd.DataFrame({'Year':InterestInsight['Year'],'Min Interest % Change':mi.values,
                                        'Mean Interest % Change':me.values,'Max Interest % Change':ma.values})
InterestInsightPercentage.fillna(0,inplace=True)


# In[351]:

InterestInsightPercentage


# In[354]:

from bokeh.io import output_notebook
from bokeh.plotting  import figure,show
p=figure(width=800,height=500)
p.line(InterestInsightPercentage["Year"],InterestInsightPercentage["Min Interest % Change"],color="red",alpha=0.5,line_width=2)
p.line(InterestInsightPercentage["Year"],InterestInsightPercentage["Mean Interest % Change"],color="blue",alpha=0.5,line_width=2)
p.line(InterestInsightPercentage["Year"],InterestInsightPercentage["Max Interest % Change"],color="green",alpha=0.5,line_width=2)
# p.line(a["DATE"],a["HOURLYDewPointTempC"],color="Green",alpha=0.5)
p.xaxis.axis_label="TimeLine"
p.yaxis.axis_label="Mortagage Interest Rate"
output_notebook()
show(p)


# In[247]:

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
tempmean=pd.DataFrame(tempscore)
plt.figure(num=None, figsize=(20, 6), dpi=80, facecolor='w', edgecolor='k')
tempmean.plot(table=True)
plt.show()



# In[146]:

##Binning of the credit Score 


# In[140]:

###Dropped this way of caluclating as it was taking too much time ....
# for i,v in originalf.CreditScore.iteritems():
#     if (originalf['CreditScore'][i]>800):
#          originalf['FicoScore'][i]='EXL'


# In[155]:




# In[ ]:



