
# coding: utf-8

# In[482]:


from sklearn.cluster import KMeans
import numpy as np
import csv
import math
#import matplotlib.pyplot
#from matplotlib import pyplot as plt


# In[483]:


maxAcc = 0.0
maxIter = 0
C_Lambda = 0.03
TrainingPercent = 80
ValidationPercent = 10
TestPercent = 10
M = 10
PHI = []
IsSynthetic = False


# In[484]:
#Calling the program to process the dataset file
import processing_data

def GetTargetVector(filePath):
    t = []
    with open(filePath, 'rU') as f:
        reader = csv.reader(f)
        for row in reader:  
            t.append(int(row[0]))       #Reading the target values to the numpy array
    #print("Raw Training Generated..")
    return t

def GenerateRawData(filePath, IsSynthetic):    
    dataMatrix = [] 
    with open(filePath, 'rU') as fi:
        reader = csv.reader(fi)
        for row in reader:
            dataRow = []
            for column in row:
                dataRow.append(float(column))
                
            dataMatrix.append(dataRow)          #storing the entire dataset in a numpy array
    
    
    if IsSynthetic == False :
        #deleting the columns whose values are zero as they won't affect the model performance
        dataMatrix = np.delete(dataMatrix, [5,6,7,8,9], axis=1) 
    dataMatrix = np.transpose(dataMatrix)     #Transposing the matrix to get a (41, 69623) matrix
    #print ("Data Matrix Generated..",np.shape(dataMatrix))
    return dataMatrix

def GenerateTrainingTarget(rawTraining,TrainingPercent = 80):
    TrainingLen = int(math.ceil(len(rawTraining)*(TrainingPercent*0.01))) #Taking only 80% of the Target data for training
    t           = rawTraining[:TrainingLen]
    #print(str(TrainingPercent) + "% Training Target Generated..")
    return t

def GenerateTrainingDataMatrix(rawData, TrainingPercent = 80):
    T_len = int(math.ceil(len(rawData[0])*0.01*TrainingPercent)) #Taking only 80% of the data matrix for training
    d2 = rawData[:,0:T_len]
    print(rawData[0:2,0:])
    #print(str(TrainingPercent) + "% Training Data Generated..")
    return d2

def GenerateValData(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData[0])*ValPercent*0.01)) #Taking only 10% of the data matrix for validation
    V_End = TrainingCount + valSize
    dataMatrix = rawData[:,TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Data Generated..")  
    return dataMatrix

def GenerateValTargetVector(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData)*ValPercent*0.01)) #Taking only 10% of the Target data for training
    V_End = TrainingCount + valSize
    t =rawData[TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Target Data Generated..")
    return t

def GenerateBigSigma(Data, MuMatrix,TrainingPercent,IsSynthetic):
    BigSigma    = np.zeros((len(Data),len(Data)))    #len(Data) will output a 41*41 matrix
    DataT       = np.transpose(Data)                 #transposing the input data to (69623, 41) matrix
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))        
    varVect     = []
    for i in range(0,len(DataT[0])):  #looping over the entire column length
        vct = []
        for j in range(0,int(TrainingLen)):   #loopng over the entire row length for 80% data
            vct.append(Data[i][j])    #storing the column values in the list
        varVect.append(np.var(vct))   #calculating the variance of the feature for all datapoints
    
    for j in range(len(Data)):      #till 41
        BigSigma[j][j] = varVect[j]   #storing all the  variances of each feature in the diagonal column of BigSigma
    if IsSynthetic == True:     
        BigSigma = np.dot(3,BigSigma)  #in case the variances are too small, this is to normalise the values as the 
    else:                               #sigma value increases it will only increase the spread of the gaussian curve
        BigSigma = np.dot(200,BigSigma)
    ##print ("BigSigma Generated..")
    return BigSigma

def GetScalar(DataRow,MuRow, BigSigInv):    #calculating the exponential part using the Gaussian radial basis function
    R = np.subtract(DataRow,MuRow)          #formula
    T = np.dot(BigSigInv,np.transpose(R))  
    L = np.dot(R,T)
    return L

def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x

def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80): #forming the design matrix
    DataT = np.transpose(Data)                         #transposing the input data to (69623, 41) matrix
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))         
    PHI = np.zeros((int(TrainingLen),len(MuMatrix)))               #creating the design matrix of size (55699, 10)
    BigSigInv = np.linalg.inv(BigSigma)
    for  C in range(0,len(MuMatrix)):   #so for each feature 10 Mu s are generation which will be the center of basis functions
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)    #Calculating phi using 
    #print ("PHI Generated..")
    return PHI

def GetWeightsClosedForm(PHI, T, Lambda):  #geting the weight matrix using the closed form solution
    Lambda_I = np.identity(len(PHI[0]))   #forming a 10x10 identity matrix for the regularization factor
    
    for i in range(0,len(PHI[0])):
        Lambda_I[i][i] = Lambda          #forming the regularization matrix
    PHI_T       = np.transpose(PHI)
    PHI_SQR     = np.dot(PHI_T,PHI)
    PHI_SQR_LI  = np.add(Lambda_I,PHI_SQR)
    PHI_SQR_INV = np.linalg.inv(PHI_SQR_LI)
    INTER       = np.dot(PHI_SQR_INV, PHI_T)
    W           = np.dot(INTER, T)
    ##print ("Training Weights Generated..")
    return W

# def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
#     DataT = np.transpose(Data)
#     TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))         
#     PHI = np.zeros((int(TrainingLen),len(MuMatrix))) 
#     BigSigInv = np.linalg.inv(BigSigma)
#     for  C in range(0,len(MuMatrix)):
#         for R in range(0,int(TrainingLen)):
#             PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
#     #print ("PHI Generated..")
#     return PHI

def GetValTest(VAL_PHI,W):     #generating the output labels for test data using the weights computed
    Y = np.dot(W,np.transpose(VAL_PHI))
    ##print ("Test Out Generated..")
    return Y

def GetErms(VAL_TEST_OUT,ValDataAct):  #calculating Erms
    sum = 0.0
    t=0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)  #Sum of squared error
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]): #rounding off the output values and comparing with actual labels
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
    ##print ("Accuracy Generated..")
    ##print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))


# ## Fetch and Prepare Dataset

# In[485]:


RawTarget = GetTargetVector('RawTargetData.csv')
RawData   = GenerateRawData('RawInputData.csv',IsSynthetic)


# ## Prepare Training Data

# In[486]:


TrainingTarget = np.array(GenerateTrainingTarget(RawTarget,TrainingPercent))
TrainingData   = GenerateTrainingDataMatrix(RawData,TrainingPercent)
print(TrainingTarget.shape)
print(TrainingData.shape)


# ## Prepare Validation Data

# In[487]:


ValDataAct = np.array(GenerateValTargetVector(RawTarget,ValidationPercent, (len(TrainingTarget))))
ValData    = GenerateValData(RawData,ValidationPercent, (len(TrainingTarget)))
print(ValDataAct.shape)
print(ValData.shape)


# ## Prepare Test Data

# In[488]:


TestDataAct = np.array(GenerateValTargetVector(RawTarget,TestPercent, (len(TrainingTarget)+len(ValDataAct))))
TestData = GenerateValData(RawData,TestPercent, (len(TrainingTarget)+len(ValDataAct)))
print(ValDataAct.shape)
print(ValData.shape)


# ## Closed Form Solution [Finding Weights using Moore- Penrose pseudo- Inverse Matrix]

# In[489]:


ErmsArr = []
AccuracyArr = []

kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingData)) #the clusters and the centroids formed are fitted onto the data points
Mu = kmeans.cluster_centers_  #getting the centroids of the 10 clusters formed to be used for generating design matrix



BigSigma     = GenerateBigSigma(RawData, Mu, TrainingPercent,IsSynthetic)
TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent)
W            = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(C_Lambda)) 
TEST_PHI     = GetPhiMatrix(TestData, Mu, BigSigma, 100) 
VAL_PHI      = GetPhiMatrix(ValData, Mu, BigSigma, 100)


# In[490]:

#########Plotting the clusters of the dataset
#X=np.transpose(TrainingData)
#labels=kmeans.labels_
#plt.figure(figsize=(8, 6))
#plt.scatter(X[:,0],X[:,1],c=labels.astype(np.float))
#plt.hold(True)
#plt.scatter(Mu[:,0],Mu[:,1],c=np.arange(M),marker='^',s=150)
#plt.show()


# In[491]:


print(Mu.shape)
print(BigSigma.shape)
print(TRAINING_PHI.shape)
print(W.shape)
print(VAL_PHI.shape)
print(TEST_PHI.shape)


# ## Finding Erms on training, validation and test set 

# In[492]:


TR_TEST_OUT  = GetValTest(TRAINING_PHI,W)
VAL_TEST_OUT = GetValTest(VAL_PHI,W)
TEST_OUT     = GetValTest(TEST_PHI,W)

TrainingAccuracy   = str(GetErms(TR_TEST_OUT,TrainingTarget))
ValidationAccuracy = str(GetErms(VAL_TEST_OUT,ValDataAct))
TestAccuracy       = str(GetErms(TEST_OUT,TestDataAct))


# In[493]:


print ('UBITname      = tkaushik')
print ('Person Number = 50289070')
print ('----------------------------------------------------')
print ("------------------LeToR Data------------------------")
print ('----------------------------------------------------')
print ("-------Closed Form with Radial Basis Function-------")
print ('----------------------------------------------------')
print ("M = 10 \nLambda = 0.9")
print ("Accuracy Training   = " + str(float(TrainingAccuracy.split(',')[0])))
print ("E_rms Training   = " + str(float(TrainingAccuracy.split(',')[1])))
print ("Accuracy validation   = " + str(float(ValidationAccuracy.split(',')[0])))
print ("E_rms Validation = " + str(float(ValidationAccuracy.split(',')[1])))
print ("Accuracy Testing   = " + str(float(TestAccuracy.split(',')[1])))
print ("E_rms Testing    = " + str(float(TestAccuracy.split(',')[1])))


# ## Gradient Descent solution for Linear Regression

# In[494]:


print ('----------------------------------------------------')
print ('--------------Please Wait for 2 mins!----------------')
print ('----------------------------------------------------')


# In[ ]:


W_Now        = np.dot(220, W)   #Initialising the weight matrox formed by closed form solution to some random values
La           = 2
LR = [0.001, 0.015, .1, .25]
L_Erms_Val   = []

L_Erms_Test  = []
W_Mat        = []

for learningRate in (LR):
    L_Erms_TR    = []
    for i in range(0,400):
        
        #print ('---------Iteration: ' + str(i) + '--------------')
        Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i])
        La_Delta_E_W  = np.dot(La,W_Now)
        Delta_E       = np.add(Delta_E_D,La_Delta_E_W)  #Computing derivative of E from the derivative of the error and the regularizer
        Delta_W       = -np.dot(learningRate,Delta_E)
        W_T_Next      = W_Now + Delta_W
        W_Now         = W_T_Next             #Updating the weights for each iteration based on loss function
    
        #-----------------TrainingData Accuracy---------------------#
        TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next)  #Generating the training labels based on updated weights
        Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget) #Getting RMS error and accuracy
        L_Erms_TR.append(float(Erms_TR.split(',')[1]))

    
        #-----------------ValidationData Accuracy---------------------#
        VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next)      #Generating the validation labels based on updated weights
        Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct) #Getting RMS error and accuracy
        L_Erms_Val.append(float(Erms_Val.split(',')[1]))

    
        #-----------------TestingData Accuracy---------------------#
        TEST_OUT      = GetValTest(TEST_PHI,W_T_Next)   #Generating the testing labels based on updated weights
        Erms_Test = GetErms(TEST_OUT,TestDataAct)     #Getting RMS error and accuracy
        L_Erms_Test.append(float(Erms_Test.split(',')[1]))

    print("For learning rate:"+str(learningRate)+" the trainin E_RMS is: "+str(np.around(min(L_Erms_TR),5)))


# In[ ]:


print ('----------Gradient Descent Solution--------------------')
#print ("M = 15 \nLambda  = 0.0001\neta=0.01")
print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))

