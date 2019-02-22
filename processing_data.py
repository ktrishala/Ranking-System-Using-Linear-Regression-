import numpy as np

def import_data(file_path):
    data_target = []
    data_input = []
    with open(file_path, 'r') as f:
        while True:
            # Splitting our dataset per colums based on the space separator
            data_str = f.readline().split(' ')
            
            #to stop reading the file once it has reached the last line
            if not data_str[0]:
                break
            # Target is the first column
            data_target.append(float(data_str[0]))
            # Features are columns from 2 to 48, where we take only the values
            #Features are splited at the ':' and only the second part of the split is taken as that contains the float 
            # feature values. 
            data_input.append([float(a.split(':')[1]) for a in data_str[2:48]])
    
    #the feature inputs are converted into a numpy array
    x = np.array(data_input)
    #the target values are converted into Numpy array as  an one dimensional vector.
    y = np.array(data_target).reshape((-1, 1))
    #rounding of the target values
    t=np.around(y)
    #normalising the feature values between 0 and 1
    xmin=np.amin(x)
    xmax=np.amax(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i][j]=(x[i][j]-xmin)/(xmax-xmin)
 
    return x, t

filepath=r'Querylevelnorm.txt'
x, y = import_data(filepath)
print("X shape is", x.shape, "\nt shape is", y.shape)

np.savetxt("RawInputData.csv", x, fmt='%1.5f', delimiter=",")
np.savetxt("RawTargetData.csv", y, fmt='%1.0f', delimiter=",")
