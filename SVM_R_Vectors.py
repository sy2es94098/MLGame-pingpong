import pickle
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D


"""
def transformCommand(command):
    if 'RIGHT' in str(command):
       return 2
    elif 'LEFT' in str(command):
        return 1
    else:
        return 0
    pass"""


def get_PingPongData(filename):
    Frames = []
    Balls = []
    PlatformPos2 = []
    BallSpeed = []
    Blocker = []
    log = pickle.load((open(filename, 'rb')))
    for sceneInfo in log:
        Frames.append(sceneInfo['frame'])
        Balls.append([sceneInfo['ball'][0], sceneInfo['ball'][1]])
        BallSpeed.append([sceneInfo['ball_speed'][0], sceneInfo['ball_speed'][1]])
        #PlatformPos1.append([sceneInfo[platform_1P[0]], sceneInfo[platform_1P[1]]])
        PlatformPos2.append([sceneInfo['platform_2P'][0], sceneInfo['platform_2P'][1]])
        Blocker.append([sceneInfo['blocker'][0], sceneInfo['blocker'][1]])

    frame_ary = np.array(Frames)
    frame_ary = frame_ary.reshape((len(Frames), 1))
    data = np.hstack((frame_ary, Balls, BallSpeed, PlatformPos2, Blocker))#0, 1 2, 3 4, 5 6, 7 8
    return data

def readlog():
    Path = "D:\MLGame\games\pingpong\log"
    filenames = os.listdir(Path)
    table = np.arange(9)
    for file in filenames:
        filename = os.path.join(Path, file)
        File = get_PingPongData(filename)
        File = File[1::]
        table = np.vstack((table, File))
    table = table[1::]            
    
    return table
        
if __name__ == '__main__':
    #filename = path.join(path.dirname(__file__), 'pingpong_dataset.pickle')
    data = readlog()

    PlatformPos2 = data[:, 5:7]
    PlatformPos2_next = np.array(PlatformPos2[1:])
    vectors_2P = PlatformPos2_next - PlatformPos2[:-1]#用連續兩個frame的x,y相減得到2P的移動向量    
    data = np.hstack((data[1:, :], vectors_2P))#9 10

    Blocker = data[:, 7:9]
    Blocker_next = np.array(Blocker[1:])
    vectors_Blocker = Blocker_next - Blocker[:-1]#用連續兩個frame的x,y相減得到Blocker的移動向量    
    data = np.hstack((data[1:, :], vectors_Blocker))#11 12

    direction=[]

    for i in range(len(data)-1):
        if(data[i,3]>=0 and data[i,4]>=0):
            direction.append(0) #球移動方向為右上為0
        elif(data[i,3]>0 and data[i,4]<0):
            direction.append(1) #球移動方向為右下為1
        elif(data[i,3]<0 and data[i,4]>0):
            direction.append(2) #球移動方向為左上為2
        elif(data[i,3]<0 and data[i,4]<0):
            direction.append(3) #球移動方向為左下為3
    direction = np.array(direction)
    direction = direction.reshape((len(direction),1))
    
    data = np.hstack((data[1:, :], direction))#13


    mask = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] #mask
    X = data[:-1, mask]
    Y = data[1:, 1:2]

    
    x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)
    
    print(y_train.shape)
    
    from sklearn.svm import SVR     
    platform_predict_regressor = SVR(gamma=0.001, C=1.0, epsilon = 0.2)
    platform_predict_regressor.fit(x_train, y_train)
    
    y_predict = platform_predict_regressor.predict(x_test)
    print(y_predict)
    
    from sklearn.metrics import mean_squared_error
    MSE = mean_squared_error(y_test, y_predict)
    RMSE = np.sqrt(MSE)

    with open('save/SVMRegression_VectorsAndDirection.pickle', 'wb') as f:
        pickle.dump(platform_predict_regressor, f)
"""
    ax = plt.subplot(111, projection='3d')  
    ax.scatter(X[Y==0][:,1], X[Y==0][:,2], X[Y==0][:,3], c='#AE0000', alpha = 1)  
    ax.scatter(X[Y==1][:,1], X[Y==1][:,2], X[Y==1][:,3], c='#2828FF', alpha = 1)
    ax.scatter(X[Y==2][:,1], X[Y==2][:,2], X[Y==2][:,3], c='#007500', alpha = 1)
    plt.title('RegressionMSE = %f.2' % (RMSE))    
    ax.set_xlabel('Vectors_x')
    ax.set_ylabel('Vectors_y')
    ax.set_zlabel('Direction')    
               
    plt.show()"""
    

    
    


