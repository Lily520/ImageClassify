'''
    Created by zll 2016/11/10
    plot the context-based difference curve of the shot using cn feature or bof feature.
'''
import numpy as np
import matplotlib.pyplot as plt

def DifferenceFunc():
    #cn   feature:1050 columns  shot:1 column
    #********************************************************************
    #********************************************************************
    cnshotpath = "C:\\Users\\Administrator\\Desktop\\video\\filename\\25\\shot.txt"   #cn
    cnfpath = "C:\\Users\\Administrator\\Desktop\\video\\filename\\25\\cn.txt"   #cn
    #cnshotpath = "C:\\Users\\Administrator\\Desktop\\video\\filename\\10\\bofshot.txt"   #bof
    #cnfpath = "C:\\Users\\Administrator\\Desktop\\video\\filename\\10\\bof.txt"     #bof
    file1 = open(cnshotpath)
    file2 = open(cnfpath)

    cnshot = file1.readlines()
    cnfeature = file2.readlines()

    num1 =len(cnshot) #row

    cnshotMat = np.zeros((num1,1))

    #*******************************************
    cnfeatureMat = np.zeros((num1,1050))  #cn
    #cnfeatureMat = np.zeros((num1, 500))  #bof

    #cnshotMat
    index = 0
    for line in cnshot:
        line = line.strip('\n')
        linelist = line.split(',')
        cnshotMat[index,:] = linelist[:]
        index += 1

    #cnfeatureMat
    index = 0
    for line in cnfeature:
        line = line.strip('\n')
        linelist = line.split(',')
        cnfeatureMat[index,:] = linelist[:]
        index += 1

    cn = np.column_stack((cnshotMat,cnfeatureMat)) #add column to list
    cn2 = sorted(cn ,key = lambda x:x[0]) # sort by column 1
    cn3 = np.array(cn2) #list to array
    cnA = [x[0] for x in cn3] #list
    cnB = [x[1:] for x in cn3] #list
    cnB = np.array(cnB)

    cnDist = np.sqrt(np.dot((cnB ** 2),(np.ones((cnB.T).shape))) + np.dot((np.ones(cnB.shape)),((cnB.T) ** 2))- 2 * np.dot(cnB,(cnB.T)))
    row,col =cnDist.shape #row and column of array
    for i in range(row):
        if i == row-1:
            cnB[i,2]=cnDist[i,i-1]
        else:
            cnB[i,2]=cnDist[i,i+1]

    cnx = cnB[:,2].T
    cnx = cnx.tolist()

    #plt.plot(cnx)
    #plt.show()
    return cnA,cnx


if __name__ == '__main__':
    cnA,cnx = DifferenceFunc()
    print(cnA)
    print(cnx)



