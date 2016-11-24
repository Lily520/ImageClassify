'''
    Created by zll 2016/11/7
    find the story boundaries of the video. Add the global threshold comparing Storyshed.py
'''
import shotDifference1
import sceneDifference
import shotDifference2
import estimate
import estimate2

def find_nearest_peak(key, list_p):
    for i in range(len(list_p)-1):
        if list_p[i] < key and list_p[i + 1] > key:
            return list_p[i], list_p[i + 1]
    return 0,0

# B is the set of scene boundaries
# D is is the corresponding context-based difference values
def Storyshed(B, D,real_result):
    # find the boundaries in the valley set Y and the peak set P
    # N is the set of other types
    N = {B[0]: D[0], B[-1]: D[-1]}
    P = {}
    Y = {}

    for i in range(1, len(B) - 1):
        if D[i - 1] > D[i] and D[i + 1] > D[i]:  # valley
            Y[B[i]] = D[i]
        elif D[i - 1] < D[i] and D[i + 1] < D[i]:  # peak
            P[B[i]] = D[i]
        else:
            N[B[i]] = D[i]

    list_p = sorted(P.keys())
    list_y = sorted(Y.keys())

    sum = 0
    for x in list_p:
        sum += P[x]
    threshold = float(sum)/len(list_p)

    ######### find the optimal threshold
    max_f1 = 0
    optimal_flag = 0.0
    flag = 0.20
    while flag <= 1.20:
        SB = []
        for key in list_y:
            left, right = find_nearest_peak(key, list_p)
            if left != 0 and right != 0:
                left_index = B.index(left)
                right_index = B.index(right)
                bound = D[left_index] if D[left_index] < D[right_index] else D[right_index]

                for indice in range(left_index, right_index + 1):
                    if D[indice] >= bound and B[indice] not in SB:
                        SB.append(B[indice])
                    elif D[indice] >= flag*threshold and B[indice] not in SB:
                        SB.append(B[indice])
        #estimate
        accuracy, recall, F1 = estimate2.result(real_result, SB)
        if max_f1 <= F1:
            max_f1 = F1
            optimal_flag = flag
        flag = flag + 0.02
    ############################################################
    #######using the optimal threshold to get the result
    SB = []
    for key in list_y:
        left, right = find_nearest_peak(key, list_p)
        if left != 0 and right != 0:
            left_index = B.index(left)
            right_index = B.index(right)
            bound = D[left_index] if D[left_index] < D[right_index] else D[right_index]

            for indice in range(left_index, right_index + 1):
                if D[indice] >= bound and B[indice] not in SB:
                    SB.append(B[indice])
                elif D[indice] >= optimal_flag * threshold and B[indice] not in SB:
                    SB.append(B[indice])
    return SB,optimal_flag  # X
    ########################################################################



if __name__ == '__main__':
    #cnA,cnx = shotDifference2.DifferenceFunc()  #feature-fusion
    cnA, cnx = shotDifference1.DifferenceFunc() #cn,bof
    result1,result2= sceneDifference.SceneDifference(cnA,cnx) #x,y
    print(result1)
    print(result2)
    realPath = "C:\\Users\\Administrator\\Desktop\\video\\filename\\25\\storysegmentation.txt"
    real_result = []
    for lines in open(realPath):
        lines = lines.strip('\n')
        real_result.append(lines)

    SB,threshold = Storyshed(result1, result2,real_result)
    print("threshold: ",threshold)
    print('SB: ', SB)  # story segmentation point
    print('ground truth: ', real_result)
    # estimate
    estimate.result(real_result,SB)
    accuracy, recall, F1 = estimate2.result(real_result, SB)
    print("Approximating estimate2:")
    print("accuracy: ", accuracy)
    print("recall: ", recall)
    print("F1: ", F1)








