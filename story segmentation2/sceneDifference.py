'''
    Created by zll 2016/11/10
    plot the difference curve of the scene using shotDifference
'''
import shotDifference1


def SceneDifference(cnA,cnx):
    ScenePath = "C:\\Users\\Administrator\\Desktop\\video\\filename\\25\\scene.txt"
    result1 = [] #x
    result2 = [] #y
    for lines in open(ScenePath):
        lines = lines.strip('\n')
        linelist = lines.split(' ')
        try:
            start = float(linelist[0])
            end = float(linelist[1])
            start_index = cnA.index(start)
            end_index = cnA.index(end)
        except Exception as e:
            print(e, '---', lines)
        sum = 0
        count = 0
        max = 0
        for i in range(start_index,end_index+1):
            sum += cnx[i]
            count += 1
            if cnx[i] > max:
                max = i
        ave = sum/count
        result1.append(max)
        result2.append(ave)
    return result1,result2


if __name__ == '__main__':
    cnA, cnx = shotDifference1.DifferenceFunc()
    print(cnA)
    result1,result2 = SceneDifference(cnA,cnx)
    print(result1)
    print(result2)

