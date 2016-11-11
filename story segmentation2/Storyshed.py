from contextBasedDifference import cnA,cnx

def find_nearest_peak(key,list_p):
	for i in range(len(list_p)):
		if list_p[i] < key and list_p[i+1] > key:
			return list_p[i],list_p[i+1]

#B is the set of scene boundaries
#D is is the corresponding context-based difference values


def Storyshed(B,D):
	#find the boundaries in the valley set Y and the peak set P
	#N is the set of other types
	N = {B[0]:D[0],B[-1]:D[-1]}
	P = {}
	Y = {}
	SB = []
	for i in range(1,len(B)-1):
		if D[i-1] > D[i] and D[i+1] > D[i]: #valley
			Y[B[i]] = D[i]
		elif D[i-1] < D[i] and D[i+1] < D[i]: # peak
			P[B[i]] = D[i]
		else:
			N[B[i]] = D[i]


	list_p = sorted(P.keys())
	list_y = sorted(Y.keys())
	for key in list_y:
		left,right = find_nearest_peak(key,list_p)
		left_index = B.index(left)
		right_index = B.index(right)
		bound = D[left_index] if D[left_index] < D[right_index] else D[right_index]
		
		for indice in range(left_index,right_index+1):
			if D[indice] >= bound and B[indice] not in SB:
				SB.append(B[indice])

	#test
	print(SB)

if __name__ == '__main__':
	b = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
 	d = [2,3,1,5,10,8,3.5,12,4]
	Storyshed(b,d)







