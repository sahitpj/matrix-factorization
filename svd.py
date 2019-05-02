from pyspark import SparkContext
from pyspark.mllib.linalg.distributed import CoordinateMatrix
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark import sql
import numpy as np
import time
from pyspark.mllib.linalg import Matrices
from pyspark.mllib.linalg.distributed import BlockMatrix

###array



def shape(obj):
	m = obj.numRows()  # 6
	n = obj.numCols()  
	return m,n

def bks(obj):
	m = obj.colsPerBlock()  # 6
	n = obj.rowsPerBlock() 
	return m,n


def find_eigen(mat):
	global blocks

	#print (mat.multiply(vector).toLocalMatrix())
	eigen_vectors = []

	for j in range(m):
		vector = BlockMatrix(blocks,m//2,1)

		for i in range(20):
			#print (i)
			vector = mat.multiply(vector)
		numpy_arr = np.copy(vector.toLocalMatrix().toArray())
		#eigen_value = np.min(np.abs(numpy_arr))
		val =  np.linalg.norm(numpy_arr)
		numpy_arr = numpy_arr/val
		#print (np.min(np.abs(numpy_arr)))
		# print (numpy_arr/np.min(np.abs(numpy_arr)))
		eigen_vectors.append(numpy_arr/np.min(np.abs(numpy_arr)))
		blocks = sc.parallelize([((0, 0), Matrices.dense(m//2, 1, numpy_arr[:m//2].flatten())),
					 ((1, 0), Matrices.dense(m//2, 1, numpy_arr[m//2:].flatten()))])
		#print (numpy_arr[:m//2].flatten().shape)
		#print (numpy_arr[m//2:].flatten().shape)
		vector = BlockMatrix(blocks,m//2,1)
		# print (shape(vector))
		#print (shape(vector))
		proj = (mat.multiply(vector)).multiply(vector.transpose())
		#print (proj.toLocalMatrix())
		mat = mat.subtract(proj)
		#print (vector.toLocalMatrix())

	return eigen_vectors
	


	
conf = SparkConf().setAppName("myFirstApp").setMaster("local")
sc = SparkContext(conf=conf)
sqlContext = sql.SQLContext(sc)





# Create an RDD of sub-matrix blocks.
#print (n,m)
#print (ipt)

ipt=np.load('orginal.npy')
n,m=ipt.shape
n = int(n)
m = int(m)
tt=ipt.flatten()



blocks = []

index_1 = 0
for i in range(0,n,n//2):
	index_2 = 0
	for j in range(0,m,m//2):

		arr = ipt[i:(i+n//2),j:(j+m//2)]
		blocks.append(((index_1,index_2),Matrices.dense(n//2 ,m//2, arr.flatten('F'))))
		index_2+=1

	index_1+=1

#print (blocks)

blocks = sc.parallelize(blocks)


		

#blocks = sc.parallelize([((0, 0), Matrices.dense(n1, m1, tt))])

# Create a BlockMatrix from an RDD of sub-matrix blocks.
mat = BlockMatrix(blocks, n//2, m//2)


#localMat = mat1.toLocalMatrix()

#print (mat.toLocalMatrix())


#nnn=[]
#for i in range(m1):
#	nnn.append(((0, i), Matrices.dense(n1, 1, [1]*n1)))
#blocks = sc.parallelize(nnn)


#mat2 = BlockMatrix(blocks, n1, 1)



#mat = mat1.multiply(mat2)
#print ()
# Get its size.


#m = mat1.numRows()  # 6
#n = mat1.numCols()  # 2

#print (m,n)



#m = mat2.numRows()  # 6
#n = mat2.numCols()  # 2

#print (m,n)


#print (mat1.multiply(mat2).collect())



# Create a BlockMatrix from an RDD of sub-matrix blocks.

blocks = []


blocks.append(((0,0),Matrices.dense(m//2,1,np.ones((m//2)).flatten())))
blocks.append(((1,0),Matrices.dense(m//2,1,np.ones((m//2)).flatten())))		

blocks = sc.parallelize(blocks)

vector = BlockMatrix(blocks,m//2,1)

#res = mat1.multiply(vector)

#res2 = mat1.multiply(res)

#print (vector.toLocalMatrix())
#print (shape(vector),shape(mat))
#print (bks(vector),bks(mat))
#print (res.toLocalMatrix())
#print (vector.toLocalMatrix())

u = find_eigen(mat.multiply(mat.transpose()))
print ('\n\n\n')
v = find_eigen((mat.transpose()).multiply(mat))


'''
#print (mat.multiply(vector).toLocalMatrix())
eigen_vectors = []

for j in range(m):
	vector = BlockMatrix(blocks,m//2,1)

	for i in range(100):
		#print (i)
		vector = mat.multiply(vector)
	numpy_arr = np.copy(vector.toLocalMatrix().toArray())
	#eigen_value = np.min(np.abs(numpy_arr))
	val =  np.linalg.norm(numpy_arr)
	numpy_arr = numpy_arr/val
	#print (np.min(np.abs(numpy_arr)))
	print (numpy_arr/np.min(np.abs(numpy_arr)))
	eigen_vectors.append(numpy_arr/np.min(np.abs(numpy_arr)))
	blocks = sc.parallelize([((6, 0), Matrices.dense(m//2, 1, numpy_arr[:m//2].flatten())),
				 ((1, 0), Matrices.dense(m//2, 1, numpy_arr[m//2:].flatten()))])
	#print (numpy_arr[:m//2].flatten().shape)
	#print (numpy_arr[m//2:].flatten().shape)
	vector = BlockMatrix(blocks,m//2,1)
	#print (shape(vector))
	proj = (mat.multiply(vector)).multiply(vector.transpose())
	#print (proj.toLocalMatrix())
	mat = mat.subtract(proj)
	#print (vector.toLocalMatrix())
	
#print (shape(res),shape(mat1),shape(vector))


#print (bks(res),bks(mat1),bks(vector))

#res2 = 
'''

'''
for i in range(100):
	res = mat1.multiply(vector)
	print (res.numRows(),res..numCols())
	vector = res
	

'''

#localMat = res.toLocalMatrix()

#arr = res.toLocalMatrix().toArray()

#print (arr/np.min(np.abs(arr)))
#print (localMat)


#print (mat.multiply(mat))

'''
# Get the blocks as an RDD of sub-matrix blocks.
blocksRDD = mat.blocks

# Convert to a LocalMatrix.
localMat = mat.toLocalMatrix()

# Convert to an IndexedRowMatrix.
indexedRowMat = mat.toIndexedRowMatrix()

# Convert to a CoordinateMatrix.
coordinateMat = mat.toCoordinateMatrix()'''

u = np.reshape(np.array(u), (ipt.shape))
v = np.reshape(np.array(v), (ipt.shape))

print(ipt.shape)

print(u.shape, v.shape)