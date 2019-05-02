from pyspark import SparkContext
from pyspark.mllib.linalg.distributed import CoordinateMatrix
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark import sql
import numpy as np
from pyspark.mllib.linalg import Matrices
from pyspark.mllib.linalg.distributed import BlockMatrix

def shape(obj):
	m = obj.numRows()  # 6
	n = obj.numCols() 
	return m,n

def bks(obj):
	m = obj.colsPerBlock()  # 6
	n = obj.rowsPerBlock() 
	return m,n


conf = SparkConf().setAppName("myFirstApp").setMaster("local")
sc = SparkContext(conf=conf)
sqlContext = sql.SQLContext(sc)

ipt = np.load('orginal.npy')
n, m = ipt.shape
n = int(n)
m = int(m)
tt = ipt.flatten()

def giveblocks(ipt):
    blocks = []
    index_1 = 0
    for i in range(0,n,n//2):
        index_2 = 0
        for j in range(0,m,m//2):
            arr = ipt[i:(i+n//2),j:(j+m//2)]
            blocks.append(((index_1,index_2),Matrices.dense(n//2 ,m//2, arr.flatten('F'))))
            index_2+=1
        index_1+=1
    return blocks

blocks = giveblocks(ipt)


def LUDecompose(mat):
    global blocks
    P = sc.parallelize(np.eye(n))
    P = P.toLocalMatrix()
    Pblocks = giveblocks(P)
    P = BlockMatrix(Pblocks,0,0)
    # indexof = (lambda x,y: [i for i, e in enumerate(x) if e==y ])
    mat = mat.toLocalMatrix()
    for i in range(n):
        maxi = 0
        maxindex = 0
        for j in range(i,n):
            # pairs = mat.entries.map{case MatrixEntry(i, j, v) => ((i, j), v)}.partitionBy(new HashPartitioner(n))
            maxi = max(maxi,mat[j,i])
            if(maxi == mat[j,i]):
                maxindex = j
        j = i+ maxindex
        addP = np.zeros((n,n))
        addP[i,i] = -1
        addP[j,j] = -1
        addP[i,j] = 1
        addP[j,i] = 1
        addP = sc.parallelize(addP)
        addP = addP.toLocalMatrix()
        addPblocks = giveblocks(addP)
        addP = BlockMatrix(addPblocks,0,0)
        swap = P + addP
        print(type(swap))
        P = swap.multiply(P)
        mat = swap.multiply(mat)
        for j in range(i,n):
            mat[j,i] = mat[j,i]/mat[i,i]
        for j in range(i,n):
            for k in range(i,n):
                mat[j,k] = mat[j,k] - mat[j,i]*mat[i,k]
    return (mat,)


blocks = sc.parallelize(blocks)


		

#blocks = sc.parallelize([((0, 0), Matrices.dense(n1, m1, tt))])

# Create a BlockMatrix from an RDD of sub-matrix blocks.
mat = BlockMatrix(blocks, n//2, m//2)


# m an RDD of sub-matrix blocks.

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
# start = time.
u = LUDecompose(mat)
print ('\n\n\n')
# v = find_eigen((mat.transpose()).multiply(mat))

