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

blocks = []
index_1 = 0
for i in range(0,n,n//2):
	index_2 = 0
	for j in range(0,m,m//2):
		arr = ipt[i:(i+n//2),j:(j+m//2)]
		blocks.append(((index_1,index_2),Matrices.dense(n//2 ,m//2, arr.flatten('F'))))
		index_2+=1
	index_1+=1


