import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
from tqdm import tqdm
from progress.bar import Bar


data = pd.read_csv('data.csv', names = ["MAXT(C)", "MINT", "Rhmax", "RHMIN", "WS", "Rs", "ET"])
X=data.drop('ET',axis=1)
# y=data['ET']
y = np.asarray(data['ET'], dtype="|S6")
# print (data.dtypes)
# X=X.astype('float')
# y=y.astype('float')
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()

# # Fit only to the training data
scaler.fit(X_train)

StandardScaler(copy=True, with_mean=True, with_std=True)

# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

bar = Bar('Processing', max=20)
mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
mlp.fit(X_train,y_train)
bar.finish()
predictions = mlp.predict(X_test)
predictions_train = mlp.predict(X_train)

outfile1 = open('./ann_result.txt', 'w')
outfile2 = open('./expected.txt', 'w')
outfile3 = open('./prediction.txt', 'w')

#outputfile 1
# print("classification_report\n\n", file=outfile1, end='\n')
# print(classification_report(y_test,predictions), file=outfile1, end='\n')


# #outputfile 2

# # print("X train ", file=outfile2, end='\n')
# # for i in X_train:
# # 	print(i, file=outfile2, end='\n')

# # print("\n\n\n\n\n\n\n\n\n\n", file=outfile2, end='\n'	)
# print("y train ", file=outfile2, end='\n')
# for i in y_train:
# 	print(i, file=outfile2, end='\n')
# print("\n\n\n\n\n\n\n\n\n\n", file=outfile2, end='\n'	)
# # print("X test ", file=outfile2, end='\n')
# # for i in X_test:
# # 	print(i, file=outfile2, end='\n')
# # print("\n\n\n\n\n\n\n\n\n\n", file=outfile2, end='\n'	)
# print("y test ", file=outfile2, end='\n')
# for i in y_test:
# 	print(i, file=outfile2, end='\n')


#outputfile 3

print("y predictions from ANN train ", file=outfile3, end='\n')
for i in predictions_train:
	print(i, file=outfile3, end='\n')
print("\n\n\n\n\n\n\n\n\n\n", file=outfile3, end='\n'	)

print("y predictions from ANN test ", file=outfile3, end='\n')
for i in predictions:
	print(i, file=outfile3, end='\n')
