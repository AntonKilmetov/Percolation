import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from numpy import *
L = 8# линейный размер

def mark(field): # метод меток Хошена-Копельмана из ЛР №8
    markedField = zeros([L,L])
    np = []
    marker = 0
    for i in range(L):
        for j in range(L):
            markedField[i, j] = -1
    for i in range(L-1,-1,-1):
        for j in range(L):
            if (i == L - 1 and j == 0):
                if (field[i, j] == 1):
                    markedField[i, j] = marker
                    np.append(-1)
                    marker += 1
            elif (i == L - 1):
                if (field[i, j] == 1):
                    if (field[i, j - 1] == 1):
                        markedField[i, j] = markedField[i, j - 1]
                    else:
                        markedField[i, j] = marker
                        np.append(-1)
                        marker += 1
            elif (j == 0):
                if (field[i, j]==1):
                    if (field[i + 1, j]==1):
                        markedField[i, j] = markedField[i + 1, j]
                    else:
                        markedField[i, j] = marker
                        np.append(-1)
                        marker+=1
            else:
                if (field[i, j]==1):
                    if (field[i + 1, j]==1 and field[i, j - 1]==1):
                        if (markedField[i + 1, j] == markedField[i, j - 1]):
                            markedField[i, j] = markedField[i + 1, j]
                        else:
                            markedField[i, j] = min(markedField[i + 1, j], markedField[i, j - 1])
                            np[int(max(markedField[i + 1, j], markedField[i, j - 1]))] = min(markedField[i + 1, j], markedField[i, j - 1])
                            relabel(np, markedField)
                    elif (field[i + 1, j]==1):
                        markedField[i, j] = markedField[i + 1, j]
                    elif (field[i, j - 1]==1):
                        markedField[i, j] = markedField[i, j - 1]
                    else:
                        markedField[i, j] = marker
                        np.append(-1)
                        marker+=1
    return markedField
def relabel(np, markedField): # перемаркировка
    for i in range(len(np)-1,-1,-1):
        if (np[i] != -1):
            for j in range(L):
                for k in range(L):
                    if (markedField[j, k] == i):
                        markedField[j, k] = np[i]
            np[i] = -1
def detect_connecting_cluster(marked_field): # проверка на наличее соединяющий кластер
    up = marked_field[0]
    down = marked_field[len(marked_field[0]) - 1]
    for i in up:
        if (i in down and i != -1):
            return True 
    return False
def to_d_array(array): # преобразование одномерного массива в двумерный 
    new_array = zeros([L,L])
    k=0
    for i in range(L):
        for j in range(L):
            new_array[i,j]=array[k]
            k+=1
    return new_array

batchsize=2000
def make_batch():
    global batchsize     
    yin=zeros([batchsize,L*L])
    yout=zeros([batchsize,2])
    p1=zeros(int(batchsize/2))
    p2=zeros(int(batchsize/2))
    p1[:]=random.uniform(low=0.1,high=0.3,size=int(batchsize/2))
    p2[:]=random.uniform(low=0.7,high=0.9,size=int(batchsize/2))
    p=concatenate((p1,p2), axis=0)
    random.shuffle(p)
    for i in range(batchsize):
        for j in range(L*L):
            yin[i,j]= 1 if random.uniform(low=0,high=1)<p[i] else 0 # 2000 наборов решеток
        #new_yin = to_d_array(yin[i,:])
        #check = detect_connecting_cluster(mark(new_yin))  # проверка на наличее соединяющий кластер
        yout[i,0]= 1 if p[i]<0.5 else 0
        yout[i,1]= 0 if p[i]<0.5 else 1
    return yin,yout

from matplotlib import pyplot as plt

N=100
p=zeros([N,1])
p[:,0] = linspace(0,1,N)
yin_test=zeros([N,L*L]) # тестовый набор
for i in range(N):
        for j in range(L*L):
            yin_test[i,j]= 1 if random.uniform(low=0,high=1)<p[i,0] else 0
'''
yin,ytarget=make_batch()
net2=Sequential() # FFN
net2.add(Dense(80, input_shape=[L*L],activation='sigmoid'))
#net2.add(Dense(80,activation='sigmoid'))
net2.add(Dense(2,activation='softmax'))
net2.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

batches=400
costs=zeros(batches)
for k in range(batches):
    yin,ytarget=make_batch()
    costs[k]=net2.train_on_batch(yin,ytarget)[0]
#plt.plot(costs)
#plt.show()

#net2.fit(yin, ytarget, batch_size=64, epochs=200, validation_split=0.05)
test = zeros([N,2])
N=100
p=zeros([N,1])
p[:,0] = linspace(0,1,N)
for l in range(100):
    yin_test=zeros([N,L*L]) # тестовый набор
    for i in range(N):
        for j in range(L*L):
            yin_test[i,j]= 1 if random.uniform(low=0,high=1)<p[i,0] else 0
    yout_test=net2.predict_on_batch(yin_test)
    for j in range(N):
        for k in range(2):
            test[j,k] += yout_test[j,k]
for x in test:
    x[0] /= 1000
    x[1] /= 1000
plt.plot(p[:,0],test[:,0])
plt.plot(p[:,0],test[:,1])
plt.show()
'''
yin,ytarget=make_batch()
net3=Sequential()  # CNN
net3.add(Conv2D(80, 2, padding='same', input_shape=(L, L, 1),activation='relu'))

net3.add(Flatten())
net3.add(Dropout(0.5))

net3.add(Dense(2,activation='softmax'))
net3.summary()

new_yin = zeros((batchsize, L, L, 1), dtype=float)
yin=reshape(yin,(batchsize, L, L))
new_yin[::, ::, ::, 0] = yin[::, ::, ::]


net3.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#net3.fit(new_yin, ytarget, batch_size=32, epochs=100, validation_split=0.05)


batches=10
new_yin = zeros((batchsize, L, L, 1), dtype=float)
costs=zeros(batches)
for k in range(batches):
    yin,ytarget=make_batch() 
    yin=reshape(yin,(batchsize, L, L))
    new_yin[::, ::, ::, 0] = yin[::, ::, ::]
    costs[k]=net3.train_on_batch(new_yin,ytarget)[0]

N=100
p=zeros([N,1])
p[:,0] = linspace(0,1,N)
yin_test=zeros([N,L*L])
for i in range(N):
    for j in range(L*L):
        yin_test[i,j]= 1 if random.uniform(low=0,high=1)<p[i,0] else 0
new_yin_test = zeros((N, L, L, 1), dtype=float)
yin_test=reshape(yin_test,(N, L, L))
new_yin_test[::, ::, ::, 0] = yin_test[::, ::, ::]
yout_test=net3.predict_on_batch(new_yin_test)
plt.plot(p[:,0],yout_test[:,0])
plt.plot(p[:,0],yout_test[:,1])
plt.show()