'''
Name: Sai Venkatesh Kurella
Campus ID : VR62250
E-mail: vr62250@umbc.edu


HW1: Decision tree and KNN

Before running this program install tensorflow along with python 3.X
Link for installing tensorflow is: https://www.tensorflow.org/install/

Instruction for running the program:
After going through directions of installing tensoflow. You must have installed a virtual environment

After creating virtual environment
start it by following command: source ~/(virtual_env)/bin/activate

For installing any modules like matplotlib, numpy use pip3 install <module_name>

To run program use : python3 KNN_mnist.py

'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plot


def KNN(train_data,train_label,test_data,test_label,k):
    '''
    This function is used to return accuracy vector for 10 way classification along with average accuracy.
    '''

    data_train = tf.placeholder(dtype=tf.float32,shape = [None,28*28])
    label_train = tf.placeholder(dtype= tf.float32,shape = [None,10])
    data_test = tf.placeholder(dtype = tf.float32,shape = [None,28*28])
    label_test = tf.placeholder(dtype = tf.float32,shape = [None,10])

    distance = tf.reduce_sum(tf.abs(tf.subtract(data_train,tf.expand_dims(data_test,1))), axis=2)
    session = tf.Session()
    batch = 1

    top_k_data ,top_k_indices = tf.nn.top_k(tf.negative(distance),k=k)
    K_nearest_labels = tf.gather(label_train,top_k_indices)

    pred = tf.argmax(tf.reduce_sum(K_nearest_labels,axis=1),axis = 1)
    test_output = []
    actual = []
    num_loops = int(np.ceil(len(test_data)/batch))
    for i in range(0,num_loops):
        min_i = i*batch
        max_i = min((i+1)*batch,len(test_data))

        batch_data = test_data[min_i:max_i]
        batch_label = test_label[min_i:max_i]

        prediction = session.run(pred,feed_dict={data_train:train_data,label_train:train_label,data_test:batch_data,label_test:batch_label})
        test_output.extend(prediction)
        actual.extend(np.argmax(batch_label,axis = 1))

    acc = []
    acc_av = 0
    for no in range(0,9):
        a = 0
        c = 0
        for i in range(0,len(test_data)):
            if test_label[i][no] == 1.0:
                c += 1
                if actual[i] == test_output[i]:
                    a += 1
        a = round(a * 100 / c,5)
        acc.append(a)

    for ac in acc:
        acc_av += ac
    acc_av = acc_av / len(acc)

    return acc,acc_av

def graph(train_data,train_labels,test_data,test_labels):
    '''
    This function displays graph with varying training data for k =1.
    '''
    datapoints = np.logspace(np.log10(31.0),np.log10(10000.0),num = 10,base=10.0,dtype='int')
    accuracy = []
    for i in range(10):
        accuracy.append(KNN(train_data[0:datapoints[i]],train_labels[0:datapoints[i]],test_data,test_labels,1)[1])
    plot.xlabel("Datapoints")
    plot.ylabel("Accuracy")
    plot.plot(datapoints,accuracy,'-o')
    plot.savefig('Datapoints_vs_Accuracy.png')

def k_accuracy(train_data,train_labels,test_data,test_labels):
    '''
    This function plots graph with varying training data for different values of k=[1,2,3,5,10]
    '''
    val = [1,2,3,5,10]
    color = ['r','g','b','y','b']
    datapoints = np.logspace(np.log10(31.0),np.log10(10000.0),num = 10,base=10.0,dtype='int')
    accuracy = []
    for k in val:
        for i in range(0,len(datapoints)):
            accuracy.append(KNN(train_data[0:datapoints[i]],train_labels[0:datapoints[i]],test_data,test_labels,k)[1])
        plot.plot(datapoints,accuracy,label='k='+str(k))
        accuracy = []
    plot.legend(loc = 'best')
    plot.xlabel("Datapoints")
    plot.ylabel("Accuracy")
    plot.savefig('Data_vs_Acc_diff_k.png')

def best_k(train_data,train_labels,valid_data,valid_lables):
    '''
    This function plots graph for 1000 training data and 1000 validation data for different K and return best K.
    '''
    val  = [1,2,3,5,10]
    accuracy = []
    for k in val:
        accuracy.append(KNN(train_data,train_labels,valid_data,valid_lables,k)[1])
    plot.plot(val,accuracy,'-o')
    plot.xlabel("K-values")
    plot.ylabel("Accuracy")
    plot.savefig('Best_k.png')
    m = max(accuracy)
    return [val[i] for i,j in enumerate(accuracy) if j == m]



def main():
    print("Importing MNIST Dataset")
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    print("1.Accuracy for 10 classes and average accuracy\n2.Accuracy plot for different dataset sizes\n3.Plot 2 for different K cases[1,2,3,5,10]\n4.Best K for 1000 Training and 1000 Validation data")
    choice = int(input("***Enter your choice***"))
    if choice == 1:
        accuracy = KNN(mnist.train.images,mnist.train.labels,mnist.test.images[0:1000],mnist.test.labels[0:1000],5)
        print("Accuracy vector",accuracy[0])
        print("Accuracy average",accuracy[1])
    elif choice == 2:
        print("Plot for different dataset sizes from 30 to 10000")
        graph(mnist.train.images,mnist.train.labels,mnist.test.images[0:1000],mnist.test.labels[0:1000])
    elif choice == 3:
        print("Plotting above graph for K=[1,2,3,5,10]")
        k_accuracy(mnist.train.images,mnist.train.labels,mnist.test.images[0:1000],mnist.test.labels[0:1000])
    elif choice == 4:
        print("Plot of K vs. 1000 Training data 1000 validation data")
        k = best_k(mnist.train.images[0:1000],mnist.train.labels[0:1000],mnist.train.images[1001:2000],mnist.train.labels[1001:2000])
        print("Best value of K for above data is",k[0])
    else:
        print("Exiting")

if __name__ == "__main__":
    main()
