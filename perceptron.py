import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio

#reading the dataset
dataset_no = 1
learning_rate = 0.001
no_of_iterations = 100
w = np.ndarray((3,1), buffer = np.array([0.0, 0, 0]))
x = pd.read_csv('Datasets/dataset_' + str(dataset_no) + '.csv')
print("Running On Dataset-" + str(dataset_no))
new = []
for i in zip(x['x'], x['y'], x['t']):
    new.append((i[0], i[1], i[2]))
    
new = np.array(new)
#sepearating the third column into two classes
class0 = np.array([i for i in new if i[2]==0])
class1 = np.array([i for i in new if i[2]==1])

gif = []

#scattering the points of the two different classes
plt.scatter(class0[:,0], class0[:, 1], label="dots", color="blue", marker=".", s=3)
plt.scatter(class1[:,0], class1[:, 1], label="dots", color="red", marker=".", s=3)
plt.savefig('Outputs/Perceptron/dataset-' + str(dataset_no) + '-scatterplot.png')
plt.show()

data = []
for i in zip(x['x'], x['y'], x['t']):
    data.append((i[0], i[1], 1))
    
data = np.array(data)

#Assigning the sign and calculating the error
for iter in range(0, no_of_iterations):
    for row in range(len(data)):
        x = np.ndarray((3, 1), buffer = np.array([data[row][0], data[row][1], data[row][2]]))   
        sign = np.transpose(w).dot(x)
        if(sign>=0):
            sign=1
        else:
            sign=0
        res = new[row][2] - sign
        w = w + learning_rate*res*x
    fig, ax = plt.subplots()
    x = np.linspace(-3, 2,1000)
    ax.scatter(class0[:,0], class0[:, 1], label="dots", color="blue", marker=".", s=3)
    ax.scatter(class1[:,0], class1[:, 1], label="dots", color="red", marker=".", s=3)
    y = (-w[0]/w[1])*x -w[2]/w[1]
    ax.plot(x, y, label='the line', color = "green")
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    gif.append(image.reshape(fig.canvas.get_width_height()[::-1] + (3,)))
    
imageio.mimsave('Outputs/Perceptron/dataset-' + str(dataset_no) + '.gif', gif, fps=2)
print("w: {}".format(w))