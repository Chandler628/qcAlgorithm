import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.optim
import time
import torch.utils.data as Data

n = 21
Delta = 2
clusterSet = []
device = torch.device('cuda:0')
is_gpu = torch.cuda.is_available()


#更改程序为在cpu下执行
# device = "cpu"
# is_gpu = 0

#随机生成数据
def read_data():
    data, target = make_blobs(n_samples=n, n_features=1, centers=[[4, 4], [6, 6], [8, 8]], shuffle=False,
                              cluster_std=[0.2, 0.2, 0.2],random_state=7)
    return data, target

#算法实现，通过gpu进行算法加速，计算potential函数以及梯度
def potential_gpu(x, data):
    global Delta
    x = x.clone().detach().requires_grad_(True)
    data = torch.tensor(data, device=device, dtype=torch.float64)
    sum1 = 0
    sum2 = 0
    for i in range(len(data)):
        distance = torch.sqrt(torch.sum((data[i] - x) ** 2, dim=1, keepdim=True)+1e-8)#该处平方根内若为零，则反向传播报异常值nan。解决思路是加一个极小值使其不为零
        sum1 += (distance ** 2) * torch.exp((-distance ** 2) / 2 * (Delta ** 2))
        sum2 += torch.exp((-distance ** 2) / (2 * (Delta ** 2)))
    y = (1 / (2 * (Delta ** 2))) * (sum1 / sum2)
    z = torch.ones((len(x), 1),device = device)
    y.backward(z)
    return y, x.grad

#梯度下降算法
def gradient_descent(data, learning_rate, iters):
    x = torch.tensor(data, device=device, dtype=torch.float64)
    y, grad = potential_gpu(x, data)
    for i in range(iters):
        if is_gpu:
            print(f"epoch:{i + 1:4d}||loss\n{y.cpu().detach().numpy()}")
        else:
            print(f"epoch:{i + 1:4d}||loss\n{y.detach().numpy()}")
        x = x - grad * learning_rate
        grad = np.round(grad.cpu().numpy(), decimals=3)
        if grad.any() == 0:
            break
        y, grad = potential_gpu(x, data)
        print(f"epoch:{i + 1:4d}  running  {time.time() - time1:0.5f}s")

    return x

#算法主体框架，寻找数据簇心
def qc(data, k, accuracy):
    x = gradient_descent(data, 5, 200)
    x = np.round(x.cpu().numpy(), decimals=accuracy)
    label = []
    cluster_center = np.unique(x, axis=0)
    cluster_number = np.zeros((cluster_center.shape[0], 1))
    for i in range(len(x)):
        for j in range(len(cluster_center)):
            if (x[i] == cluster_center[j]).all():
                label.append(j)
                cluster_number[j] += 1
                break
    print(label)
    print(cluster_number)
    for i in range(len(cluster_center)-1, -1, -1):
        if cluster_number[i] <= k:
            cluster_center = np.delete(cluster_center, i, axis=0)
    print("簇心坐标为: \n", cluster_center)

#若待检测数据为二维，可视化该数据
def visualization(data):
    x = np.linspace(4, 10, 100)
    y = np.linspace(4, 10, 100)
    x_mesh, y_mesh = np.meshgrid(x, y, indexing='ij')
    z_mesh = np.c_[np.reshape(x_mesh, (x_mesh.size, 1)), np.reshape(y_mesh, (y_mesh.size, 1))]
    z_mesh = torch.tensor(z_mesh, device=device, dtype=torch.float64)
    z_mesh, grad = potential_gpu(z_mesh, data)
    z_mesh = z_mesh.cpu().detach().numpy()
    z_mesh = np.reshape(z_mesh, x_mesh.shape)
    fig = plt.figure()
    sub = fig.add_subplot(111, projection='3d')
    sub.plot_surface(x_mesh, y_mesh, z_mesh, cmap=plt.cm.PuRd)
    sub.set_xlabel(r'$pc1$')
    sub.set_ylabel(r'$pc2$')
    sub.set_zlabel(r'$potential$')
    for i in range(len(data)):
        x = data[i][0]
        y = data[i][1]
        z = 0
        sub.scatter(x, y, z, color="black")
    plt.contour(x_mesh, y_mesh, z_mesh)
    plt.show()


if __name__ == '__main__':
    data, target = read_data()
    k = 3
    time_current = time.time()
    qc(data, k, 0)
    print(f"entire program running {time.time()-time_current}s")
    #二维数据方可可视化
    # visualization(data)
