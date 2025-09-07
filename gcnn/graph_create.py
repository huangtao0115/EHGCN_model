'''''
主要生成，图，每个词的特征
'''
import numpy as np
import torch
import scipy.sparse as sp
from tqdm import tqdm

from scipy.sparse import coo_matrix
def adj_create(word_2_index,name):
    #首先构建的密集矩阵
    adj=np.zeros((word_2_index, word_2_index))
    print('加载边...')
    print('自环添加1')
    for i in range(word_2_index):
        adj[i][i]=1
    print('添加边.....')
    for e in tqdm(range(48)):
        edges_unordered = np.genfromtxt("./cora_bian_3/cora_r52"+str(e)+".cites",
                                    dtype=np.int32)
        for i in edges_unordered:  # 这里同以前相比有些许改动，加了一个权重的因素  以前的代码直接是 adj[i[0]][i[1]] = 1
            adj[i[0]][i[1]] += 1
            adj[i[1]][i[0]] += 1


    adj[range(word_2_index, word_2_index)] = 1
    # 假设 adj 是你的密集矩阵,转为稀疏矩阵
    adj_sparse = coo_matrix(adj)
    print('开始归一化和数据处理')
    adj = normalize(adj_sparse)
    print('转为torch')
    # 假设 adj 是你的稀疏矩阵
    adj_dense = adj.toarray()  # 将稀疏矩阵转换为密集矩阵
    adj_tensor = torch.FloatTensor(adj_dense)  # 将密集矩阵转换为PyTorch张量
    print('保存图')
    torch.save(adj_tensor, 'adj_glove_'+name+'_15_new_clean_all.pt')

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))#这是度矩阵
    print('1')
    r_inv = np.power(rowsum, -0.5).flatten()#生成D-1/2
    print('2')
    r_inv[np.isinf(r_inv)] = 0.
    print('3')
    r_mat_inv = sp.diags(r_inv)
    print('4')
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    print('5')
    return mx
import pickle
print('正在加载GCN图 loding....')
#这里输入的数字是，整个文档的汉字符号个数。
# 加载glove词向量，word_index

name ='ohsumed'


adj = adj_create(13361,name)


print('加载完成！')
