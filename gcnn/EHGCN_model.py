import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from  torch.utils.data import Dataset,DataLoader
import pywt  # 导入PyWavelets库
import torch.nn.init as init
import csv
import pandas as pd
import math
from torch.nn.parameter import Parameter
from scipy.sparse import issparse, diags, linalg
from torch.nn.modules.module import Module
import manifolds
def read_data0(train_or_test, name, num=None):
    # Construct file path
    file_path = f"../data/{name}_0/{train_or_test}.xlsx"

    # Read xlsx file using pandas
    df = pd.read_excel(file_path, engine='openpyxl', header=None)

    # Assume text is in the first column, labels in the second column
    texts = df.iloc[:, 0].tolist()
    labels = df.iloc[:, 1].tolist()

    if num is None:
        return texts, labels
    else:
        return texts[:num], labels[:num]
class TextDataset(Dataset):
    # Pre-store externally passed data into the class's internal functions
    def __init__(self,all_text,all_label,word_2_index,max_len):
        self.all_text = all_text
        self.all_label = all_label
        self.word_2_index = word_2_index
        self.max_len = max_len
        # self.tokenizer = tokenizer
    # Count the number of batch_size data to be extracted and define how to process each piece of data (data acquisition and preprocessing)
    # 1. Get data based on index; 2. Convert text data to numerical form; 3. Standardize data length to max_len
    def __getitem__(self, index):
        # Double ensure type safety
        text = str(self.all_text[index])
        # 1. Get the complete text content
        # text = self.all_text[index]

        # 2. Split text into word list
        words = text.split()

        # 3. Only keep characters that exist in word_2_index
        valid_words = [word for word in words if word in self.word_2_index]

        # 4. Crop to maximum length
        valid_words = valid_words[:self.max_len]

        # 5. Convert to index sequence
        text_idx = [self.word_2_index[word] for word in valid_words]

        # 6. Pad insufficient length
        text_idx = text_idx + [0] * (self.max_len - len(text_idx))

        label = int(self.all_label[index])
        text_idx = torch.tensor(text_idx).unsqueeze(dim=0)
        return text_idx, label
    # Return the total length of the data
    def __len__(self):
        return len(self.all_text)

class GraphConvolution2(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features,bias=True):
        super(GraphConvolution2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.relu = nn.ReLU()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)
    def forward(self, input, adj):
        output1 = torch.mm(input, self.weight) + self.bias
        output1 = torch.sparse.mm(adj, output1)
        return output1

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, manifold,in_features, out_features,dropout):
        super(GraphConvolution, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.bias = Parameter(torch.Tensor(out_features))
        self.dropout = dropout
        self.reset_parameters()
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)
    def forward(self, input, adj,self_c):
        output1 = self.manifold.mobius_matvec1(self.weight,input,self_c)

        bias1 = self.manifold.proj_tan0(self.bias.view(1, -1), self_c)
        hyp_bias = self.manifold.expmap0(bias1, self_c)
        hyp_bias = self.manifold.proj(hyp_bias, self_c)
        output1 = self.manifold.mobius_add(output1, hyp_bias, c=self_c)

        output1 = self.manifold.logmap0(output1, c=self_c)
        output1 = torch.sparse.mm(adj, output1)
        output1 = self.manifold.proj(self.manifold.expmap0(output1, c=self_c), c=self_c)

        return output1

class MODEL_1(nn.Module):
    def __init__(self, embedding_num, output1_dim,nclass, dropout,max_len,aaa):
        super(MODEL_1, self).__init__()
        self.output1_dim=output1_dim
        self.nclass=nclass
        self.max_len = max_len
        self.gc1 = GraphConvolution2(embedding_num-1, self.output1_dim, dropout)
        self.gc2 = GraphConvolution2(self.output1_dim, self.output1_dim,dropout)  #注意这里的gcn
        self.gc3 = GraphConvolution2(self.output1_dim, self.output1_dim, dropout)  # 注意这里的gcn
        self.n = Parameter(torch.FloatTensor(aaa, 1))  # 设定一个保留原数据特征的参数

        self.relu = nn.ReLU()

        self.Lin1 = Parameter(torch.FloatTensor(embedding_num-1, self.output1_dim))
        self.Lin1_bias = Parameter(torch.FloatTensor(self.output1_dim))

        self.dropout = nn.Dropout(p=dropout)
        self.dropout0 = dropout

        self.weight2 = Parameter(torch.FloatTensor(max_len, self.output1_dim))
        self.bias2 = nn.Parameter(torch.FloatTensor(max_len))

        self.classifier = Parameter(torch.FloatTensor(max_len*2, self.nclass))
        self.classifier_bias = Parameter(torch.FloatTensor(self.nclass))

        self.conv1 = nn.Conv1d(in_channels=max_len, out_channels=1, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=max_len, out_channels=1, kernel_size=1)


        self.loss_fun = nn.CrossEntropyLoss()

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.n, 0, 1)

        init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight2)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias2, -bound, bound)

        init.kaiming_uniform_(self.Lin1, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.Lin1)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.Lin1_bias, -bound, bound)

        init.kaiming_uniform_(self.classifier, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.classifier)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.classifier_bias, -bound, bound)

    def forward(self, A1_tensor,adj,batch_idx, batch_label=None):
        batch_idx1=batch_idx.squeeze(dim=1)

        drop_weight2 = F.dropout(self.weight2, self.dropout0, training=self.training)  # 应用dropout
        A1_tensor1 = self.dropout(A1_tensor)[:, 1:]
        A1_tensor1 = self.dropout(A1_tensor1)
        a0 = torch.mm(A1_tensor1, self.Lin1) + self.Lin1_bias

        a1 = a0 * self.n
        x_1 = self.gc1(A1_tensor1, adj)
        x_1 = a1 + (x_1 * (1-self.n))
        x_2 = self.gc2(x_1, adj)
        x_2 = self.gc3(x_2, adj)

        drop_weight = drop_weight2.transpose(0, 1)

        select_dims1 = torch.index_select(x_2, 0, batch_idx1.flatten())
        select_dims1 = select_dims1.view(batch_idx1.shape[0], batch_idx1.shape[1], self.output1_dim)
        select_dims1 = torch.matmul(select_dims1, drop_weight) + self.bias2
        select_dims1 = self.conv1(select_dims1)
        select_dims1 = select_dims1.squeeze(1)


        select_dims2 = torch.index_select(a0, 0, batch_idx1.flatten())
        select_dims2 = select_dims2.view(batch_idx1.shape[0], batch_idx1.shape[1], self.output1_dim)
        select_dims2 = self.relu(select_dims2)
        select_dims2 = torch.matmul(select_dims2, drop_weight) + self.bias2
        select_dims2 = self.relu(select_dims2)
        select_dims2 = self.conv2(select_dims2)
        select_dims2 = select_dims2.squeeze(1)

        select_dims = torch.cat([select_dims2 , select_dims1], dim=1)

        pre = torch.mm(select_dims, self.classifier) + self.classifier_bias

        if batch_label is not None:
            loss = self.loss_fun(pre, batch_label)
            return loss, select_dims
        else:
            return torch.argmax(pre, dim=-1), select_dims

# 模型
class MLDEL_2(nn.Module):
    def __init__(self, embedding_num, output1_dim,nclass, dropout,max_len,aaa):
        super(MLDEL_2, self).__init__()
        self.manifold = getattr(manifolds, 'Hyperboloid')()

        self.raw_c = nn.Parameter(torch.tensor([math.log(math.exp(1.0) - 1)]))
        self.softplus = nn.Softplus()

        self.nclass = nclass
        self.output1_dim=output1_dim
        self.gc1 = GraphConvolution(self.manifold,embedding_num, output1_dim, dropout)
        self.gc2 = GraphConvolution(self.manifold,self.output1_dim, self.output1_dim, dropout)
        self.n = Parameter(torch.Tensor(aaa, 1))


        self.Lin1 = Parameter(torch.Tensor(embedding_num, self.output1_dim))
        self.Lin1_bias = Parameter(torch.Tensor(self.output1_dim))

        self.dropout = nn.Dropout(p=dropout)
        self.dropout0 = dropout

        self.weight = Parameter(torch.Tensor(self.output1_dim, max_len))
        self.weight2 = Parameter(torch.Tensor(self.output1_dim, max_len))

        self.conv1 = nn.Conv1d(in_channels=max_len, out_channels=1, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=max_len, out_channels=1, kernel_size=1)

        self.classifier = Parameter(torch.Tensor(max_len*2, self.nclass))
        self.classifier_bias = Parameter(torch.Tensor(self.nclass))

        self.loss_fun = nn.CrossEntropyLoss()

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.n, 0, 1)
        init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.Lin1, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.Lin1)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.Lin1_bias, -bound, bound)
        init.kaiming_uniform_(self.classifier, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.classifier)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.classifier_bias, -bound, bound)

    def get_c(self):
        return self.softplus(self.raw_c) + 1e-5

    def forward(self, A1_tensor,adj,batch_idx, batch_label=None):
        self_c = self.get_c()

        batch_idx1=batch_idx.squeeze(dim=1)

        A1_tensor1 = self.dropout(A1_tensor)
        x_tan = self.manifold.proj_tan0(A1_tensor1, self_c)
        x_hyp = self.manifold.expmap0(x_tan, c=self_c)
        a = self.manifold.proj(x_hyp, c=self_c)

        a1 = self.manifold.mobius_matvec1(self.Lin1,a,self_c)

        bias1 = self.manifold.proj_tan0(self.Lin1_bias.view(1, -1), self_c)
        hyp_bias = self.manifold.expmap0(bias1, self_c)
        hyp_bias = self.manifold.proj(hyp_bias, self_c)
        a1 = self.manifold.mobius_add(a1, hyp_bias, c=self_c)
        a2 = self.manifold.mobius_matvec0(self.n, a1, self_c)

        x_1 = self.gc1(a, adj,self_c)
        x_1 = self.manifold.mobius_matvec0((1-self.n), x_1, self_c)
        x_1 = self.manifold.mobius_add0(x_1, a2, c=self_c)
        x_2 = self.gc2(x_1, adj,self_c)

        select_dims2 = torch.index_select(x_2, 0, batch_idx1.flatten())
        select_dims2 = select_dims2.view(batch_idx1.shape[0], batch_idx1.shape[1], self.output1_dim)
        drop_weight3 = F.dropout(self.weight, self.dropout0, training=self.training)
        select_dims2 = self.manifold.logmap3(select_dims2, self_c)
        select_dims2 = torch.matmul(select_dims2,drop_weight3)
        select_dims2 = self.conv1(select_dims2)
        select_dims2 = select_dims2.squeeze(1)

        select_dims3 = torch.index_select(a1, 0, batch_idx1.flatten())
        select_dims3 = select_dims3.view(batch_idx1.shape[0], batch_idx1.shape[1], self.output1_dim)
        drop_weight_2 = F.dropout(self.weight2, self.dropout0, training=self.training)

        select_dims3 = self.manifold.logmap3(select_dims3, c=self_c)
        select_dims3 = torch.matmul(select_dims3, drop_weight_2)
        select_dims3 = self.conv2(select_dims3)
        select_dims3 = select_dims3.squeeze(1)
        select_dims = torch.cat([select_dims2, select_dims3], dim=1)

        pre = torch.mm(select_dims, self.classifier) + self.classifier_bias
        if batch_label is not None:
            loss = self.loss_fun(pre, batch_label)
            return loss, select_dims
        else:
            return torch.argmax(pre, dim=-1), select_dims

import torch
import torch.nn as nn

class MLDEL_3(nn.Module):
    def __init__(self, nclass,max_len):
        super(MLDEL_3, self).__init__()
        self.nclass = nclass

        self.max_len = max_len

        self.liner_1 = nn.Linear(self.max_len*4, self.max_len*3)
        self.liner_2 = nn.Linear(self.max_len*3, self.max_len)

        self.classifier = Parameter(torch.Tensor(self.max_len, self.nclass))
        self.classifier_bias = Parameter(torch.Tensor(self.nclass))
        self.loss_fun = nn.CrossEntropyLoss()
        self.reset_parameters()
    def reset_parameters(self):
        init.kaiming_uniform_(self.classifier, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.classifier)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.classifier_bias, -bound, bound)

    def forward(self, feature_1,feature_2, batch_label=None):
        feature = torch.cat([feature_1,feature_2], dim=1)

        feature = self.liner_1(feature)
        feature = self.liner_2(feature)
        pre = torch.mm(feature, self.classifier) + self.classifier_bias
        if batch_label is not None:
            loss = self.loss_fun(pre, batch_label)
            return loss, pre
        else:
            return torch.argmax(pre, dim=-1), pre

def wavelet_packet_decomposition_2d(feature, wavelet, level):

    wpt_results = []

    for sample in feature:

        coeffs = pywt.wavedec(sample, wavelet, level=level)

        wpt_sample = np.concatenate(coeffs)
        wpt_results.append(wpt_sample)

    wpt_results = np.array(wpt_results)
    return wpt_results
def guiyi(feature):
    max_norms = torch.max(torch.abs(feature), dim=1, keepdim=True)[0]
    max_norms[max_norms == 0] = 1
    feature = feature / max_norms
    return feature

if __name__ == "__main__":
    epoch = 200
    batch_size = 32
    max_len = 100
    hidden_num = 64
    output1_dim = 128
    output2_dim = 128
    name = 'R52'       # ohsumed  R8   R52   mr     20ng  TREC
    import pickle
    with open('../glove/word_index_' + name + '.pkl', 'rb') as f:
         word_2_index = pickle.load(f)
    feature = torch.load('../glove/embedding_' + name + '.pt')
    feature = feature.float()
    max_norms = torch.max(torch.abs(feature), dim=1, keepdim=True)[0]
    max_norms[max_norms == 0] = 1
    feature = feature / max_norms

    train_texts1, train_labels_p = read_data0("train_new",name)
    assert len(train_texts1) == len(train_labels_p)

    type_list = list(set(train_labels_p))
    train_labels = []
    for i in train_labels_p:
        index = type_list.index(i)
        train_labels.append(str(index))

    dev_texts, dev_labels_p = read_data0("test_new",name)
    assert len(dev_texts) == len(dev_labels_p)

    dev_labels = []
    for i in dev_labels_p:
        index = type_list.index(i)
        dev_labels.append(str(index))
    del type_list

    class_num = len(set(train_labels))
    print('class_num',class_num)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_dataset = TextDataset(train_texts1, train_labels, word_2_index,max_len)
    train_dataloader = DataLoader(train_dataset,batch_size,shuffle=False)

    dev_dataset = TextDataset(dev_texts, dev_labels, word_2_index, max_len)
    dev_dataloader = DataLoader(dev_dataset, batch_size, shuffle=False)

    del train_texts1
    del dev_texts
    aaa = len(feature)

    M = 25
    adj = torch.load('D_adj_' + name + '.pt').to_dense()
    adj[range(aaa, aaa)] = 0
    top_values, top_indices = torch.topk(adj, M, dim=1)
    adj.zero_()
    adj.scatter_(1, top_indices, top_values)
    indices = torch.nonzero(adj)
    values = adj[indices[:, 0], indices[:, 1]]
    i = indices.t()
    adj = torch.sparse_coo_tensor(i, values, adj.size())

    adj2 = torch.load('H_adj_' + name + '.pt')
    adj2[range(aaa, aaa)] = 0
    top_values, top_indices = torch.topk(adj2, M, dim=1)
    adj2.zero_()
    adj2.scatter_(1, top_indices, top_values)
    indices = torch.nonzero(adj2)
    values = adj2[indices[:, 0], indices[:, 1]]
    i = indices.t()
    adj2 = torch.sparse_coo_tensor(i, values, adj2.size())
    del indices,top_values, top_indices,values,i

    wavelet = 'bior3.5' # haar db4 sym8 bior3.5
    level = 2
    feature = wavelet_packet_decomposition_2d(feature, wavelet, level)
    A1_tensor = torch.from_numpy(feature)

    A1_tensor = A1_tensor / max_norms

    o = torch.zeros_like(A1_tensor)

    A1_tensor = torch.cat([o[:, 0:1], A1_tensor], dim=1)
    A1_tensor = A1_tensor.to(device)
    adj = adj.to(device)
    adj2 = adj2.to(device)
    embedding_num1 = A1_tensor.shape[1]
    for i in range(100):
        print(f"Training iteration {i}")
        model_1 = MODEL_1(embedding_num=embedding_num1,
                      output1_dim=output1_dim,
                      nclass=class_num,
                      dropout=0.5,
                      max_len=max_len,
                      aaa=aaa)
        model_1 = model_1.to(device)

        model_2 = MLDEL_2(embedding_num=embedding_num1,
                      output1_dim=output1_dim,
                      nclass=class_num,
                      dropout=0.5,
                      max_len=max_len,
                      aaa=aaa)
        model_2 = model_2.to(device)

        model_3 = MLDEL_3(nclass=class_num,
                          max_len=max_len,)
        model_3 = model_3.to(device)
        lr = 0.01

        opt_1 = torch.optim.AdamW(model_1.parameters(), lr=lr)
        scheduler_1 = torch.optim.lr_scheduler.StepLR(opt_1, step_size=20, gamma=0.5)

        opt_2 = torch.optim.AdamW(model_2.parameters(), lr=lr)
        scheduler_2 = torch.optim.lr_scheduler.StepLR(opt_2, step_size=20, gamma=0.5)

        opt_3 = torch.optim.AdamW(model_3.parameters(), lr=lr)
        scheduler_3 = torch.optim.lr_scheduler.StepLR(opt_3, step_size=20, gamma=0.5)

        loss_1 = -1
        loss_2 = -1
        loss_3 = -1
        count = 0
        accuracys = 0
        for e in range(epoch):
            model_1.train()
            model_2.train()
            model_3.train()
            for batch_idx, batch_label in train_dataloader:
                # 放入gpu中进行训练
                batch_idx = batch_idx.to(device)
                batch_label = batch_label.to(device)

                loss_1, feature_1 = model_1.forward(A1_tensor, adj, batch_idx, batch_label)
                loss_1.backward()
                opt_1.step()
                opt_1.zero_grad()

                loss_2, feature_2 = model_2.forward(A1_tensor, adj2, batch_idx, batch_label)
                loss_2.backward()
                opt_2.step()
                opt_2.zero_grad()

                # --- 阻断梯度 + 优化 model_3 ---
                with torch.no_grad():
                    feature_1 = feature_1.detach()
                    feature_2 = feature_2.detach()
                feature_1 = guiyi(feature_1)
                feature_2 = guiyi(feature_2)

                loss_3, feature_3 = model_3.forward(feature_1, feature_2, batch_label)
                if loss_3.item() == 0:
                    loss_3 = torch.tensor(0.0001, requires_grad=True)
                loss_3.backward()
                opt_3.step()
                opt_3.zero_grad()
            scheduler_1.step()
            scheduler_2.step()
            scheduler_3.step()

            model_1.eval()
            model_2.eval()
            model_3.eval()

            pre_list = []
            for batch_idx, batch_label in dev_dataloader:
                batch_idx = batch_idx.to(device)
                pre_1, feature0_1 = model_1.forward(A1_tensor, adj, batch_idx)
                pre_2, feature0_2 = model_2.forward(A1_tensor, adj2, batch_idx)
                pre, feature_3 = model_3.forward(feature0_1, feature0_2)
                pre = pre.tolist()
                pre_list.extend(pre)

            dev_labels = [int(x) for x in dev_labels]

            correct = sum(1 for l, p in zip(dev_labels, pre_list) if l == p)
            accuracy = correct / len(dev_labels)

            print(
                f"Epoch {e} training result: loss_1: {loss_1} loss_2: {loss_2} loss_3: {loss_3} Accuracy: {accuracy} ")

            if accuracys < accuracy:
                accuracys = accuracy
                count = 0
            if count > 40:
                print(
                    f"Maximum accuracy: {accuracys} ")
                break
            if e == 199:
                print(
                    f"Maximum accuracy: {accuracys} ")
            count += 1