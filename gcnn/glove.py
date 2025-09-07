import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import torch
import numpy as np
import pickle


name='SST1'# SST1
embedding_num=300

# 输入文件
glove_file = '../glove/vectors_'+name+'_clean.txt'
# 输出文件
# w2v_file = '../glove/w2v_400.bin'

# 使用新的加载方法
model = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)
# model.save(w2v_file)

# 加载模型
# model = KeyedVectors.load(w2v_file, mmap='r')

# 创建一个空列表来存储词向量
word_vectors_list = []
word_index = model.key_to_index

# 创建一个新的字典
new_word_index = {}

# 添加 <PAD> 和 <UNK>
new_word_index["<PAD>"] = 0
new_word_index["<UNK>"] = 1

# 将原始字典的内容添加到新字典中，索引增加2
for i, (word, index) in enumerate(word_index.items()):
    new_word_index[word] = index + 2


# 想通的数据集，想通的glove设定，他们的字典是相同的

# 删除索引最大的键值对
new_word_index.popitem()
print('new_word_index构建完成')
# 保存 word_index 字典
with open('../glove/word_index_'+name+'_clean.pkl', 'wb') as f:
    pickle.dump(new_word_index, f)
print('保存word_index成功!')


# 遍历模型中的词汇索引并添加词向量到列表
for index in range(len(model)):  # 使用索引遍历
    word = model.index_to_key[index]  # 获取词汇
    word_vectors_list.append(model[word])  # 使用词汇获取词向量

# 将词向量列表转换为NumPy数组，然后再转换为PyTorch张量
word_vectors_np_array = np.array(word_vectors_list, dtype=np.float32)
word_vectors_tensor = torch.from_numpy(word_vectors_np_array)

# 提取最后一行特征
last_feature = word_vectors_tensor[-1]

# 将 last_feature 转换为形状 [1, 300] 的张量
last_feature_unsqueezed = last_feature.unsqueeze(0)

# 选择除了最后一行之外的所有行
new_word_vectors_tensor = word_vectors_tensor[:-1, :]


print('加unk维度情况',new_word_vectors_tensor.shape)
# 将全0向量连接到现有的张量
new_word_vectors_tensor = torch.cat((last_feature_unsqueezed, new_word_vectors_tensor), dim=0)
print('加pad维度情况',new_word_vectors_tensor.shape)
# 创建一个全0的300维向量
zero_vector = torch.zeros(1, embedding_num)           # 这里是400维度
new_word_vectors_tensor = torch.cat((zero_vector, new_word_vectors_tensor), dim=0)

print('添加维度成功')

# 保存 new_word_vectors_tensor 张量
torch.save(new_word_vectors_tensor, '../glove/embedding_'+name+'_self_clean.pt')

print('保存new_word_vectors_tensor成功！')

print('new_word_index长度',len(new_word_index))
print('new_word_vectors_tensor长度',len(new_word_vectors_tensor))

# # 加载 word_index 字典
# with open('../glove/word_index_R52_300.pkl', 'rb') as f:
#     loaded_word_index = pickle.load(f)
#
# # 加载 new_word_vectors_tensor 张量
# loaded_word_vectors_tensor = torch.load('../glove/new_word_vectors_tensor_R52_300_15.pt')
#
# print("word_index 和 new_word_vectors_tensor 已加载。")
