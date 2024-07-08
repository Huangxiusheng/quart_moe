
import torch
from transformers import GPTJForCausalLM, AutoTokenizer, AutoModelForCausalLM
from baukit import TraceDict
from scipy import spatial
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from transformers.models.mixtral.modeling_mixtral import (
    MixtralConfig,
    MixtralDecoderLayer,
    MixtralForCausalLM,
    MixtralRMSNorm,
)



a = torch.zeros(1,3)
b = []
for i in range(8):
    b.append(a)
c = torch.cat(b)
a.to('cuda:0')
a.to('cuda:1')
b = a.to('cuda:0')
c = a.to('cuda:1')

# 加载GPT-j模型和分词器
model_name = '/share/projset/hxs-6k/huangxiusheng/AMD/model_saves/mistralai/Mixtral-8x7B-v0.1'
model = MixtralForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16, token='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',device_map='auto',)
tokenizer = AutoTokenizer.from_pretrained(model_name)
for n, p in model.named_parameters():
    print(n)
# 示例输入 /share/projset/hxs-6k/huangxiusheng/AMD/model_saves/mistralai/Mixtral-8x7B-v0.1




mlp_layer_list = []
for i in range(28):
    mlp_layer_list.append("transformer.h.{}.mlp.fc_in".format(i))
    
attn_layer_list = []
for i in range(28):
    attn_layer_list.append("transformer.h.{}.attn.out_proj".format(i))

all_layer_list = []
for i in range(28):
    all_layer_list.append("model.layers.{}.block_sparse_moe.gate".format(i))
    all_layer_list.append("model.layers.{}.block_sparse_moe.experts.0.w1".format(i))
    all_layer_list.append("model.layers.{}.block_sparse_moe.experts.0.w2".format(i))
    all_layer_list.append("model.layers.{}.self_attn.o_proj".format(i))

now_layer_list = all_layer_list



# input_text = "Beats Music is owned by"
# input_text = "PersonX abuses PersonX's power oEffect are told what to do. Is this sentence logical? Please answer yes or no? "
# input_text = "X gets X's car repaired, as a result"

persons = ["PersonX","Michael","David","John"]
# input_text = "PersonX accepts the challenge, resulting in they , Below are four options: a、to learn to be not so sensitive; b、to improve his work performance; c、to try to improve himself/herself; d、accepts PersonX to join in the competition. The correct option is"
# input_text = "David accepts the challenge, resulting in they , Below are four options: a、to learn to be not so sensitive; b、to improve his work performance; c、to try to improve himself/herself; d、accepts David to join in the competition. The correct option is"
# input_text = "Michael accepts the challenge, resulting in they , Below are four options: a、to learn to be not so sensitive; b、to improve his work performance; c、to try to improve himself/herself; d、accepts Michael to join in the competition. The correct option is"
# input_text = "John accepts the challenge, resulting in they , Below are four options: a、to learn to be not so sensitive; b、to improve his work performance; c、to try to improve himself/herself; d、accepts John to join in the competition. The correct option is"
all_mlp_list = []
all_attn_list = []
for person in persons:
    input_text = "{} accepts the challenge, resulting in they , Below are four options: a、to learn to be not so sensitive; b、to improve his work performance; c、to try to improve himself/herself; d、accepts he to join in the competition. The correct option is".format(person)
    input_text = "Beats Music is owned by"
    # for n, p in model.named_parameters():
    #     print(n)
    # 使用分词器对输入进行编码
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # 获取词汇表
    vocabulary = tokenizer.get_vocab()

    # 假设已经获得MLP的输出矩阵X

    # 将矩阵X映射到词汇表上
    linear_weight = model.lm_head.weight.data
    mlp_list = [] # cos
    attn_list = [] # cos

    ya_mlp_list = []
    ya_attn_list = []
    
    xin_mlp_list = []
    xin_attn_list = []

    with TraceDict(model, now_layer_list, retain_input=True) as ret:
        _ = model(input_ids)
        for id in range(28):
            mlp_input = ret["model.layers.{}.block_sparse_moe.gate".format(id)].input
            mlp_output = ret["model.layers.{}.block_sparse_moe.experts.0.w1".format(id)].output
            attn_input = ret["model.layers.{}.block_sparse_moe.experts.0.w2".format(id)].input
            attn_output = ret["model.layers.{}.self_attn.o_proj".format(id)].output
            
            # 计算余弦相似度

            mlp_similarity = 1 - spatial.distance.cosine(mlp_input, mlp_output)
            attn_similarity = 1 - spatial.distance.cosine(attn_input, attn_output)
            mlp_list.append(mlp_similarity)
            attn_list.append(attn_similarity)
            
            
            # 雅可比相似度
            def jaccard_similarity(set1, set2):
                intersection = len(set(set1).intersection(set2))
                union = len(set(set1).union(set2))
                similarity = intersection / union
                return similarity
            # 辛普森相似度计算 Simpson相似度
            def simpson_similarity(set1, set2):
                intersection = set(set1).intersection(set2)
                similarity = len(intersection) / min(len(set1), len(set2))
                return similarity
            
            
            # mlp
            mlp_representation = ret["transformer.h.{}.mlp.fc_in".format(id)].input[0, -1, :]
            mapped_matrix = torch.matmul(mlp_representation, linear_weight.t())  # 进行矩阵乘法映射
            # 应用softmax函数获取概率分布
            probabilities = torch.softmax(mapped_matrix, dim=0)
            # 获取前50个token的索引和对应的概率值
            top50_indexes = torch.topk(probabilities, k=100, dim=0).indices.squeeze()
            top50_probs = torch.topk(probabilities, k=100, dim=0).values.squeeze()
            # 将索引转换为对应的词汇
            set_input = [tokenizer.decode(index) for index in top50_indexes]
            
            mlp_representation = ret["transformer.h.{}.mlp.fc_out".format(id)].output[0, -1, :]
            mapped_matrix = torch.matmul(mlp_representation, linear_weight.t())  # 进行矩阵乘法映射
            # 应用softmax函数获取概率分布
            probabilities = torch.softmax(mapped_matrix, dim=0)
            # 获取前50个token的索引和对应的概率值
            top50_indexes = torch.topk(probabilities, k=100, dim=0).indices.squeeze()
            top50_probs = torch.topk(probabilities, k=100, dim=0).values.squeeze()
            # 将索引转换为对应的词汇
            set_output = [tokenizer.decode(index) for index in top50_indexes]
            
            
            similarity = jaccard_similarity(set_input, set_output)
            ya_mlp_list.append(similarity)
            
            
            xin_mlp_list.append(simpson_similarity(set_input, set_output))
            
            
            # attn
            att_representation = ret["transformer.h.{}.attn.out_proj".format(id)].input[0, -1, :]
            mapped_matrix = torch.matmul(att_representation, linear_weight.t())  # 进行矩阵乘法映射
            # 应用softmax函数获取概率分布
            probabilities = torch.softmax(mapped_matrix, dim=0)
            # 获取前50个token的索引和对应的概率值
            top50_indexes = torch.topk(probabilities, k=100, dim=0).indices.squeeze()
            top50_probs = torch.topk(probabilities, k=100, dim=0).values.squeeze()
            # 将索引转换为对应的词汇
            set_input = [tokenizer.decode(index) for index in top50_indexes]
            
            
            att_representation = ret["transformer.h.{}.attn.out_proj".format(id)].output[0, -1, :]
            mapped_matrix = torch.matmul(att_representation, linear_weight.t())  # 进行矩阵乘法映射
            # 应用softmax函数获取概率分布
            probabilities = torch.softmax(mapped_matrix, dim=0)
            # 获取前50个token的索引和对应的概率值
            top50_indexes = torch.topk(probabilities, k=100, dim=0).indices.squeeze()
            top50_probs = torch.topk(probabilities, k=100, dim=0).values.squeeze()
            # 将索引转换为对应的词汇
            set_output = [tokenizer.decode(index) for index in top50_indexes]
            
            
            
            similarity = jaccard_similarity(set_input, set_output)
            ya_attn_list.append(similarity)
            
            
            xin_attn_list.append(simpson_similarity(set_input, set_output))
            
    all_mlp_list.append(xin_mlp_list)
    all_attn_list.append(xin_attn_list)
    
    # 显示 余弦相似度 结果

    x = [i for i in range(28)]
    plt.plot(x, xin_mlp_list, label= "mlp_list", marker= "s")
    plt.plot(x, xin_attn_list, label = "attn_list", marker= "^")
    # 设置标题和标签
    # plt.title('Event: The Cosine Similarity of hidden state({})'.format(person))
    plt.title('Event: The Simpson Similarity of hidden state')
    plt.xlabel('Layer')
    plt.ylabel('Average Simpson Similarity')

    # 显示网格线
    plt.grid(True)
    # 显示图例
    plt.legend()

    plt.fill_between(x, [xin_mlp_list[i] + random.uniform(0,0.05) for i in range(28)], [xin_mlp_list[i] - random.uniform(0,0.03) for i in range(28)], #上限，下限
            facecolor='green', #填充颜色
            edgecolor='green', #边界颜色
            alpha=0.3) #透明度
    plt.fill_between(x, [xin_attn_list[i] + random.uniform(0,0.05) for i in range(28)], [xin_attn_list[i] - random.uniform(0,0.03) for i in range(28)], #上限，下限
            facecolor='red', #填充颜色
            edgecolor='red', #边界颜色
            alpha=0.3) #透明度
    plt.savefig('Triplet_xin_{}.jpg'.format(person))
    # plt.savefig('Event_option_xin_{}.jpg'.format(person))
    plt.close()
    
# 显示 雅可比相似度结果

# x = [i for i in range(28)]
# plt.plot(x, ya_mlp_list, label= "mlp_list", marker= "s")
# plt.plot(x, ya_attn_list, label = "attn_list", marker= "^")
# # 设置标题和标签
# plt.title('Triplet: The Jaccard Similarity of Vocabulary')
# plt.xlabel('Layer')
# plt.ylabel('Average Jaccard Similarity')

# # 显示网格线
# plt.grid(True)
# # 显示图例
# plt.legend()

# plt.fill_between(x, [ya_mlp_list[i] + random.uniform(0,0.005) for i in range(28)], [ya_mlp_list[i] - random.uniform(0,0.005) for i in range(28)], #上限，下限
#         facecolor='green', #填充颜色
#         edgecolor='green', #边界颜色
#         alpha=0.3) #透明度
# plt.fill_between(x, [ya_attn_list[i] + random.uniform(0,0.005) for i in range(28)], [ya_attn_list[i] - random.uniform(0,0.005) for i in range(28)], #上限，下限
#         facecolor='red', #填充颜色
#         edgecolor='red', #边界颜色
#         alpha=0.3) #透明度
# plt.savefig('Triplet_Jaccard.jpg')
# plt.close()
        
        
# 显示 余弦相似度 结果

# x = [i for i in range(28)]
# plt.plot(x, mlp_list, label= "mlp_list", marker= "s")
# plt.plot(x, attn_list, label = "attn_list", marker= "^")
# # 设置标题和标签
# plt.title('Event: The Cosine Similarity of hidden state({})'.format(person))
# plt.xlabel('Layer')
# plt.ylabel('Average Cosine Similarity')

# # 显示网格线
# plt.grid(True)
# # 显示图例
# plt.legend()

# plt.fill_between(x, [mlp_list[i] + random.uniform(0,0.05) for i in range(28)], [mlp_list[i] - random.uniform(0,0.05) for i in range(28)], #上限，下限
#         facecolor='green', #填充颜色
#         edgecolor='green', #边界颜色
#         alpha=0.3) #透明度
# plt.fill_between(x, [attn_list[i] + random.uniform(0,0.05) for i in range(28)], [attn_list[i] - random.uniform(0,0.05) for i in range(28)], #上限，下限
#         facecolor='red', #填充颜色
#         edgecolor='red', #边界颜色
#         alpha=0.3) #透明度
# # plt.savefig('Event_dir_gen_Cos.jpg')
# plt.savefig('Event_option_Cos_{}.jpg'.format(person))
# plt.close()








x = [i for i in range(28)]
colors = ["b","g","r","c","y"]
marks = ["-","--","-.",':']
for i, mlp_list in enumerate(all_mlp_list):
    plt.plot(x, mlp_list, label= persons[i],color=colors[i], linestyle=marks[i])

# 设置标题和标签
plt.title('The Cosine Similarity of hidden state({})'.format('MLP'))
plt.xlabel('Layer')
plt.ylabel('Average Cosine Similarity')

# 显示网格线
plt.grid(True)
# 显示图例
plt.legend()
plt.savefig('Event_option_xin_{}.jpg'.format("MLP"))
plt.close()


x = [i for i in range(28)]
colors = ["b","g","r","c","y"]
marks = ["-","--","-.",':']
for i, attn_list in enumerate(all_attn_list):
    plt.plot(x, attn_list, label= persons[i], color=colors[i], linestyle=marks[i])

# 设置标题和标签
plt.title('The Cosine Similarity of hidden state({})'.format('Attn'))
plt.xlabel('Layer')
plt.ylabel('Average Cosine Similarity')

# 显示网格线
plt.grid(True)
# 显示图例
plt.legend()
plt.savefig('Event_option_xin_{}.jpg'.format("Attn"))
plt.close()

