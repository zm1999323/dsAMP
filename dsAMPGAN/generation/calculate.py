# import pandas as pd
#
# # 读取 CSV 文件
# df = pd.read_csv("/home/zm/PycharmProjects/pythonProject6/dataset/sequence_length_range/data/AMPs_cidHit90_result.csv")
#
# # 计算第三列和第四列的单独数值之和
# column5_sum = df.iloc[:, 4].sum()
# column6_sum = df.iloc[:, 2].sum()
#
# # 创建包含结果的 DataFrame
# result_df = pd.DataFrame({
#     "Column3_Sum": [column5_sum],
#     "Column4_Sum": [column5_sum]
# })
#
# # 将结果保存到新的 CSV 文件
# result_df.to_csv("sums.csv", index=False)




# import random
#
# def extract_random_sequences(input_file, output_file, num_sequences=500):
#     # 读取所有序列和标识符，并存储在列表中
#     sequences = []
#     with open(input_file, 'r') as f_in:
#         current_sequence = ''
#         for line in f_in:
#             line = line.strip()
#             if line.startswith('>'):
#                 if current_sequence:
#                     sequences.append(current_sequence)
#                 current_sequence = line + '\n'
#             else:
#                 current_sequence += line + '\n'  # 添加换行符以保持每行序列的格式
#         if current_sequence:
#             sequences.append(current_sequence)
#
#     # 从列表中随机选择指定数量的序列
#     random_sequences = random.sample(sequences, min(num_sequences, len(sequences)))
#
#     # 将选定的序列写入输出文件
#     with open(output_file, 'w') as f_out:
#         for seq in random_sequences:
#             f_out.write(seq)
#
# input_file = '/home/zm/PycharmProjects/pythonProject6/dataset/others/Escherichia_colirenamed_sequences.fasta'  # 输入的FASTA文件名
# output_file = '/home/zm/PycharmProjects/pythonProject6/dataset/others/Escherichia_coli500.fasta'   # 输出的文本文件名
# extract_random_sequences(input_file, output_file)

# def rename_sequences(input_file, output_file):
#     with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
#         sequence_count = 0
#         for line in f_in:
#             line = line.strip()
#             if line.startswith('>'):
#                 # 增加序列计数
#                 sequence_count += 1
#                 # 写入新的标识符行
#                 f_out.write(f'>{sequence_count}\n')
#             else:
#                 # 写入序列行
#                 f_out.write(line + '\n')
#
# input_file = '/home/zm/PycharmProjects/pythonProject6/dataset/others/Streptococcus_pneumoniae.fasta'  # 输入的FASTA文件名
# output_file = '/home/zm/PycharmProjects/pythonProject6/dataset/others/Streptococcus_pneumoniaerenamed_sequences.fasta'   # 输出的FASTA文件名
# rename_sequences(input_file, output_file)


# def filter_sequences(input_file, output_file):
#     with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
#         current_sequence = ''
#         current_sequence_name = ''
#         for line in f_in:
#             line = line.strip()
#             if line.startswith('>'):
#                 # 如果当前序列不为空且长度大于等于8，则写入输出文件
#                 if current_sequence and len(current_sequence) >= 8:
#                     f_out.write(current_sequence_name + '\n')
#                     f_out.write(current_sequence + '\n')
#                 # 重置当前序列和序列名称
#                 current_sequence_name = line
#                 current_sequence = ''
#             else:
#                 # 将序列添加到当前序列中
#                 current_sequence += line
#         # 写入最后一个序列（如果长度大于等于8）
#         if current_sequence and len(current_sequence) >= 8:
#             f_out.write(current_sequence_name + '\n')
#             f_out.write(current_sequence + '\n')
#
# input_file = '/home/zm/PycharmProjects/pythonProject6/dataset/others/Streptococcus_pneumoniaerenamed_sequences.fasta'  # 输入的FASTA文件名
# output_file = '/home/zm/PycharmProjects/pythonProject6/dataset/others/Streptococcus_pneumoniaerenamed_sequences8.fasta'   # 输出的过滤后的FASTA文件名
# filter_sequences(input_file, output_file)

def count_sequences(fasta_file):
    sequence_count = 0
    with open(fasta_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                sequence_count += 1
    return sequence_count

fasta_file = "/home/zm/PycharmProjects/pythonProject6/dataset/anti-pseu/pseudo_neg_cdhit90.fasta"  # 替换为你的FASTA文件路径
num_sequences = count_sequences(fasta_file)
print("序列个数：", num_sequences)
