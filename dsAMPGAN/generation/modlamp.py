import pandas as pd
import re

# 读取CSV文件
df = pd.read_csv('/home/zm/PycharmProjects/pythonProject7/AMP/modlamp/randommodlamp.csv')

# 提取第二列到第三列的数字部分，包括小数点
def extract_numbers(text):
    return ' '.join(re.findall(r'\d+\.\d+|\d+', text))

for col in df.columns[1:11]:
    df[col] = df[col].astype(str).apply(extract_numbers)

# 将结果保存到一个新的CSV文件
df.to_csv('/home/zm/PycharmProjects/pythonProject7/AMP/modlamp/randommodlampoutput.csv', index=False)

print("处理后的数据已保存到 'output.csv'")
