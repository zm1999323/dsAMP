import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from model import ProteinGenerator


# tokenizer = BertTokenizer.from_pretrained("/home/wuyou/project/pythonProject5/pythonProject/pythonProject1/prot_bert", do_lower_case=False )


def weight_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data,0.0,0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data,1.0,0.02)
        nn.init.constant(m.bias.data,0)


class ProteinDataset(Dataset):
    def __init__(self, fasta_file, max_length=100):
        self.sequences = self._load_sequences(fasta_file)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained("/home/wuyou/project/pythonProject5/pythonProject/pythonProject1/prot_bert", do_lower_case=False )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        # print(sequence)
        # padded_sequence = self._pad_sequence(sequence, self.max_length)
        # print(padded_sequence)
        seq = ''
        for i in range(0,len(sequence)):
            if i != 0:
                seq = seq+' '
            seq = seq+sequence[0]

        encoded_sequence = tokenizer(seq, return_tensors='pt')
        #encoded_sequence = self.tokenizer.encode(sequence[:self.max_length], add_special_tokens=False)
        # 填充序列至相同长度
        # padded_sequence = self._pad_sequence(encoded_sequence, self.max_length)
        #return torch.tensor(padded_sequence).to(device)
        return encoded_sequence
    def _load_sequences(self, fasta_file):
        sequences = []
        for record in SeqIO.parse(fasta_file, "fasta"):
            sequences.append(str(record.seq))
        return sequences

    def _pad_sequence(self, sequence, max_length):
        if len(sequence) < max_length:
            padding_length = max_length - len(sequence)
            padded_sequence = sequence + [self.tokenizer.pad_token_id] * padding_length
        else:
            padded_sequence = sequence[:max_length]
        return padded_sequence


# 计算损失函数
def compute_loss(outputs, targets, criterion):
    print(outputs.view(-1, outputs.size(-1)))
    print(targets.view(-1))
    return criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

def collate_fn(batch_encoding):
    tensor_list = [encoding['input_ids'] for encoding in batch_encoding]
    return pad_sequence(tensor_list,batch_first=True,padding_value=0)


def train_generator(generator, dataloader, optimizer, criterion, tokenizer, num_epochs=10,
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    generator.train()
    losses = []
    generated_sequences = []
    for epoch in range(num_epochs):
        total_loss = 0.0
        for inputs in dataloader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            print(inputs['input_ids'][0])
            outputs = generator(inputs['input_ids'][0])
            outputs = outputs.float()

            # 创建目标序列（原始序列向右移动一个位置）
            targets = inputs['input_ids'][0].contiguous().to(device)
            loss = compute_loss(outputs, targets, criterion)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            for output in outputs:
                decoded_sequence = tokenizer.decode(output, skip_special_tokens=True)
                generated_sequences.append(decoded_sequence)

            # 在训练结束后，将生成的序列写入新的 FASTA 文件
        with open("generated_sequences.fasta", "w") as output_handle:
            for i, sequence in enumerate(generated_sequences):
                seq_record = SeqRecord(Seq(sequence), id=f"generated_sequence_{i}", description="")
                SeqIO.write(seq_record, output_handle, "fasta")

        # 计算平均损失并记录
        epoch_loss = total_loss / len(dataloader)
        losses.append(epoch_loss)
        print("Epoch {} Loss: {:.4f}".format(epoch + 1, epoch_loss))

    # 绘制损失曲线
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator Loss')
    plt.show()


# 准备数据
fasta_file = "/home/wuyou/project/pythonProject5/pythonProject/pythonProject1/sample/natural_cpp.fasta"  # 请替换为您的 FASTA 文件路径
dataset = ProteinDataset(fasta_file)
dataloader = DataLoader(dataset, batch_size=1,collate_fn=collate_fn,shuffle=True)

# 初始化生成器模型和优化器
generator = ProteinGenerator()
generator.apply(weight_init)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)
optimizer = optim.Adam(generator.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
tokenizer = BertTokenizer.from_pretrained("/home/wuyou/project/pythonProject5/pythonProject/pythonProject1/prot_bert")

# 训练生成器模型并绘制损失曲线
train_generator(generator, dataloader, optimizer, criterion, tokenizer, num_epochs=10)
