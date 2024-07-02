from transformers import BertModel, BertTokenizer
import re
tokenizer = BertTokenizer.from_pretrained("/home/wuyou/project/pythonProject5/pythonProject/pythonProject1/prot_bert", do_lower_case=False )
model = BertModel.from_pretrained("/home/wuyou/project/pythonProject5/pythonProject/pythonProject1/prot_bert")
sequence_Example = "A E T C Z A O"
print(sequence_Example)
sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)
encoded_input = tokenizer(sequence_Example, return_tensors='pt')
print(encoded_input)
print(type(encoded_input))
output = model(**encoded_input)
print(output)