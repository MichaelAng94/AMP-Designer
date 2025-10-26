import time
import torch
import argparse
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from transformers.models.gpt2 import GPT2LMHeadModel
from transformers import BertTokenizer
import torch.nn.functional as F

class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        input_ids = self.data_list[index]
        return input_ids

    def __len__(self):
        return len(self.data_list)


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default="./small_final_model", type=str, help='')
    parser.add_argument('--vocab_path', default="", type=str, help='')
    parser.add_argument('--batch_size', default=128, type=int, required=False, help='batch size')
    parser.add_argument('--epochs', default=1001, type=int, required=False, help='epochs')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--log_step', default=100, type=int, required=False, help='print log steps')
    return parser.parse_args()


def decode(matrix):
    chars = []
    for i in matrix:
        if i == '[SEP]': break
        chars.append(i.upper())
    seq = "".join(chars)
    return seq

def predict(model, tokenizer, batch_size, text=""):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    model.to(device)
    model.eval()
    time1 = time.time()
    max_length = 32
    input_ids = []
    input_ids.extend(tokenizer.encode(text))
    input_ids = input_ids[0]

    input_tensor = torch.zeros(batch_size, 1).long()

    input_tensor[:] = input_ids

    Seq_list = []

    finished = torch.zeros(batch_size,1).byte().to(device)

    for i in range(max_length):
        # input_tensor = torch.tensor([input_ids])
        inputs = {"input_ids": input_tensor.to(device)}
        outputs = model(**inputs)
        logits = outputs.logits

        logits = F.softmax(logits[:,-1,:])

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"Warning: Invalid values in probability distribution - NaN: {torch.isnan(logits).any()}, Inf: {torch.isinf(logits).any()}")
            prob = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=0.0)
            
        # 确保非负
        prob = torch.clamp(prob, min=0)
            
        # 归一化
        prob_sum = prob.sum(dim=-1, keepdim=True)
        zero_mask = (prob_sum == 0)
        if zero_mask.any():
            print("Warning: Zero probability sum, using uniform distribution")
            # 对于和为0的情况，使用均匀分布
            uniform_prob = torch.ones_like(prob) / prob.size(-1)
            prob = torch.where(zero_mask, uniform_prob, prob / prob_sum)
        else:
            prob = prob / prob_sum

        last_token_id = torch.multinomial(prob, 1)
        # .detach().to('cpu').numpy()
        EOS_sampled = (last_token_id == tokenizer.sep_token_id)
        finished = torch.ge(finished + EOS_sampled, 1)
        if torch.prod(finished) == 1:
            print('End')
            break

        last_token = tokenizer.convert_ids_to_tokens(last_token_id)

        input_tensor = torch.cat((input_tensor, last_token_id.detach().to('cpu')), 1)

        Seq_list.append(last_token)
    # print(Seq_list)
    Seq_list = np.array(Seq_list).T


    print("time cost: {}".format(time.time() - time1))
    return Seq_list
    # print(Seq_list)


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':
    args = setup_args()
    args.model_path, args.vocab_path = '', './voc/vocab.txt'


    tokenizer = BertTokenizer(vocab_file=args.vocab_path)

    tokenizer.bos_token_id = tokenizer.cls_token_id
    tokenizer.eos_token_id = tokenizer.sep_token_id

    model = GPT2LMHeadModel.from_pretrained('./final_prompt_model')

    print(f"BOS token: {tokenizer.bos_token_id} -> {tokenizer.convert_ids_to_tokens([tokenizer.bos_token_id])}")
    print(f"EOS token: {tokenizer.eos_token_id} -> {tokenizer.convert_ids_to_tokens([tokenizer.eos_token_id])}")
    print(f"PAD token: {tokenizer.pad_token_id} -> {tokenizer.convert_ids_to_tokens([tokenizer.pad_token_id])}")
    
    output = []
    Seq_all = []
    for i in range(100):
        Seq_list = predict(model,tokenizer,batch_size=128)

        Seq_all.extend(Seq_list)
    for j in Seq_all:
        output.append(decode(j))

    output = pd.DataFrame(output)

    output.to_csv('generate_seq.csv', index=False, header=False, sep=' ')


