#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 GPT-2 升级为 GPT-OSS-20B，同时保留自有词汇表
> python train_AMP_GPTOSS.py
"""
import os, csv, random, time, logging, argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    GptOssConfig, AutoModelForCausalLM, GptOssForCausalLM,
    get_linear_schedule_with_warmup, BertTokenizer, BitsAndBytesConfig
)
from transformers.loss.loss_utils import ForCausalLMLoss
from peft import LoraConfig, get_peft_model, TaskType
from tqdm.auto import tqdm
from early_stop.pytorchtools import EarlyStopping   # 你原来的 early stop 工具

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ---------- 数据集 ----------
class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return torch.tensor(self.data_list[idx], dtype=torch.long)

# ---------- 参数 ----------
def setup_args():
    p = argparse.ArgumentParser()
    p.add_argument('--vocab_path', default='./voc/vocab.txt', type=str)
    p.add_argument('--train_raw_path', default='./data/uniprot/uniport_seq.csv', type=str)
    p.add_argument('--save_model_path', default='./gptoss_save_model', type=str)
    p.add_argument('--final_model_path', default='./gptoss_final_model', type=str)
    p.add_argument('--batch_size', default=4, type=int)
    p.add_argument('--epochs', default=20, type=int)
    p.add_argument('--warmup_steps', default=500, type=int)
    p.add_argument('--lr', default=1e-4, type=float)
    p.add_argument('--max_grad_norm', default=1.0, type=float)
    p.add_argument('--log_step', default=50, type=int)
    p.add_argument('--use_lora', default=1, type=int, help='1=LoRA, 0=full fine-tune')
    return p.parse_args()

# ---------- 分词器 ----------
def build_tokenizer(vocab_file):
    """
    用 GPT-2 分词器加载自有词汇表，保证格式与 GPT-2 相同（每行一个 token）
    """
    tokenizer = BertTokenizer(
        vocab_file=vocab_file,
        # merges_file=None,               # 无 BPE merges，仅字符级
        # bos_token='<s>',
        # eos_token='</s>',
        # pad_token='<pad>',
        # unk_token='<unk>'
    )
    return tokenizer

# ---------- 数据读取 ----------
def data_loader(args, tokenizer):
    data_list, eval_list = [], []
    with open(args.train_raw_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        data = [row[0] for row in reader if row]
    random.shuffle(data)
    split = len(data) // 10
    train_data, eval_data = data[split:], data[:split]

    for sent in tqdm(train_data, desc='tokenize train'):
        data_list.append(tokenizer.encode(sent))
    for sent in tqdm(eval_data, desc='tokenize eval'):
        eval_list.append(tokenizer.encode(sent))

    def collate(batch):
        # 确保所有输入都是列表形式
        batch = [x.tolist() if isinstance(x, torch.Tensor) else x for x in batch]
        lens = [len(x) for x in batch]
        max_len = max(lens)
        padded = [x + [tokenizer.pad_token_id] * (max_len - len(x)) for x in batch]
        return torch.tensor(padded, dtype=torch.long)

    train_loader = DataLoader(MyDataset(data_list),
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=collate)
    eval_loader = DataLoader(MyDataset(eval_list),
                             batch_size=args.batch_size,
                             shuffle=False,
                             collate_fn=collate)
    return train_loader, eval_loader

# ---------- 损失 ----------
def calculate_loss_and_accuracy(outputs, labels, pad_id, device):
    logits = outputs.logits[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous().to(device)
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    not_ignore = labels.ne(pad_id)
    num = not_ignore.sum()
    correct = (logits.argmax(-1) == labels) & not_ignore
    acc = correct.float().sum() / num.clamp(min=1)
    return loss / num.clamp(min=1), acc

# ---------- 训练 ----------
def train(args, model, base_model, train_loader, eval_loader, tokenizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)   # ① 手动 loss
    best_loss = float('inf')
    patience = 5

    total_steps = args.epochs * len(train_loader)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=total_steps)
    global_step = 0
    for epoch in range(args.epochs):
        epoch_loss = []
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}'):
            global_step += 1
            batch = batch.to(device)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # ② 不送 labels，只拿 logits
                logits = model(input_ids=batch).logits

            # ③ 手动计算交叉熵
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            torch.cuda.empty_cache()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if global_step % args.log_step == 0:
                logging.info(f'step {global_step}  loss={loss.item():.4f}')

        eval_loss = evaluate(model, eval_loader, tokenizer, device)
        logging.info(f'Epoch {epoch+1}  eval_loss={eval_loss:.4f}')
        if eval_loss < best_loss:
            best_loss = eval_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                logging.info(f"Early stopping at epoch {epoch + 1}")
                model_to_save = model.module if hasattr(model, 'module') else model
                logging.info(f"Saving model to {args.final_model_path}")
                model_to_save.save_pretrained(os.path.join(args.final_model_path, "oss_model.bin"))
                break
        # early_stop(eval_loss, model)
        # if early_stop.early_stop:
        #     logging.info('Early stopping')
        #     break

    # 保存最终模型
    os.makedirs(args.final_model_path, exist_ok=True)
    base_model.save_pretrained(args.final_model_path)
    model.save_pretrained(args.final_model_path)
    tokenizer.save_pretrained(args.final_model_path)
    logging.info(f'模型已保存至 {args.final_model_path}')

# ---------- 验证 ----------
@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device):
    model.eval()
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    losses = []
    for batch in data_loader:
        batch = batch.to(device)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = model(input_ids=batch).logits   # 同样不传 labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch[..., 1:].contiguous()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1))
        losses.append(loss.item())
    model.train()
    return np.mean(losses)

# ---------- 调整词表大小 ----------
def resize_token_embeddings(model, tokenizer, new_vocab_size):
    """调整模型的词表嵌入层大小"""
    old_embeddings = model.get_input_embeddings()
    old_vocab_size = old_embeddings.weight.size(0)
    
    if old_vocab_size == new_vocab_size:
        return model
    
    # 创建新的嵌入层
    embedding_dim = old_embeddings.weight.size(1)
    new_embeddings = torch.nn.Embedding(new_vocab_size, embedding_dim)
    
    # 初始化新的嵌入层
    new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)
    
    # 复制旧词表的权重（对于重叠的部分）
    min_vocab_size = min(old_vocab_size, new_vocab_size)
    new_embeddings.weight.data[:min_vocab_size] = old_embeddings.weight.data[:min_vocab_size]
    
    # 对新增加的token进行随机初始化
    if new_vocab_size > old_vocab_size:
        new_embeddings.weight.data[old_vocab_size:] = torch.normal(
            mean=0.0, 
            std=0.02, 
            size=(new_vocab_size - old_vocab_size, embedding_dim),
            device=new_embeddings.weight.device,
            dtype=old_embeddings.weight.dtype     # keep dtype consistent
        )
    
    model.set_input_embeddings(new_embeddings)
    
    # 如果存在输出嵌入层，也需要调整
    if hasattr(model, 'get_output_embeddings') and model.get_output_embeddings() is not None:
        output_embeddings = model.get_output_embeddings()
        new_output_embeddings = torch.nn.Linear(
            embedding_dim, new_vocab_size, bias=output_embeddings.bias is not None
        )
        
        if output_embeddings.bias is not None:
            new_output_embeddings.bias.data[:min_vocab_size] = output_embeddings.bias.data[:min_vocab_size]
        
        new_output_embeddings.weight.data[:min_vocab_size] = output_embeddings.weight.data[:min_vocab_size]
        
        if new_vocab_size > old_vocab_size:
            new_output_embeddings.weight.data[old_vocab_size:] = torch.normal(
                mean=0.0, 
                std=0.02, 
                size=(new_vocab_size - old_vocab_size, embedding_dim),
                device=new_output_embeddings.weight.device,
                dtype=output_embeddings.weight.dtype
            )
        
        model.set_output_embeddings(new_output_embeddings)
    
    model.config.vocab_size = new_vocab_size
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    
    # # 更新损失函数相关的配置
    # if hasattr(model, 'loss_function'):
    #     if hasattr(model.loss_function, 'vocab_size'):
    #         model.loss_function.vocab_size = new_vocab_size
        # model.loss_function = ForCausalLMLoss(vocab_size=new_vocab_size)
    return model

# ---------- 主入口 ----------
def main():
    args = setup_args()
    tokenizer = build_tokenizer(args.vocab_path)
    
    # 设置特殊token ID
    none = tokenizer.bos_token_id
    tokenizer.bos_token_id = tokenizer.cls_token_id
    tokenizer.eos_token_id = tokenizer.sep_token_id

    train_loader, eval_loader = data_loader(args, tokenizer)

    # 加载 GPT-OSS-20B
    model_id = "openai/gpt-oss-20b"
    
    # 先加载原始配置
    # config = GptOssConfig.from_pretrained(
    #     model_id,
    #     num_hidden_layers = 12,
    #     # head_dim = 32,
    #     # num_attention_heads = 16,
    #     # num_key_value_heads = 4,
    #     num_experts_per_tok = 2,
    #     # num_local_experts = 32,
    #     pad_token_id = tokenizer.pad_token_id,
    #     bos_token_id = tokenizer.bos_token_id,
    #     eos_token_id = tokenizer.eos_token_id)
    config = GptOssConfig.from_pretrained(
        model_id,
        num_hidden_layers=6,          # 原 36
        num_experts_per_tok=1,        # 原 4
        num_local_experts=4,          # 原 128
        sliding_window=64,            # 原 128
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # config = AutoConfig.from_pretrained(model_id)
    
    # 加载原始模型
    model = GptOssForCausalLM(config)
    # model = GptOssForCausalLM.from_pretrained(
    #     model_id,
    #     config=config,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto"
    # )
    model.gradient_checkpointing_enable()
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     config=config,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto"
    # )
    
    # 调整词表大小以匹配自定义词汇表
    new_vocab_size = len(tokenizer)
    base_model = resize_token_embeddings(model, tokenizer, new_vocab_size)
    
    logging.info(f"调整词表大小: {base_model.config.vocab_size} -> {new_vocab_size}")
    # if hasattr(model, 'loss_function'):
    #     logging.info(f"损失函数词汇表大小: {model.loss_function}")

    # for name, _ in model.named_modules():
    #     if 'proj' in name or 'fc' in name or 'attn' in name:
    #         logging.info(name)

    # # 可选：LoRA
    if args.use_lora:
        lora_conf = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj"]
        )
        model = get_peft_model(base_model, lora_conf)
        model.print_trainable_parameters()
    train(args, model, base_model, train_loader, eval_loader, tokenizer)

if __name__ == '__main__':
    main()