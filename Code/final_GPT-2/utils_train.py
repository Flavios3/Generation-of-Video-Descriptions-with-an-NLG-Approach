import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelWithLMHead, AutoConfig, \
    AutoModelForPreTraining, GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader

# Defining the personalized tokens used during the tokenization process.
MODEL_NAME = "gpt2-medium"  # change in case for gpt2-medium usage
MAX_LEN = 768  # 124 suitable only for colab base  -> check which size is good for colab pro
special_tokens = {"bos_token": "<OVERV_START>",
                  "eos_token": "<OVERV_END>",
                  "unk_token": "<|UNK|>",
                  "pad_token": "<|PAD|>",
                  "additional_special_tokens": [
                      "<NAME_START>",
                      "<NAME_END>",
                      "<DESCR_START>",
                      "<DESCR_END>",
                      "<FEAT_START>",
                      "<NEXT_FEAT>",
                      "<FEAT_END>"]}


# Retrieval of the gpt2 tokenizer.
def myTokenizer(model="gpt2-medium", special_tokens=None):
    tokenizer = AutoTokenizer.from_pretrained(model)  # retrieve the tokenizer from the module
    tokenizer.add_special_tokens(special_tokens)  # add the customized tokenizer to the tokenizer set
    return tokenizer


def myModel(model, tokenizer):
    # add the new tokenizer to retrieve our customized model for fine tuning
    config = AutoConfig.from_pretrained(model,
                                        bos_token_id=tokenizer.bos_token_id,
                                        eos_token_id=tokenizer.eos_token_id,
                                        pad_token_id=tokenizer.pad_token_id,
                                        output_hidden_states=False)
    model = AutoModelForPreTraining.from_pretrained(model, config=config).cuda()
    model.resize_token_embeddings(len(tokenizer))
    return model


# Defining the class used to create the new dataset, the one tokenized.
class newDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        for txt in txt_list:
            encodings_dict = tokenizer(txt, truncation=True,
                                       max_length=max_length, padding="max_length", return_tensors="pt")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {'label': self.input_ids[idx],
                'input_ids': self.input_ids[idx],
                'attention_mask': self.attn_masks[idx]}


# Defining the method used to prepare the dataset to the fine tuning of the GPT-2
def prepareDataset(filepath, tokenizer):
    """
    with open(filepath,"r") as f:
        data=f.read()
    """
    with open(filepath, encoding="utf8", errors='ignore') as f:
        data = f.read()
    d = data.split(' <OVERV_END>')
    data_new = []
    for element in d:
        data_new.append(element + ' <OVERV_END>')
    final_dataset = newDataset(data_new, tokenizer, MAX_LEN)
    return final_dataset


def myTrainer(train_dataset, valid_dataset, output_dir, epochs, batch_size, tokenizer, model):
    training_args = TrainingArguments(output_dir=output_dir + '/model_medium_1epoches', num_train_epochs=epochs,
                                      logging_steps=500, save_steps=15000,
                                      per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
                                      fp16=True, fp16_opt_level='01',
                                      warmup_steps=500, weight_decay=0.01, logging_dir=output_dir + '/logs',
                                      report_to='none')
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer)

    trainer.train()
    trainer.save_model()