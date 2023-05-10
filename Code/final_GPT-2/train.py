import os
import shutil
from utils_train import *
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
import spacy
from sklearn.utils import shuffle
from utils_create_train import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    model_dir = "../final_NER/model"
    data_dir = './data/'

    df = pd.read_csv('final_dataset.csv').drop(["Unnamed: 0"], axis=1)
    df = shuffle(df).reset_index().drop(["index"], axis=1)
    df = df.drop_duplicates(subset=[
        'description'])
    df.reset_index(drop=True, inplace=True)

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    # load spacy NER module
    nlp = spacy.load(model_dir)
    df['att_list'] = df.apply(lambda row: extract_info(nlp, row['description']), axis=1)

    train, test = train_test_split(df, train_size=0.9, random_state=42, shuffle=True)
    valid, test = train_test_split(test, train_size=0.8)

    train.reset_index(drop=True, inplace=True)
    valid.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    shutil.rmtree('./data', ignore_errors=True)
    if not os.path.exists("./data"):
        # if the demo_folder directory is not present
        # then create it.
        os.makedirs("./data")

    df2text(train, data_dir + 'train_medium.txt')
    df2text(valid, data_dir + 'valid_medium.txt')
    df2text(test, data_dir + 'test_medium.txt')

    print(train.shape)
    print(valid.shape)
    print(test.shape)

    # start the fine-tune of GPT2 module
    tokenizer = myTokenizer(MODEL_NAME, special_tokens)
    model = myModel(MODEL_NAME, tokenizer)

    # Preparing the datasets

    train_dataset = prepareDataset(filepath=data_dir + "train_medium.txt", tokenizer=tokenizer)
    valid_dataset = prepareDataset(filepath=data_dir + "valid_medium.txt", tokenizer=tokenizer)
    test_dataset = prepareDataset(filepath=data_dir + "test_medium.txt", tokenizer=tokenizer)

    # settings  max_length = 768, epoch = 5, batch_size = 4

    epochs = 5
    batch_size = 4
    trainer = myTrainer(train_dataset, valid_dataset, 'output', epochs, batch_size, tokenizer, model)

    # Important: it is required to unzip the result.tar.gz file before trying to test it

    # brief test of the fine-tuned module
    # print('Brief test of the trained module')
    # tokenizer = myTokenizer(MODEL_NAME, special_tokens)
    # model = AutoModelWithLMHead.from_pretrained(
    #     '/model')
    # model.resize_token_embeddings(len(tokenizer))
    # text = "<OVERV_START> <NAME_START> DELL PRECISION SERIE 7000 7560 ON7560WM01AU <NAME_END> <FEAT_START> Full HD " \
    #        "Display <NEXT_FEAT> 11th gen Intel Core i7 <NEXT_FEAT> 15.6 inch <NEXT_FEAT> 1920 x 1080 pixel " \
    #        "<NEXT_FEAT> 512GB of internal storage (SSD NVMe M.2) <NEXT_FEAT> 16GB DDR4-SRAM <NEXT_FEAT> Optical Drive " \
    #        "<NEXT_FEAT> NVIDIA T1200 (4GB) Discrete Graphics <NEXT_FEAT> Wi-Fi 6 <NEXT_FEAT> 1 HDMI <NEXT_FEAT> 2 x " \
    #        "USB 3.2 Type-A port <NEXT_FEAT> 2 x USB 3.2 Type-C port <NEXT_FEAT> 2 Microphones <NEXT_FEAT> Windows 10 " \
    #        "Pro 64 bit <NEXT_FEAT> Realtek ALC3204 Audio <NEXT_FEAT> 16:9 aspect ratio <NEXT_FEAT> 2 Speakers (2 W) " \
    #        "<NEXT_FEAT> HD front camera 0.92MP (1280 x 720 pixels) <NEXT_FEAT> Ethernet LAN port <FEAT_END> "
    # inputs = tokenizer.encode(text, return_tensors='pt')
    # outputs = model.generate(inputs, max_length=768, do_sample=True, temperature=0.1,
    #                          early_stopping=True,
    #                          num_return_sequences=1)
    # predicted = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(predicted.partition("<|PAD|>")[0].replace("  ", "."))
