from utils_custom_ner import ner_Trainer
import sys
from utils_preprocessing import *

if __name__ == "__main__":

    def load(file="ner_dataset_final.txt"):  # define the load function
        with open(file, "r", encoding="utf8") as f:
            data = json.load(f)
        return data


    # Pre-processing part -> we obtain in ./data folder 2 files: a dataset with tags and the same dataset without tags
    new_data = load()  # load the input data
    n = ner_PreProcessing()  # set preprocessing data structure
    n.start()  # start preprocessing
    for text in new_data:  # remove the tags
        n.add_element(text)
    n.save()  # save the data

    # NER module training part
    try:
        output_dir = sys.argv[1]  # set the output path
        ner_trainer = ner_Trainer(output_dir=output_dir)  # define the training structure
    except:
        ner_trainer = ner_Trainer()

    ner_trainer.train()  # train
