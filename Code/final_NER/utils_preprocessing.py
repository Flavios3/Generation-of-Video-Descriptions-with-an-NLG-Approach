
import json
import os
import shutil

'''
This file is used to define the prerocessing functions.
In put data --> Overview of manual tagging
Output --> Two txt files: one dataset with tags, one without.
'''


ner_dir = "data/"
config_dir_default = "conf/"  # the path of label type ('ATTR' and 'PROD')


class TagObejct():  # sefine the data structure
    def __init__(self, tag, name):
        self.tag = tag  # start tag ('<attr>, <prod>')
        self.name = name  # tag name ('ATTR', 'PROD')
        self.close_tag = tag.replace('<', '</')  # end tag ('</attr>, </prod>')


class ner_PreProcessing():  # define the preprocessing class
    def __init__(self, filename="ner_dataset", tags=['<prod>', '<attr>'],
                 config_path=config_dir_default):
        self.tags = tags  # tag name
        self.ner_dir = ner_dir    # output direction
        self.default_dict = self.load(config_path + "conf.json")  # the path of label type ('ATTR' and 'PROD')
        self.filename = filename  # input filename
        self.setup()  # label name

    def setup(self):  # define the label name ('<prod>...</prod>', '<attr>...</attr>')
        self.tags_obj = []
        for tag in self.tags:
            self.tags_obj.append(TagObejct(tag, self.default_dict.get(tag)))

    def start(self):  # load the output file
        shutil.rmtree('./data', ignore_errors=True)
        try:
            self.ner_dataset = self.load(self.ner_dir + self.filename + ".txt")
        except:  # if there is no such file, create one
            print("No previous dataset found. Creating...")
            self.ner_dataset = []

    def add_element(self, text):  # detect if the input data has already in the output file
        if text in self.ner_dataset:
            pass
        else:  # if not, add them
            self.ner_dataset.append(text)

    def save(self):  # save the output file

        if not os.path.exists("./data"):
            # if the demo_folder directory is not present
            # then create it.
            os.makedirs("./data")
        with open(self.ner_dir + self.filename + ".txt", "w") as f:  # the data with tags
            f.write(json.dumps(self.ner_dataset))
        with open(self.ner_dir + self.filename + "_notags.txt", "w") as f:  # the data without tags
            f.write(json.dumps(self.remove_tags()))
        print("Saved!")

    def remove_tags(self):  # remove tags to create 'raw_file'
        ner_dataset_notags = []
        for element in self.ner_dataset:  # split '>' and punctuation
            element = element.replace(">.", "> .").replace(">,", "> ,")
            for tag in self.tags_obj:
                element = element.replace(tag.tag, "").replace(tag.close_tag,
                                                               "")  # replace the start('<..>') and end('</..>')
            ner_dataset_notags.append(element)
        # print(ner_dataset_notags)
        return ner_dataset_notags

    def load(self, file):  # load the input file
        with open(file, "r") as f:
            data = json.load(f)
        return data
