
import json
import re
import random
import os
import spacy
from tqdm import tqdm
from spacy.training.example import Example
from spacy import displacy

'''
This file is used to define the training-related functions.
'''

input_dir_default="data/"               #input data
output_dir_default="model_new/"         #The path to save the model
config_dir_default="conf/"              #the name of tags ('prod', 'attr')

class TagObejct():                      #define the data sturcture of tags
    def __init__(self,tag,name):
        self.tag=tag                    #the start of tags -> '<'
        self.name=name                  #tag name
        self.close_tag=tag.replace('<','</')    #the end of tags -> '</'

class ner_Trainer():                    #define the data structure of training function.
    def __init__(self,input_dir=input_dir_default,output_dir=output_dir_default,n_iter=100,tags=['<prod>','<attr>'],config_path=config_dir_default):
        self.input_dir=input_dir        #the path of input data
        self.output_dir=output_dir      #the path of saving model
        self.n_iter=n_iter              #the epoch number of training
        self.tags=tags                  #tag name
        self.default_dict=self.load_data(config_path+"conf.json")               #tag name ->'{"<prod>":"PROD","<attr>":"ATTR"}'
        self.tag_data=self.load_data(self.input_dir+"ner_dataset.txt")          #load the input data with tags
        self.raw_data=self.load_data(self.input_dir+"ner_dataset_notags.txt")   #load the input data without tags
        self.setup()                    #label name

    def setup(self):                    #define the label name ('<prod>...</prod>', '<attr>...</attr>')
        self.tags_obj=[]
        for tag in self.tags:
            self.tags_obj.append(TagObejct(tag,self.default_dict.get(tag)))

    def train(self):                    #
        self.training_list=self.create_list_entities(self.tag_data,self.raw_data)
        self.start_train()

    def start_train(self):
        nlp=spacy.blank("en")                           #load the spacy pipeline
        ner = nlp.add_pipe('ner')                       #load ner from the pipeline
        for tag_obj in self.tags_obj:
            ner.add_label(tag_obj.name)                 #add the tags which need to predict into the model
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']        #keep only ner in the pipeline
        with nlp.disable_pipes(*other_pipes):  # only train NER
            optimizer = nlp.begin_training()            #defien the optimizer, default is sgd
            for itr in range(self.n_iter):              #training loop
                print(str(itr+1)+"/"+str(self.n_iter))
                random.shuffle(self.training_list)      #shuffle the training data(since we need to train 100 rounds)
                losses = {}                             #loss
                for text, annotations in tqdm(self.training_list):  #text:overview without tags (single row)
                                                                    #annotations: a list of entities [start pos, end pos, 'tag type']
                    doc = nlp.make_doc(text)                        #let model predicts the entities
                    example = Example.from_dict(doc, annotations)   #Construct an Example object from the predicted document and the reference                                                                          annotations provided as a dictionary. doc:predict entities, anno:real entities
                    nlp.update(
                        [example],
                        drop=0.5,                                   #drop rate 'https://machinelearningmastery.com/dropout-for-regularizing-deep                                                                        -neural-networks/'
                        sgd=optimizer,                              #optimizer
                        losses=losses)
                #print(losses)
        nlp.to_disk(self.output_dir)                                #Save the current state to a directory.
        try:                                                        #save the model
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            print("\nSaved model to", self.output_dir)
        except Exception as e:
            print(e)
            print("\nUnable to save the model", self.output_dir)

    def find_elements(self,text,tag_obj):                                   #input: overview and tags structure -> TagObejct()
        return re.findall(rf'{tag_obj.tag}(.+?){tag_obj.close_tag}', text)  #output: entity names

    def create_list_entities(self,data,raw_data):                           #input: data with tags, data without tags
        training_list=[]                                                    #output: a list of [overview, [a list of entities]]
        for i,description in enumerate(data):
            pattern={"entities":()}
            entity_list=(raw_data[i],pattern)
            entities=[]
            for tag_obj in self.tags_obj:
                for e in list(set(self.find_elements(description,tag_obj))):
                    tmp=self.find_pos(e.replace("(","\(").replace(")","\)").replace("+","\+"),raw_data[i],tag_obj.name)
                    entities+=tmp
            entity_list[1]["entities"]=entities
            training_list.append(entity_list)
        return training_list


    def find_pos(self,text,data,tag):                   #input:tag name,tag tpye, overview, output:a list contains:[start, end, 'type']
        tmp_list=[(m.start(),m.end(),tag) for m in re.finditer(text,data)]
        return tmp_list

    def load_data(self,file):
        with open(file, "r") as f:
            data = json.load(f)
        return data

#Test the model

class ner_Tester():                                     #define the test function structure
    def __init__(self,model_dir="model/"):              #load the model
        self.nlp = spacy.load(model_dir)

    def test(self,text):                                #input a single overview
        doc = self.nlp(text)                            #let model to predict the entities of the text
        print('Entities', [(ent.text, ent.label_) for ent in doc.ents]) #print the entites
        displacy.render(doc,style="ent",jupyter=True)   #show the result





