import json
import random
from spacy.training.example import Example
import pandas as pd
import spacy
from sklearn.model_selection import KFold, RepeatedKFold
from tqdm import tqdm


class Validator():
    def __init__(self, df, tags):
        self.df = df
        self.tags = tags
        self.tag1 = tags[0]
        self.tag2 = tags[1]
        self.metrics = self.init_metrics()
        cols = list(self.metrics[self.tag1].keys())
        cols.append('tag')
        self.df_track = pd.DataFrame(columns=cols)

    def init_metrics(self):
        my_dict = {}
        for tag in self.tags:
            # my_dict[tag]={'correct':[],'missed_init':[],'FT_init%':[],'correct2':[],'partial':[],'missed':[],'FT%':[],'misclassified':[],'misclassified%':[]}
            my_dict[tag] = {'correct': 0, 'missed_init': 0, 'FT_init': 0, 'correct2': 0, 'perfect': 0, 'partial': 0,
                            'missed': 0, 'FT': 0, 'misclassified': 0, 'misclassified%': 0, 'total_t': 0, 'total_r': 0,
                            'total_t_set': 0, 'total_r_set': 0}

        return my_dict

    def kfold_valid(self, k, n):
        kf = KFold(n_splits=k, shuffle=True)
        random_state = 12883823
        rkf = RepeatedKFold(n_splits=k, n_repeats=n, random_state=random_state)
        for train, test in kf.split(self.df):
            self.metrics = self.init_metrics()
            train_df = self.df.loc[self.df.index[train]]
            training_list = create_train(train_df)

            test_df = self.df.loc[self.df.index[test]]

            # training the model without the transformer
            nlp, losses = train_ner(100, self.tags, training_list)

            # here we should load the ner with the tranformer, and create a for loop

            self.compare(nlp, test_df)
            for tag in self.tags:
                self.update_df(tag)
        # self.compute_stats()

    def update_df(self, tag):
        tmp_dict = self.metrics[tag].copy()
        tmp_dict['tag'] = tag
        print(tmp_dict)
        self.df_track.loc[len(self.df_track.index)] = list(tmp_dict.values())

    def compute_stats(self):
        self.df_track['correct'] = self.df_track['correct'] / self.df_track['total_t_set']
        self.df_track['FT_init'] = self.df_track['FT_init'] / self.df_track['total_t_set']
        self.df_track['missed_init'] = self.df_track['missed_init'] / self.df_track['total_r_set']

        self.df_track['correct2'] = self.df_track['correct2'] / self.df_track['total_t']
        self.df_track['partial'] = self.df_track['partial'] / self.df_track['total_t']
        self.df_track['perfect'] = self.df_track['perfect'] / self.df_track['total_t']
        self.df_track['FT'] = self.df_track['FT'] / self.df_track['total_t']
        self.df_track['misclassified'] = self.df_track['misclassified'] / self.df_track['total_t']
        self.df_track['missed'] = self.df_track['missed'] / self.df_track['total_r']

    def compare(self, nlp, test_df):
        for index, row in test_df.iterrows():
            test_id = row.id
            print(test_id)
            text = row.Description
            doc = nlp(text)
            attr_list = [ent.text for ent in doc.ents if ent.label_ == self.tag1]
            print("Detected features\n" + str(attr_list))
            prod_list = [ent.text for ent in doc.ents if ent.label_ == self.tag2]
            print("Detected product names\n" + str(prod_list))
            ref_list_a = self.df.loc[self.df['id'] == test_id].Attributes.tolist()[0]
            print("Reference attributes:\n" + str(ref_list_a))
            ref_list_p = self.df.loc[self.df['id'] == test_id].Product.tolist()[0]
            print("Reference product\n" + str(ref_list_p))

            complete_list_a = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents if
                               ent.label_ == self.tag1]
            complete_list_p = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents if
                               ent.label_ == self.tag2]
            self.compute_metrics(ref_list_a, attr_list, self.tag1)
            if len(ref_list_p) != 0:
                self.compute_metrics(ref_list_p, prod_list, self.tag2)
            ref_list_a = self.df.loc[self.df['id'] == test_id].Attributes_pos.tolist()[0]
            ref_list_p = self.df.loc[self.df['id'] == test_id].Product_pos.tolist()[0]
            self.compute_pos2({self.tag1: ref_list_a, self.tag2: ref_list_p},
                              {self.tag1: complete_list_a, self.tag2: complete_list_p})

    def compute_metrics(self, ref, test, tag):
        ref = list(map(lambda x: x.lower(), ref))
        test = list(map(lambda x: x.lower(), test))
        ref_set = set(ref)
        test_set = set(test)

        miss = len(ref_set.difference(test_set))
        FT = len(test_set.difference(ref_set))
        correct = len(test_set.intersection(ref_set))

        self.metrics[tag]['missed_init'] += miss
        self.metrics[tag]['FT_init'] += FT
        self.metrics[tag]['correct'] += correct
        self.metrics[tag]['total_r_set'] += len(ref_set)
        self.metrics[tag]['total_t_set'] += len(test_set)

    def compute_pos2(self, ref_dict, test_dict):
        ref_list = []
        for tag in self.tags:
            ref_list = ref_list + ref_dict[tag]
        tmp_list = []
        for tag in self.tags:
            test_list = test_dict[tag].copy()
            accuracy = 0
            partial = 0
            perfect = 0
            misclassified = 0
            for test in test_list:
                for ref in ref_list:
                    if (ref[0] <= test[1] <= ref[1]) or (ref[0] <= test[0] <= ref[1]):
                        tmp_list.append((ref[0], ref[1]))
                        if test[0] == ref[0] and test[1] == ref[1]:
                            accuracy = accuracy + 1
                            if test[2] == ref[2]:
                                perfect += 1
                        else:
                            partial = partial + 1

                        if test[2] != ref[2]:
                            misclassified = misclassified + 1
                        break

            missed = len(set([(x[0], x[1]) for x in ref_dict[tag]]) - set(tmp_list))
            print(missed)
            print(ref_dict[tag])
            print(tmp_list)
            print(set([(x[0], x[1]) for x in ref_dict[tag]]) - set(tmp_list))
            self.metrics[tag]['correct2'] += accuracy
            self.metrics[tag]['perfect'] += perfect
            self.metrics[tag]['partial'] += partial
            self.metrics[tag]['FT'] += len(test_list) - (accuracy + partial)
            self.metrics[tag]['missed'] += missed
            self.metrics[tag]['misclassified'] += misclassified
            self.metrics[tag]['total_t'] += len(test_list)
            self.metrics[tag]['total_r'] += len(ref_dict[tag])


def create_train(df):
    training = []
    for index, row in df.iterrows():
        pattern = {"entities": ()}
        entity_list = (row.Description, pattern)
        entity_list[1]["entities"] = row.Attributes_pos + row.Product_pos
        training.append(entity_list)
    return training


def train_ner(n_iter, tags, training_list):
    nlp = spacy.blank("en")
    ner = nlp.add_pipe('ner')

    ner.add_label(tags[0])
    ner.add_label(tags[1])

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        # optimizer = nlp.create_optimizer()
        for itn in range(n_iter):
            random.shuffle(training_list)
            losses = {}
            for text, annotations in tqdm(training_list):
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update(
                    [example],
                    drop=0.5,
                    sgd=optimizer,
                    losses=losses)
            # print(losses)
    return nlp, losses


def load_data(file):
    with open(file, "r") as f:
        data = json.load(f)
    return data
