from pymongo import MongoClient
from bson.objectid import ObjectId
import json
import os
import random
import re
from transformers import AutoTokenizer, AutoModelWithLMHead
from difflib import SequenceMatcher
import itertools
import spacy
import nltk

stopwords = nltk.corpus.stopwords.words('english')
# blacklist=["battery","pocket","carry","purse","ergonomic"]
blacklist = []
nlp = spacy.load('en_core_web_sm')

filter_dir = "filter_fields/"
filter_cat_dir = "filter_cat/"
# output_dir="output/"
output_dir = "/data/"
parse_dir = "parse_feat/"

MODEL_NAME = "gpt2-medium"
MAX_LEN = 768
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


def process_json_rnd(content):
    tmp = {}
    sample_index = random.sample(range(0, len(content)), min(10, len(content)))
    for index in sample_index:
        tmp[list(content.keys())[index].replace("_", " ").title()] = list(content.values())[index]
    return tmp


class DB_Handler:
    def __init__(self,
                 db='mongodb+srv://giacomo:1234@adspflowygoproject.bulqzdd.mongodb.net/test?authSource=admin'
                    '&replicaSet=atlas-3qf7qz-shard-0&readPreference=primary&appname=MongoDB%20Compass&ssl=true',
                 collection="Products", schema='tv', saveFlag=False, output=output_dir):
        self.client = MongoClient(db)
        self.db = self.client[collection]
        self.schema = schema
        print("Query for {} - {}".format(collection, self.schema))
        self.saveFlag = saveFlag
        self.output_dir = output
        self.parse_content = load_json(parse_dir, self.schema + ".json")

    def complete_input(self, prod, category_list):
        self.select_by_cat_list(prod, category_list)
        output = self.content
        self.select(my_id=prod)
        self.create_input()
        overview = {'category': 'General Overview', 'input': self.input_list[0]}
        output.append(overview)
        output.insert(0, output.pop())
        return output

    def select(self, my_id=None, limit=None, rnd_flag=False):
        # print("\nFiltering...")
        black_list = load_txt(filter_dir, "ignore")
        if my_id is not None:
            cur = self.db.items.aggregate(
                [{"$match": {"_id": ObjectId('{}'.format(my_id))}}, {"$project": {field: 0 for field in black_list}}])

        else:
            if limit is not None:
                cur = self.db.items.aggregate([{"$match": {"schema": self.schema}}, {"$limit": limit},
                                               {"$project": {field: 0 for field in black_list}}])

            else:
                cur = self.db.items.aggregate(
                    [{"$match": {"schema": self.schema}}, {"$project": {field: 0 for field in black_list}}])
            # pprint.pprint(db.items.find_one({"schema":"tv"},{"brand":1,"model":1},{ "$project": { "_id": {
            # "$toString": "$_id" } } }))
        self.content = []
        for doc in cur:
            # print(doc)
            doc['_id'] = str(doc['_id'])
            # print(doc['_id'])
            try:
                doc['specs'] = {key: val for key, val in doc['specs'].items() if val != "No" and "package" not in key}
                if not rnd_flag:
                    doc['specs'] = self.process_json(doc['specs'])
                else:
                    doc['specs'] = process_json_rnd(doc['specs'])
                self.content.append(doc)
            except:
                print("Error with product: " + doc['_id'])
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if self.saveFlag:
            save_json(self.content, self.output_dir + doc['_id'] + "_general" + ".json")
        #     print("Saving...")
        # print("Operation completed!\n")

    def select_prod(self, limit):
        prod_list = []
        if limit is not None:
            try:
                cur = self.db.items.aggregate([{"$match": {"_id": ObjectId('{}'.format(limit))}}])
            except:
                cur = self.db.items.aggregate([{"$match": {"schema": self.schema}}, {"$sample": {"size": int(limit)}}])
            for doc in cur:
                prod_list.append(str(doc['_id']))
        return prod_list

    def find_spec(self, cat):
        cur = self.db.features.aggregate([{"$match": {"schema": self.schema, "category": cat, "visible": True}}])
        mydict = {}
        for doc in cur:
            mydict[doc['path'].split("specs.")[1]] = doc['name']
        return mydict

    def create_input2(self, content, rnd_filter=False):

        for cat in content:
            attributes = []
            name = (cat['title'] + " " + self.schema).upper()
            if not cat['specs']:
                res = False
            else:
                for pair in cat['specs'].items():
                    if pair[1] is not None:
                        full_attribute = self.parse_field(str(pair[1]), str(pair[0]))
                        attributes.append(full_attribute)
                # attributes.append(str(product['price'])+" euros")
                res = "<OVERV_START> <NAME_START> " + name + " <NAME_END> <FEAT_START> " + " <NEXT_FEAT> ".join(
                    attributes) + " <FEAT_END>"
                res = res.replace("Yes ", "").replace('"', " inch").replace("Streaming: ", "")
            # res=re.sub(r"\([^()]*\)", "", res)
        return res

    def create_input(self):
        self.input_list = []
        for product in self.content:
            attributes = []
            name = product['title'].upper() + " " + product['schema'].upper()
            for pair in product['specs'].items():
                full_attribute = self.parse_field(str(pair[1]), str(pair[0]))
                attributes.append(full_attribute)
            # attributes.append(str(product['price'])+" euros")
            res = "<OVERV_START> <NAME_START> " + name + " <NAME_END> <FEAT_START> " + " <NEXT_FEAT> ".join(
                attributes) + " <FEAT_END>"
            res = res.replace("Yes ", "").replace('"', " Inch")
            self.input_list.append(res)
        return self.input_list

    def parse_field(self, field_value, field_name):
        try:
            if field_value.find(",") != -1:
                field_value = str(list(field_value.split(","))[0])
            field = self.parse_content[field_name]
            if field['to_upper']:
                field_value = str(field_value).upper()
            if field['parameter_name'] is not None:
                field_name = field['parameter_name']
                return str(field_value + " " + field_name)
            else:
                return str(field_value)
        except Exception as e:
            # print(e)
            return str(field_value + " " + field_name.replace("_", " ").title())

    def process_json(self, content):
        tmp = {}
        fields = load_txt(filter_dir, self.schema + "_fields")
        for field in fields:
            try:
                # tmp[field.replace("_"," ").title()]=content[field]
                tmp[field] = content[field]
            except Exception as e:
                pass
                # print(e)
        return tmp

    def select_by_cat_list(self, my_id, cat_list):
        print("Filtering...")
        self.content = []
        for merged_list in cat_list:
            res = {"_id": None, "title": None, "specs": {}}
            tmp_content = []
            for cat in merged_list:
                print(cat)
                spec_dict = self.find_spec(cat)
                spec_list = list(spec_dict.keys())
                # print(spec_list)
                # black_list=self.load_txt(filter_dir,"ignore")
                try:

                    black_list = load_txt(filter_dir, self.schema + "_ignore")
                    spec_list = [x for x in spec_list if x not in black_list]
                except:
                    pass
                try:
                    cur = self.db.items.aggregate([{"$match": {"_id": ObjectId('{}'.format(my_id))}}, {
                        "$project": {"title": 1,
                                     "specs": {spec: {"$ifNull": ["$specs." + spec, None]} for spec in spec_list}}}])
                    # pprint.pprint(db.items.find_one({"schema":"tv"},{"brand":1,"model":1},{ "$project": { "_id": { "$toString": "$_id" } } }))
                    for doc in cur:
                        # print(doc)
                        doc['_id'] = str(doc['_id'])
                        res["_id"] = doc['_id']
                        res["title"] = doc['title']
                        try:
                            tmp_specs = {key: val for key, val in doc['specs'].items() if
                                         val != "No" and "package" not in key}

                            for key, value in list(doc['specs'].items()):
                                try:
                                    res['specs'][key] = tmp_specs[key]
                                except:
                                    pass
                        except Exception as e:
                            print(e)
                            print(res)
                            print("Error with product: " + doc['_id'])
                except Exception as e:
                    print(e)
                    pass
            tmp_content.append(res)
            tmp_content[0]['specs'] = self.prepare_content(tmp_content[0]['specs'])

            new_input = self.create_input2(tmp_content)
            if new_input is not False:
                self.content.append({"category": " & ".join(merged_list), "input": new_input})
        # print(self.content)
        if self.saveFlag:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            save_json(self.content, self.output_dir + my_id + ".json")
            print("\nSaving...")
        print("Operation completed!\n")

    def prepare_content(self, content):
        rmv_keys = []
        content = {k: v for k, v in content.items() if v is not None}
        for k, v in content.items():
            if k.find("version") != -1:
                general_key = k.split("_version")[0]
                try:
                    general_value = content[general_key]
                    del content[general_key]
                    content[general_key + " " + v] = general_value
                except:
                    pass
                rmv_keys.append(k)
        for k in rmv_keys:
            del content[k]
        content = dict(random.sample(content.items(), min(10, len(content))))
        return content

    def save_txt(self, content, filename):
        with open(self.output_dir + filename, "w") as f:
            f.write(content)


# Define a new class of function which is used to process the text generated by the GPT2 module
def process_hidden_title(text):
    doc = nlp(text)
    sub_toks = [tok for tok in doc if (tok.dep_ == "nsubj")]
    for noun in sub_toks:
        if str(noun).istitle():
            title = text.split(str(noun))[0].split('. ')[-1] + str(noun)
            title2 = ". " + text.split(str(noun))[0].split('. ')[-1] + str(noun) + "."
            word_list = title.split()
            number_of_words = len(word_list)
            if number_of_words > 1:
                text = text.replace(title, title2).replace(". .", "..")
    return text


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def find_title(mylist):
    for i in range(len(mylist) - 1):
        if mylist[i] == '' and mylist[i + 1].istitle():
            mylist[i + 1] = "\n\n####{}\n\n".format(mylist[i + 1])
    for e in mylist:
        if e == '':
            mylist.remove(e)
    return mylist


def find_title2(mylist):
    for i in range(len(mylist)):
        temp_list = []
        # split each sentence into word lever
        for word in mylist[i].split():
            # if the word is not a stopword (loaded from nltk library)
            if word.lower() not in stopwords:
                # add the word to the temporary list
                temp_list.append(word)
        # combine the words
        title = (' '.join(temp_list))
        # if the combined words is formed by uppercases -> it is a title
        if title.istitle():
            # add extra formatting for the title
            mylist[i] = "\n\n#### {}\n\n".format(mylist[i])
    return mylist


def filter_black(mylist):
    for e in mylist:
        for black in blacklist:
            if e.lower().find(black) != -1:
                try:
                    mylist.remove(e)
                except:
                    pass
    return mylist


def clean_review(review):
    # simple processing of the review
    review = review.replace("</h2>  <br>", "</h2> <br> ").replace("..", ".")
    # split the review in sentences based on the sent_tokenize function
    tmp_list = nltk.sent_tokenize(review)
    # remove the last element of each sentence (usually the dot)
    for i in range(len(tmp_list)):
        tmp_list[i] = tmp_list[i][:-1]

    # tmp_list=list(review.split('.'))
    # del tmp_list[-1:]
    # tmp_list=self.filter_black(tmp_list)

    tmp_list = list(dict.fromkeys(tmp_list))
    # add extra formatting to the titles (with various 'newline' and 'hashtags')
    tmp_list = find_title2(tmp_list)
    # for a and b sentences formed by combination of 2 elements of the sentence list
    for a, b in itertools.combinations(tmp_list, 2):
        # if both a and b are not title, and their similarity is very high
        if a.find('#') == -1 and b.find('#') == -1 and similar(a, b) > 0.9:
            try:
                # remove the sentence from the list
                tmp_list.remove(b)
            except:
                pass

    # restore the result by concatenating all the sentences with the dot (since it was removed previously)
    result = tmp_list[0]
    for i in range(1, len(tmp_list)):
        # careful in not adding the title
        if tmp_list[i - 1].find("#") == -1:
            result = result + ". " + tmp_list[i]
        else:
            # concatenate the title(since there were already 'newline' and 'hashtags', we don't need to add more puntuaction)
            result = result + tmp_list[i]

    # add the final dot
    result = result + "."

    # remove all the <h2>s and additional spaces
    l = re.findall("<h2> (.*?) </h2>", result)
    for a, b in itertools.combinations(l, 2):
        result = result.replace("<br><h2> " + b + " </h2>", "").replace("   ", " ")
    return result


class ProcessRev:
    def __init__(self):
        pass

    # this function can be removed -> only defined but not used

    # returns the similarity defined by SequenceMatcher

    # can be removed since only defined but not used

    # From a list of sentence(mylist),


def myTokenizer(model="gpt2-medium", special_tokens=None):
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer


# wrapper function that generates the review with the GPT2 module
# with the specific parameters of
# temperature
# top_k
# top_p
# (parameters are described in the module explanation)
def generate(content, tokenizer, model, temp=.75, k=35, p=0.95):
    print("Generating: ")
    title = str(re.search('<NAME_START> (.*)<NAME_END>', content[0]['input']).group(1))
    # print(title + '\n')
    review_complete = "# " + title + "\n\n"
    for cat in content:
        # print("<h2> " + cat.get("category").upper() + " </h2> ")
        # print("\n")
        text = cat.get("input")
        inputs = tokenizer.encode(text, return_tensors='pt').to("cuda:0")
        while True:
            errors = 0
            outputs = model.generate(inputs, max_length=768,
                                         do_sample=True,
                                         # num_beams=10,
                                         temperature=temp,
                                         # repetition_penalty=1.2,
                                         top_k=k,
                                         top_p=p,
                                         # no_repeat_ngram_size=3,
                                         early_stopping=True,
                                         # num_beam_groups=2,
                                         # diversity_penalty=12.0,
                                         num_return_sequences=1)
            try:
                output = tokenizer.decode(outputs[0], skip_special_tokens=False)
                review = str(re.search('<DESCR_START>(.*)<DESCR_END>', output).group(1)).replace('<h2>', '<h4>').replace(
                    "</h2>", "</h4>")
                
                break
            except:
                errors += 1
                
                
        # print(review)
        # review=process_hidden_title(review)
        review_complete = review_complete + "<h2> " + cat.get("category").upper() + "</h2> " + "\n\n" + \
            clean_review(review) + "\n\n"
        # print("\n")
    print('number of errors : ' + str(errors))

    return review_complete


def load_json(dir_path, filename):
    with open(dir_path + filename, "r") as file:
        content = json.load(file)
    return content


def save_json(content, file):
    with open(file, 'w') as file:
        json.dump(content, file, indent=4)


def load_txt(dir_path, name):
    filename = dir_path + name + ".txt"
    with open(filename, 'r') as f:
        black_list = f.read().splitlines()
    return black_list
