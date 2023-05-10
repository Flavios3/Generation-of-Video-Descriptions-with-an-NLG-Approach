from sentence_transformers import SentenceTransformer, util
import json

#this function extracts the lsit of features
def extract_features(features_path):
    f = open(features_path)
    data = json.load(f)
    feature_list = list()
    for dict in data: 
        features = list(dict.values())
        print(features)
        print()
        #skip every other feature
        skipTitle = True
        for feature in features:
            if skipTitle:
                skipTitle = False
                continue
            else:
                skipTitle = True
                words = feature.split()
                in_features = False
                next_feature=True
                for word in words:
                    print(" I'm reading word ",word)
                    if word == "<FEAT_START>":
                        print(" I read <FEAT_START>")
                        in_features = True
                        continue
                    if word == "<FEAT_END>":
                        in_features = False
                        feature_list.append(new_feature)
                        break
                    if in_features:
                        if word=="<NEXT_FEAT>":
                            next_feature=True
                            feature_list.append(new_feature)
                        elif next_feature:
                            new_feature = word
                            next_feature = False
                        else:
                            new_feature = new_feature + " " + word
    return feature_list


#extract pargraph dict
def extract_paragraph_dict(review_path):
    paragraph_dict = { }
    #read the review line by line
    with open(review_path) as f:
        lines = f.read().split("\n")
    #remove the title and the empty lines
    lines.pop(0)
    for line in lines:
        if not line:
            lines.remove(line)
    """#debug
    for line in lines:
        print("new line: ", line)"""
    #create the dictionary
    isTitle = True
    for line in lines:
        if isTitle:
            line = line.replace("#","")
            line = line.replace("<h2>","")
            line = line.replace("</h2>","")
            line = line.replace("<h4>","")
            line = line.replace("</h4>","")
            new_key = line.strip()
            isTitle = False
        else:
            line = line.replace("#","")
            line = line.replace("<h2>","")
            line = line.replace("</h2>","")
            line = line.replace("<h4>","")
            line = line.replace("</h4>","")
            paragraph_dict[new_key] = line.strip()
            isTitle = True
    #print(paragraph_dict)
    #print(paragraph_dict.keys())
    return paragraph_dict


#compute the sentence-granularity dict
def compute_final_dict(input_features, paragraph_dict):
    """
    the output will be a dictionary with key: "paragraph title" and value: dictionary
    the dictionaries inside (from now on, value_dict) will have:
    value: sentence
    key:
        ( ) (i.e. an empty tuple) when no feature is found in the sentence
        (feature,) if one and only one feature is found is the sentence
        (feature1, ..., featurek) if k features are found in the sentence
    """
    #define the setnence embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    #initialize the final dict
    final_dict = dict()
    #since empty tuples would have the same key, but keys must be distinct, we use a placehoder int as key
    empty_counter = 0
    #create a dictionary of (feature_sentence, feature) pairs
    features_sentences_dict = dict()
    for feature in input_features:
        features_sentences_dict["this product has " + feature + "."] = feature
    ####print(" DICT: ", features_sentences_dict)
    #for each paragraph
    for item in paragraph_dict.items():
        #initialize value_dict
        value_dict = dict()
        #rename for readability
        paragraph_title = item[0]     
        paragraph_text = item[1]
        #create a copy of the feature sentences (we want to find the same feature at most 1 for each paragraph)
        candidate_features  = list(features_sentences_dict.keys())
        #move to a sentence granularity, removing the empty sentences meanwhile (caused by the dot at the end of the paragraph)
        sentences = [x for x in paragraph_text.split(".") if x != '']
        for sentence in sentences:    
            #print("\n\n Current sentence: ", sentence)
            #set of the (eventual) found matches
            matches = set()
            ##check for matches between the current setnence and all the features
            #compute the embedding of the sentence
            emb_sentence = model.encode(sentence)
            #for each feature embedding
            for feature_sentence in candidate_features:
                ####print("\n current feature: ", feature_sentence)
                #compute the embedding of the feature
                emb = model.encode(feature_sentence)
                #compute the cosine similariy between the feature and the sentence embeddings
                similarity = util.cos_sim(emb_sentence, emb)
                ####print(float(similarity))
                #if the similarity is bigger then a threshold
                if similarity>0.5:
                    #you have a match, add it to the overall matches
                    matches.add(features_sentences_dict[feature_sentence])
                    #and take it out the candidate features
                    candidate_features.remove(feature_sentence)
            ##once iterated over a certain sub-sentence, add the (k,v) pair to valuedict
            #if there were no matches
            if not matches:
                #use an int key
                value_dict[empty_counter] = sentence
                empty_counter += 1
            #o/w
            else:
                #use the tuple with the found features as key
                value_dict[tuple(matches)] = sentence
        #add the (key, value_dict) pair to the final dict
        final_dict[paragraph_title] = value_dict
    #return the final dict
    return final_dict