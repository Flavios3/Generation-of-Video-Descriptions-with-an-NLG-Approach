def extract_info(custom_nlp, text):
    doc = custom_nlp(text)
    feat_list = [(ent.text, ent.label_) for ent in doc.ents]
    att_list = []
    for element in feat_list:
        if element[1] == "ATTR":
            att_list.append(remove_special_ch(element[0]))
    return list(set(att_list))


def remove_special_ch(text):
    if text.find("(") and text.find(")"):
        pass
    else:
        try:
            text = text.replace('(', "")
        except:
            text = text.replace(')', "")
    return text


def df2text(df, output):
    err_cnt = 0
    with open(output, 'w') as f:
        for index, row in df.iterrows():
            if index % 100000 == 0:
                print("Starting...")
            name = row['name']

            description = row.description
            try:
                attributes = row['att_list']

                res = "<OVERV_START> <NAME_START> " + name + " <NAME_END> <FEAT_START> " + \
                      " <NEXT_FEAT> ".join(attributes) + " <FEAT_END> <DESCR_START> " + \
                      description + " <DESCR_END> <OVERV_END>"

                f.write("{}\n".format(res))

            except Exception as e:
                # print(e)
                err_cnt += 1
    print(err_cnt)
