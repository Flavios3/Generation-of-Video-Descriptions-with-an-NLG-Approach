from utils_custom_ner import *
from utils_test import *


if __name__ == "__main__":
    input_dir = "data/"
    output_dir = "model/"

    # set up for the evaluation process
    tag_data = load_data(input_dir + "ner_dataset.txt")
    raw_data = load_data(input_dir + "ner_dataset_notags.txt")

    t = ner_Trainer()
    t.setup()
    ner_list = t.create_list_entities(tag_data, raw_data)

    my_df = []
    for i, description in enumerate(ner_list):
        attributes_pos = []
        attributes = []
        prod_pos = []
        prod = []
        for j in range(len(description[1]['entities'])):
            if description[1]['entities'][j][2] == 'ATTR':
                attributes_pos.append(description[1]['entities'][j])
                attributes.append(description[0][description[1]['entities'][j][0]:description[1]['entities'][j][1]])
            else:
                prod_pos.append(description[1]['entities'][j])
                prod.append(description[0][description[1]['entities'][j][0]:description[1]['entities'][j][1]])

        my_df.append([i, description[0], attributes, prod, attributes_pos, prod_pos])

    # simply creates the dataframe with the following columns
    df = pd.DataFrame(my_df, columns=['id', 'Description', 'Attributes', 'Product', 'Attributes_pos', "Product_pos"])

    # start evaluation process
    v = Validator(df, ['ATTR', 'PROD'])
    v.kfold_valid(10, 1)

    v.compute_stats()
    v.df_track['correct+partial'] = v.df_track['correct'] + v.df_track['partial']
    v.df_track['correct2+partial'] = v.df_track['correct2'] + v.df_track['partial']

    # attribute mean value
    attr_mean = v.df_track.loc[v.df_track['tag'] == 'ATTR'].mean(numeric_only=True)
    print(f'attribute mean values: \n{attr_mean}')

    # attribute standard deviation
    attr_std = v.df_track.loc[v.df_track['tag'] == 'ATTR'].std(numeric_only=True)
    print(f'attribute std values: \n{attr_std}')

    # product mean value
    prod_mean = v.df_track.loc[v.df_track['tag'] == 'PROD'].mean(numeric_only=True)
    print(f'product mean values: \n{prod_mean}')

    # product std value
    prod_std = v.df_track.loc[v.df_track['tag']=='PROD'].std(numeric_only=True)
    print(f'product mean values: \n{prod_std}')

    print(f'\n\nSimple test:\n')
    text = "Redmi 9A comes with Octa-core Helio G25 processor and upto 2.0GHz clock speed. It also comes with 13 MP AI Rear camera along with 5 MP front camera. Redmi 9A also features 16.58 centimeters (6.53-inch) HD + display with 720x1600 pixels. It also comes with 5000 mAH large battery."

    ner_tester = ner_Tester("model/")
    ner_tester.test(text)
