from utils import DB_Handler
import utils
import sys
from utils_video_generation import *
from slideshow_utils import *
DEFAULT_SCHEMA = "tv"

if __name__ == "__main__":

    # code used to extract a product from MongoDB Database, given a certain id as parameter

    schema = DEFAULT_SCHEMA
    MODEL_NAME = "gpt2-medium"
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

    if len(sys.argv) < 4:

        saveFlag = True
        my_id = '5fbd0324e20dcbf0559652e7'
    # 	else:
    # 	    my_id=sys.argv[1]
    # 		schema=sys.argv[2]
    # 		saveFlag=sys.argv[3]

    if len(sys.argv) == 2:
        my_id = sys.argv[1]
    else:
        my_id = "5fbd0324e20dcbf0559652e7"
    
    # in case we want to save the output query, set the flag with True value as follows
    saveFlag = True

    # import and call the DB_Handler of utils.py to do pymongo operations
    query = DB_Handler(collection="Products", schema=schema, saveFlag=saveFlag, output="../data/")

    # retrieve the product by passing the id to it
    prods = query.select_prod(my_id)
    # usually it should be 1 element,
    # but in case id is a random number n, the function returns n products from random sampling

    for prod in prods:
        # output=query.complete_input(prod,[["Key Features","Audio"],["Special Features","Network","Display"]])
        # output=query.complete_input(prod,[["Key Features","Audio"],["Special Features","Network","Ports & Interfaces"]
        # ,["Smart TV","Performance"],["Display","Management Features"]])

        # returns an output with the following categories
        categories = [["Display", "Key Features", "Network"], ["Performance", "Ports & Interfaces"],
                      ["Audio", "Special Features"]]
        query.complete_input(prod, categories)
        # executing this command saves on the /data folder the json file with the id name
        # structured as follows: [{category : 'Display', input:<OVERV_START> <NAME_START> HISENSE H55B7500 TV <NAME_END>
        # <FEAT_START> 60Hz Native ...... <FEAT_END>}, {category : ..... and so on
        # we have one dictionary of {category,input} for each of the categories passed as input

        # Review generation
        # Recall to install all the requirements before proceeding with the code

        # load the datafile
        content = utils.load_json("../data/", prod+'.json')

        # define tokenizer and load the pretrained model
        tokenizer = utils.myTokenizer(MODEL_NAME, special_tokens)
        # the path has to be adapted to the place where the model stays
        model = utils.AutoModelWithLMHead.from_pretrained(
            '../model_medium')
        model.resize_token_embeddings(len(tokenizer))
        model.to("cuda:0")  # gpu usage

        generated_review = utils.generate(content, tokenizer, model)

        with open("review.txt","w") as text_file:
            text_file.write(generated_review)
        print(generated_review)

      
        ###

        input_features = extract_features("../data/"+prod+".json")
        paragraph_dict = extract_paragraph_dict("review.txt")
        dict_input = compute_final_dict(input_features,paragraph_dict)
        print(dict_input)
        #######

        #retrieving infos about the product from the json file
        brand_model = retrieving_structured_info("../data/"+prod+"_general.json")

        #creation of the slideshow
        images = list()
        tracks = list()
        videos = list()
        model_image_filename, logo_image_filename = create_first_slide(images,tracks,brand_model,'en')
        create_images_and_audio(dict_input, images, tracks, 'en',brand_model,model_image_filename)
        create_video(tracks,images,videos)

    