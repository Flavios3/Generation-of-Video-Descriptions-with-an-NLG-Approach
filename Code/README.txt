To run the code:
- open the given colab notebook file.
- upload  the "other_files" and "model_medium" folders to the colab runtime. 


The "other_files" folder contains:
- the .py files where all the support functions are defined (one for the text generation "utils.py", one for the distance module "utils_video_generation.py" and one for the video generation "slideshow_utilis.py")
- the main function "main.py" calls all the necessary functions for the generation of the slideshow starting from a given input (id of a product on the database).
- several .txt files used in the video generation

Note:
- To run the code please change the current directory in colab to the "other_files" folder. Once done that, you can call the main function through colab passing to it a valid product id. If the id is not specified a default id is considered for the generation.
- Directories used to store all the images/audios/videos will be created before calling the main function through a dedicated cell. 
- The GPT2_final and NER_final directories contain all the data and code used for the training of the GPT-2 and NER respectively. 
- The "model_medium" is not present because it would cause the slack's intended limit of 1GB to be exceeded (Please, if you want to run the code ask for it or just train the model). 