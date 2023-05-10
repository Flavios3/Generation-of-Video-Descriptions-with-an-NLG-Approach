#Importing all the useful libraries

from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
from moviepy.editor import *
from moviepy.video.fx import speedx
import numpy as np
import json
from bing_image_downloader import downloader
import os 

#################################################
#This function returns a dictionary with the brand and the model of the product we are interested in. [NOTE: Change the input path accordingly]
def retrieving_structured_info(structured_file):
    # Opening JSON file
    f = open(structured_file)

    # returns JSON object as a list
    data = json.load(f)

    #Retrieving the full dictionary from the list
    full_dictionary = data[0]

    #Creating a dictionary with all and only the needed infos
    brand_model_dictionary = dict()
    brand_model_dictionary["brand"] = full_dictionary['brand']
    brand_model_dictionary["model"] = full_dictionary['title']

    #print(brand_model_dictionary)
    # # Closing file
    f.close()
    return brand_model_dictionary

#################################################
#This function is used to download from Bing the images of the logo and the product. [NOTE: Change the output directory accordingly]
def download_images_from_bing(brand_model):
    downloader.download(brand_model['brand']+" logo", limit=1, output_dir='Images_Product', adult_filter_off=True, force_replace=False, timeout=60, verbose=True)
    downloader.download(brand_model['model'], limit=1, output_dir='Images_Product', adult_filter_off=True, force_replace=False, timeout=60, verbose=True)
#################################################

#################################################
#This function creates the image and audio for the very first slide of the slideshow. [NOTE: the first slide of the slideshow must have the product logo and the product image together with the name of the model] [NOTE: this function uses the downloader from Bing ]
def create_first_slide(images,tracks,brand_model,language):
    
    download_images_from_bing(brand_model)

    max_h = 600
    max_w = 600
    
    #Retrieving the product name
    product_name = brand_model['model']
    brand = brand_model['brand']
    #Downloading the images of the logo and the actual product from Bing
    #download_images_from_bing(brand_model)

    path_model = "Images_Product/"+product_name
    model_image_filename = [f for f in os.listdir(path_model) if os.path.isfile(os.path.join(path_model, f))][0]
    #print(file_name)
    path_logo = "Images_Product/"+brand+" logo"
    logo_image_filename = [f for f in os.listdir(path_logo) if os.path.isfile(os.path.join(path_logo, f))][0] 

    #Opening the two images downloaded from Bing [add if to check the type of the image]
    img1 = Image.open("Images_Product/" +product_name +"/"+model_image_filename)
    img2 = Image.open("Images_Product/"+brand+" logo/"+logo_image_filename)

    #Resizing the images to a reasonable size 
    img1 = img1.resize((800,600))
    img2 = img2.resize((400,100))
    
    #Defining a Font for the title (i.e. the name of the product)
    myFont = ImageFont.truetype('arial.ttf', 80)

    #Creating the first slide of the slideshow concatanating the 2 images on the background white image. 
    new_img = Image.new('RGB', (1920,1080), 'white')
    bg_size = new_img.size

    img_size1 = img1.size
    img_size2 = img2.size
    offset1 = ((bg_size[0] - img_size1[0]) // 2, (bg_size[1] - img_size1[1]) // 2)
    offset2 = ((bg_size[0]-img_size2[0] -10 ,bg_size[1]-img_size2[1] - 10))
    new_img.paste(img1,offset1)
    new_img.paste(img2,offset2) #add ,img2 for transparency. But it gives errors sometimes

    #Drawing the name of the product on the slide
    draw = ImageDraw.Draw(new_img)
    _, _, w1, h1 = draw.textbbox((0, 0), product_name, font = myFont)
    draw.text(((bg_size[0]-w1)/2, h1), product_name, font = myFont, fill='black')

    #Storing the image 

    image_title = "images/first.png"
    new_img.save(image_title,"PNG")
    images.append(image_title)

    #Creation of audio track

    track_title = "audios/first.mp3"
    #initialize the tts method
    tts_executer = gTTS(text = "Welcome to the "+product_name+" review. Take into account that this review is generated automatically by an IA system.", lang=language, slow=False)
    #save the audio
    tts_executer.save(track_title)
    #append the audio to the list
    tracks.append(track_title)

    print("First slide created!")

    return model_image_filename,logo_image_filename
#################################################
#This function create and Image of the given size and background color, with the given title and features list
def create_image(size, bgColor, title,features_list, font_list, fontColor,brand_model,model_filename):
    
    #set the size
    W, H = size
    #create the image object
    image = Image.new('RGB', size, bgColor)
    #create the object to draw on the image
    draw = ImageDraw.Draw(image)

    #Adding the banner in each slide
    banner = Image.open("Banner.png")#.resize((1920,300))
    image.paste(banner,(0,0))
    #Adding the product image in each slide
    product = Image.open("Images_Product/" +brand_model['model'] +"/"+model_filename).resize((400,400))
    image.paste(product,(W-500,H-500))
    
    #1st, drawing the title on the slide
    #Creating the text box and filling it with the title 
    #if len(title) >= 26:
    _, _, w1, h1 = draw.textbbox((0, 0), title, font=font_list[2])
    draw.text(((W-w1)/2, h1+95), title, font=font_list[2], fill=fontColor)
    #else:
        #_ , _, w1, h1 = draw.textbbox((0, 0), title, font=font_list[0])
        #draw.text(((W-w1)/2, h1+95), title, font=font_list[0], fill=fontColor)

    #2nd, drawing the features on the slide
    k=0
    j=0
    for idx,feature_tuple in enumerate(features_list):
        k += j
        if k >= 7:
            break
        j = 0
        for feature in feature_tuple:
            print(feature)

            #if len(feature) < 46:
              #create the text box
              #_, _, w2, h2 = draw.textbbox((0, 0), feature, font=font_list[1])
              #write your text
              #draw.text((100, 350 +(k*100 + j*100)),"• " + feature, font=font_list[1], fill=fontColor)
            #else:
              #create the text box
            _, _, w2, h2 = draw.textbbox((0, 0), feature, font=font_list[3])
              #write your text
            draw.text((100, 350 +(k*100 + j*100)),"• " + feature, font=font_list[3], fill=fontColor)
            j += 1
            if j + k >= 7:
                break
    #return the image
    return image

#################################################
#This function create a slide with the given title and list of features. Note that this function use the create_image function, in addition to that it has only the definition of the fonts, one for the title and one for the features and the overall size of the slide. 
def create_slide(title,features_list,brand_model,model_filename):
    #set the font and it's dimension
    myFont1 = ImageFont.truetype('arial.ttf', 120)
    myFont2 = ImageFont.truetype('arial.ttf', 60)
    myFont3 = ImageFont.truetype('arial.ttf', 80)
    myFont4 = ImageFont.truetype('arial.ttf', 40)
    #set the resolution of the slide
    mySize = (1920,1080)
    #call the funtion that creates the slide
    return create_image(mySize, "white", title,features_list, [myFont1,myFont2,myFont3,myFont4], "black",brand_model,model_filename)

#################################################
#This function uses the create_slide function. This function creates all the audio tracks and images useful for the slideshow. In particular, given a paragraph and its content, the function creates an image and an audio track for each subset of features contained in each sentence of the paragraph's content. The paths of the images and audio created are stored in two list, respectively "images" and "tracks". 
def create_images_and_audio(paragraph_dict, images, tracks, language,brand_model,model_filename):
    #iterate over the dict
    i = 0
    for paragraph_title,v in paragraph_dict.items():
        features = []
        for features_tuple,text in v.items():
            
            if type(features_tuple) != int:
                features.append(features_tuple)
            ##create a slide
            image_title = "images/"+str(i)+".png"
            #create the image
            myImage = create_slide(paragraph_title,features,brand_model,model_filename)
            #save the image as a PNG
            myImage.save(image_title,"PNG")
            #add the image to the list
            images.append(image_title)

            #create the audio
            #set the audio name
            track_title = "audios/"+str(i)+".mp3"
            #initialize the tts method
            tts_executer = gTTS(text = text, lang=language, slow=False)
            #save the audio
            tts_executer.save(track_title)
            #append the audio to the list
            tracks.append(track_title)
            i+=1

#################################################
#This function create a "sub video", meaning that it creates a single frame for the given image and given audio. The result is stored inside a list "videos". [NOTE: the single sub-videos are then concatenated in another function]
def create_sub_video(image_path, audio_path, output_path, videos):
    # Import the audio(Insert to location of your audio instead of audioClip.mp3)
    audio = AudioFileClip(audio_path)
    # Import the Image and set its duration same as the audio (Insert the location of your photo instead of photo.jpg)
    clip = ImageClip(image_path).set_duration(audio.duration)
    # Set the audio of the clip
    clip = clip.set_audio(audio)

    #Appending clips to the list
    videos.append(clip)
    # Export the clip  MAYBE NOT NECESSARY, CHECK
    clip.write_videofile(output_path, fps=24)

#################################################
#This function creates the final slideshow. Given the set of tracks and images it creates the final slideshow.[NOTE: this function uses the create_sub_video function] [NOTE: change accordingly the paths]
def create_video(tracks, images, videos):

    ##create the sub videos
    i = 0
    #for each (slide, audio) pair
    for _ in tracks:
        #set the sub video name
        video_path = "videos/"+str(i)+".mp4"
        #create the sub video
        create_sub_video(images[i], tracks[i], video_path, videos)
        i += 1

    #concatenate them
    # concatenating both the clips:
    #final_video = concatenate([sub_video.crossfadein(2) for sub_video in videos],padding = -1, method = "compose")
    final_video = concatenate_videoclips(videos)
    final_video = final_video.speedx(factor = 1)

    #Adding a background music to the final video
    duration_video = final_video.duration
    music_clip = AudioFileClip("sunset-vibes-lo-fichillhop-9503.mp3")

    #Looping the music till the duration of the video
    looped_music = afx.audio_loop(music_clip,duration = duration_video + 3)
    looped_music = afx.audio_fadeout(looped_music, duration = 3)
    looped_music = afx.volumex(looped_music, factor=0.1)
    final_audio = CompositeAudioClip([looped_music, final_video.audio])
    final_video_with_music = final_video.set_audio(final_audio)
    final_video_with_music.write_videofile("output/final_video.mp4", fps=24) 


