#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 13:35:25 2023

@author: nlory
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
import tensorflow_addons as tfa


def translate_class_index(class_index):
    class_names = ['A73egs-p', 'Cunwcb-y', 'Istiophorus platypterus', 'P1rozc-z', 'Pqv7dp-s', 'Acanthaluteres brownii', 'Acanthaluteres spilomelanurus', 'Acanthaluteres vittiger', 'Acanthistius cinctus', 'Acanthopagrus australis', 'Acanthopagrus berda', 'Acanthopagrus latus', 'Achoerodus gouldii', 'Achoerodus viridis', 'Acreichthys tomentosus', 'Aesopia cornuta', 'Aethaloperca rogaa', 'Alectis ciliaris', 'Alectis indica', 'Alepes kleinii', 'Aluterus monoceros', 'Aluterus scriptus', 'Amanses scopas', 'Anampses caeruleopunctatus', 'Anampses elegans', 'Anampses femininus', 'Anampses geographicus', 'Anampses lennardi', 'Anampses melanurus', 'Anampses meleagrides', 'Anampses neoguinaicus', 'Anampses twistii', 'Anodontostoma chacunda', 'Anyperodon leucogrammicus', 'Aphareus furca', 'Aphareus rutilans', 'Aprion virescens', 'Argyrops spinifer', 'Aseraggodes melanostictus', 'Atractoscion aequidens', 'Atule mate', 'Auxis rochei', 'Auxis thazard', 'Bathylagichthys greyae', 'Beryx decadactylus', 'Bodianus anthioides', 'Bodianus axillaris', 'Bodianus bilunulatus', 'Bodianus bimaculatus', 'Bodianus diana', 'Bodianus loxozonus', 'Bodianus mesothorax', 'Bodianus perditio', 'Bodianus unimaculatus', 'Bodianus vulpinus', 'Bothus mancus', 'Bothus myriaster', 'Bothus pantherinus', 'Brachaluteres jacksonianus', 'Brachirus orientalis', 'Caesioperca lepidopterus', 'Cantherhines dumerilii', 'Cantherhines fronticinctus', 'Cantherhines pardalis', 'Cantheschenia grandisquamis', 'Caprodon longimanus', 'Caprodon schlegelii', 'Carangoides caeruleopinnatus', 'Carangoides chrysophrys', 'Carangoides equula', 'Carangoides ferdau', 'Carangoides fulvoguttatus', 'Carangoides hedlandensis', 'Carangoides malabaricus', 'Carangoides orthogrammus', 'Carangoides plagiotaenia', 'Caranx ignobilis', 'Caranx lugubris', 'Caranx melampygus', 'Caranx sexfasciatus', 'Carcharhinus albimarginatus', 'Carcharhinus amblyrhynchos', 'Carcharhinus falciformis', 'Carcharhinus galapagensis', 'Carcharhinus limbatus', 'Carcharhinus melanopterus', 'Carcharhinus obscurus', 'Carcharhinus plumbeus', 'Carcharhinus sorrah', 'Centroberyx affinis', 'Centrogenys vaigiensis', 'Centroscymnus coelolepis', 'Cephalopholis argus', 'Cephalopholis boenak', 'Cephalopholis cyanostigma', 'Cephalopholis formosa', 'Cephalopholis igarashiensis', 'Cephalopholis leopardus', 'Cephalopholis microprion', 'Cephalopholis miniata', 'Cephalopholis sexmaculata', 'Cephalopholis sonnerati', 'Cephalopholis spiloparaea', 'Chascanopsetta lugubris', 'Cheilinus chlorourus', 'Cheilinus fasciatus', 'Cheilinus oxycephalus', 'Cheilinus trilobatus', 'Cheilinus undulatus', 'Cheilio inermis', 'Cheilodactylus ephippium', 'Cheilodactylus fuscus', 'Cheilodactylus spectabilis', 'Cheilodactylus vestitus', 'Chelidonichthys kumu', 'Chirocentrus dorab', 'Chirocentrus nudus', 'Choerodon anchorago', 'Choerodon cauteroma', 'Choerodon cyanodus', 'Choerodon fasciatus', 'Choerodon graphicus', 'Choerodon jordani', 'Choerodon rubescens', 'Choerodon schoenleinii', 'Choerodon venustus', 'Choerodon vitta', 'Choerodon zamboangae', 'Chromileptes altivelis', 'Cirrhilabrus bathyphilus', 'Cirrhilabrus condei', 'Cirrhilabrus cyanopleura', 'Cirrhilabrus exquisitus', 'Cirrhilabrus laboutei', 'Cirrhilabrus punctatus', 'Cirrhilabrus scottorum', 'Cirrhilabrus temminckii', 'Coris aygula', 'Coris batuensis', 'Coris bulbifrons', 'Coris caudimacula', 'Coris dorsomacula', 'Coris gaimard', 'Coris picta', 'Coris pictoides', 'Coris sandeyeri', 'Crenimugil crenilabis', 'Cymbacephalus nematophthalmus', 'Cymolutes praetextatus', 'Cymolutes torquatus', 'Cynoglossus puncticeps', 'Cyttopsis rosea', 'Dactylophora nigricans', 'Decapterus macrosoma', 'Decapterus russelli', 'Diproctacanthus xanthurus', 'Dotalabrus aurantiacus', 'Elagatis bipinnulata', 'Epibulus insidiator', 'Epinephelus areolatus', 'Epinephelus bleekeri', 'Epinephelus chlorostigma', 'Epinephelus coeruleopunctatus', 'Epinephelus coioides', 'Epinephelus corallicola', 'Epinephelus cyanopodus', 'Epinephelus epistictus', 'Epinephelus fasciatus', 'Epinephelus fuscoguttatus', 'Epinephelus hexagonatus', 'Epinephelus howlandi', 'Epinephelus lanceolatus', 'Epinephelus latifasciatus', 'Epinephelus macrospilos', 'Epinephelus maculatus', 'Epinephelus melanostigma', 'Epinephelus merra', 'Epinephelus morrhua', 'Epinephelus multinotatus', 'Epinephelus ongus', 'Epinephelus polyphekadion', 'Epinephelus quoyanus', 'Epinephelus radiatus', 'Epinephelus retouti', 'Epinephelus rivulatus', 'Epinephelus sexfasciatus', 'Epinephelus spilotoceps', 'Epinephelus tauvina', 'Epinephelus undulatostriatus', 'Etelis carbunculus', 'Etelis coruscans', 'Eubalichthys cyanoura', 'Eubalichthys mosaicus', 'Eupetrichthys angustipes', 'Euthynnus affinis', 'Evistias acutirostris', 'Gempylus serpens', 'Gnathanodon speciosus', 'Gnathodentex aureolineatus', 'Gracila albomarginata', 'Gymnocranius audleyi', 'Gymnocranius euanus', 'Gymnocranius grandoculis', 'Gymnocranius microdon', 'Gymnosarda unicolor', 'Halichoeres argus', 'Halichoeres biocellatus', 'Halichoeres chloropterus', 'Halichoeres chrysus', 'Halichoeres hartzfeldii', 'Halichoeres hortulanus', 'Halichoeres leucurus', 'Halichoeres margaritaceus', 'Halichoeres marginatus', 'Halichoeres melanochir', 'Halichoeres melanurus', 'Halichoeres melasmapomus', 'Halichoeres miniatus', 'Halichoeres nebulosus', 'Halichoeres nigrescens', 'Halichoeres scapularis', 'Halichoeres trimaculatus', 'Harriotta raleighana', 'Hemigymnus fasciatus', 'Hemigymnus melapterus', 'Hemiramphus far', 'Herklotsichthys quadrimaculatus', 'Hologymnosus annulatus', 'Hologymnosus doliatus', 'Hyporhamphus affinis', 'Hyporhamphus dussumieri', 'Inegocia japonica', 'Johnius borneensis', 'Katsuwonus pelamis', 'Labrichthys unilineatus', 'Labroides bicolor', 'Labroides dimidiatus', 'Labroides pectoralis', 'Labropsis australis', 'Labropsis manabei', 'Labropsis xanthonota', 'Latridopsis forsteri', 'Lepidocybium flavobrunneum', 'Leptojulis cyanopleura', 'Lethrinus amboinensis', 'Lethrinus atkinsoni', 'Lethrinus erythracanthus', 'Lethrinus genivittatus', 'Lethrinus harak', 'Lethrinus lentjan', 'Lethrinus microdon', 'Lethrinus miniatus', 'Lethrinus nebulosus', 'Lethrinus obsoletus', 'Lethrinus olivaceus', 'Lethrinus ornatus', 'Lethrinus rubrioperculatus', 'Lethrinus semicinctus', 'Lethrinus variegatus', 'Lethrinus xanthochilus', 'Liopropoma mitratum', 'Liopropoma susumi', 'Liza subviridis', 'Liza vaigiensis', 'Lniistius aneitensis', 'Lniistius pavo', 'Lutjanus adetii', 'Lutjanus argentimaculatus', 'Lutjanus biguttatus', 'Lutjanus bohar', 'Lutjanus carponotatus', 'Lutjanus decussatus', 'Lutjanus ehrenbergii', 'Lutjanus erythropterus', 'Lutjanus fulviflamma', 'Lutjanus fulvus', 'Lutjanus gibbus', 'Lutjanus johnii', 'Lutjanus kasmira', 'Lutjanus lemniscatus', 'Lutjanus lutjanus', 'Lutjanus malabaricus', 'Lutjanus monostigma', 'Lutjanus quinquelineatus', 'Lutjanus rivulatus', 'Lutjanus russellii', 'Lutjanus sebae', 'Lutjanus semicinctus', 'Lutjanus semicinctus quoy', 'Lutjanus timoriensis', 'Lutjanus vitta', 'Macolor macularis', 'Macolor niger', 'Macropharyngodon choati', 'Macropharyngodon kuiteri', 'Macropharyngodon meleagris', 'Macropharyngodon negrosensis', 'Macropharyngodon ornatus', 'Megalaspis cordyla', 'Meuschenia australis', 'Meuschenia freycineti', 'Meuschenia galii', 'Meuschenia hippocrepis', 'Meuschenia scaber', 'Meuschenia trachylepis', 'Monacanthus chinensis', 'Monotaxis grandoculis', 'Mugim cephalus', 'Naucrates ductor', 'Negaprion acutidens', 'Nemadactylus douglasii', 'Nemipterus furcosus', 'Nemipterus hexodon', 'Nemipterus peronii', 'Netuma thalassina', 'Nibea soldado', 'Notolabrus fucicola', 'Notolabrus gymnogenis', 'Notolabrus tetricus', 'Notorynchus cepedianus', 'Novaculichthys taeniourus', 'Novaculoides macrolepidotus', 'Oedalechilus labiosus', 'Ophthalmolepis lineolatus', 'Otolithes ruber', 'Oxycheilinus bimaculatus', 'Oxycheilinus celebicus', 'Oxycheilinus digrammus', 'Oxycheilinus unifasciatus', 'Oxymonacanthus longirostris', 'Pagrus auratus', 'Paracaesio kusakarii', 'Paracheilinus filamentosus', 'Paraluteres prionurus', 'Paramonacanthus choirocephalus', 'Paraplagusia bilineata', 'Parastromateus niger', 'Pardachirus hedleyi', 'Pardachirus pavoninus', 'Pentapodus aureofasciatus', 'Pentapodus paradiseus', 'Pentapodus vitta quoy', 'Pervagor alternans', 'Pervagor aspricaudus', 'Pervagor janthinosoma', 'Pervagor melanocephalus', 'Pervagor nigrolineatus', 'Pinjalo lewisi', 'Platycephalus indicus', 'Plectranthias longimanus', 'Plectranthias nanus', 'Plectranthias winniensis', 'Plectropomus areolatus', 'Plectropomus laevis', 'Plectropomus leopardus', 'Plectropomus maculatus', 'Plectropomus oligacanthus', 'Plotosus lineatus', 'Pristipomoides argyrogrammicus', 'Pristipomoides auricilla', 'Pristipomoides filamentosus', 'Pristipomoides flavipinnis', 'Pristipomoides sieboldii', 'Pristipomoides zonatus', 'Promethichthys prometheus', 'Protonibea diacanthus', 'Psettodes erumei', 'Pseudalutarius nasicornis', 'Pseudanthias bicolor', 'Pseudanthias cooperi', 'Pseudanthias dispar', 'Pseudanthias fasciatus', 'Pseudanthias huchtii', 'Pseudanthias hypselosoma', 'Pseudanthias lori', 'Pseudanthias luzonensis', 'Pseudanthias pictilis', 'Pseudanthias pleurotaenia', 'Pseudanthias rubrizonatus', 'Pseudanthias sheni', 'Pseudanthias smithvanizi', 'Pseudanthias squamipinnis', 'Pseudanthias tuka', 'Pseudanthias ventralis', 'Pseudocaranx dentex', 'Pseudocarcharias kamoharai', 'Pseudocheilinus evanidus', 'Pseudocheilinus hexataenia', 'Pseudocheilinus ocellatus', 'Pseudocheilinus octotaenia', 'Pseudodax moluccanus', 'Pseudojuloides cerasinus', 'Pseudolabrus biserialis', 'Pseudolabrus guentheri', 'Pseudolabrus luculentus', 'Pseudorhombus argus', 'Pseudorhombus arsius', 'Pseudorhombus elevatus', 'Pteragogus cryptus', 'Pteragogus enneacanthus', 'Pteragogus flagellifer', 'Rastrelliger kanagurta', 'Retropinna semoni', 'Rhabdosargus sarba', 'Rhincodon typus', 'Rhizoprionodon acutus', 'Ruvettus pretiosus', 'Samaris cristatus', 'Samariscus triocellatus', 'Sarda orientalis', 'Sardinella albella', 'Sardinella gibbosa', 'Sardinops sagax', 'Scaevius milii', 'Scolopsis affinis', 'Scolopsis bilineata', 'Scolopsis lineata', 'Scolopsis margaritifer', 'Scolopsis monogramma', 'Scolopsis trilineata', 'Scolopsis vosmeri', 'Scolopsis xenochrous', 'Scomberoides commersonnianus', 'Scomberoides lysan', 'Scomberomorus commerson', 'Selar crumenophthalmus', 'Selaroides leptolepis', 'Seriola dumerili', 'Seriola hippos', 'Seriola rivoliana', 'Seriolina nigrofasciata', 'Serranocirrhitus latus', 'Sillago ciliata', 'Sillago sihama', 'Soleichthys heterorhinos', 'Sphyraena barracuda', 'Sphyraena forsteri', 'Sphyraena jello', 'Sphyraena obtusata', 'Stegostoma fasciatum', 'Stethojulis bandanensis', 'Stethojulis interrupta', 'Stethojulis strigiventer', 'Stethojulis trilineata', 'Stolephorus waitei', 'Suezichthys arquatus', 'Suezichthys cyanolaemus', 'Suezichthys gracilis', 'Symphorichthys spilurus', 'Symphorus nematophorus', 'Thalassoma amblycephalum', 'Thalassoma hardwicke', 'Thalassoma jansenii', 'Thalassoma lunare', 'Thalassoma lutescens', 'Thalassoma nigrofasciatum', 'Thalassoma purpureum', 'Thalassoma quinquevittatum', 'Thalassoma trilobatum', 'Thryssa baelama', 'Thryssa hamiltonii', 'Thunnus alalunga', 'Thunnus albacares', 'Thysanophrys celebica', 'Thysanophrys chiltonae', 'Trachichthys australis', 'Trachinotus baillonii', 'Trachinotus blochii', 'Trachinotus botla', 'Trachypoma macracanthus', 'Triaenodon obesus', 'Uraspis secunda', 'Valamugil cunnesius', 'Valamugil engeli', 'Valamugil seheli', 'Variola albimarginata', 'Variola louti', 'Wattsia mossambica', 'Wetmorella albofasciata', 'Wetmorella nigropinnata', 'Xiphocheilus typus', 'Zenarchopterus dispar', 'Zeus faber']
    class_dict = {key: idx for idx, key in enumerate(class_names)}
    key_list = list(class_dict.keys())
    val_list = list(class_dict.values())
    # print key
    position = val_list.index(class_index)
    return key_list[position]
 
def find_maxima(predictions):
    sorted_pred= np.sort(predictions)
    a = sorted_pred[0][-2:-1]
    a = list(predictions[0]).index(a)
    b = sorted_pred[0][-3:-2]
    b = list(predictions[0]).index(b)
    c = sorted_pred[0][-4:-3]
    c = list(predictions[0]).index(c)
    return translate_class_index(a), translate_class_index(b), translate_class_index(c)


def classify(model_folder, image_input, image_dim):
    model_dir = model_folder
    model = keras.models.load_model(model_dir)
    img_dim_1 =  image_dim
    img_dim_2 =  image_dim
    # Load the model
    #model = keras.models.load_model('model.h5')
    
    # Load the image and resize it to (224, 224)
    #img_dir = '/content/drive/MyDrive/colab_notebooks/data/fish_data/' + image_input
    img_dir = image_input
    
    img = image.load_img(img_dir, target_size=(img_dim_1, img_dim_2))
    
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    
    # Expand the shape of the image array from (224, 224, 3) to (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize the pixel values from [0, 255] to [-1, 1]
    img_array = (img_array / 127.5) - 1
    
    # Use the model to make a prediction
    predictions = model.predict(img_array)
    
    # Get the class with the highest prediction probability
    class_index = np.argmax(predictions[0])
    # creating a new dictionary
    # list out keys and values separately
    # next most likely predictions:
    a,b,c = find_maxima(predictions)
    # Print the predicted class
    return f"Predicted species: {translate_class_index(class_index)}, \nthe next most likely predictions are {a}, {b} or {c}"
    
def main(model_folder, image_input, image_dim):
    # Just disables the warning, doesn't take advantage of AVX/FMA to run faster
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    #model_folder = "models/" + input("Which model would you like to use? (saved model folder inside /models):")
    #image_input= 'data/fish_examples/' + input('Please specify an image file (img file inside data/fish_examples folder):')
    #image_dim = input('Please specify the image dimensions (square image):')
    #image_dim = int(image_dim)
    text = classify(model_folder, image_input, image_dim)   
    print(text)    
        
if __name__ == '__main__':
    main()
