# Okrol-Image-Model
An image model created on the base of okrol model by me
<br>
![diffusion_final_image_id_example1 json_0_Sat May 25 14:21:57 2024](https://github.com/Okerew/okral-image-model/assets/93822247/f327b6bc-9e00-4d88-8f0f-ee84ae9dd51f)
![diffusion_final_image_id_example1 json_1_Sat May 25 14:35:44 2024](https://github.com/Okerew/okral-image-model/assets/93822247/a71fa222-1cef-4460-b520-7537a09bff62)
______________________
Installation
------------------
Install the requirements `pip install -r requirements.txt`, 
Install the language processer `python -m spacy download en_core_web_sm`
______________
Usage
-------------------
The model is not really great and shouldn't be used for anything else than experimenting:
1. It losses some data from epoches
2. It is unstable
3. It bugs out very often
4. The diffusion algorithm is not very good
_____________________
How to add data
-------------------
This is a blueprint for example data
``` json
[
    {
        "text": "A cat sitting on a couch.",
        "image": "images/cat1.jpg"
    }
]

```
#Note in this example images folder will be in the root folder not the traing_data folder
Ai was used for generating docstrings/comments
_________________
