# Okrol-Image-Model
An image model created on the base of okrol model by me
<br>
![Screenshot 2024-05-24 at 17 49 04](https://github.com/Okerew/okral-image-model/assets/93822247/dbdd8ce5-70d8-4e80-9cc1-6b6e927b6ffa)
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
4. For now, it only generates high entropy images
_____________________
How to add data
-------------------
This is a blueprint for example data
```
[
    {
        "text": "A cat sitting on a couch.",
        "image": "images/cat1.jpg"
    }
]

```
_________________
Diffusion
--------------
For now it is not producing proper images with the diffusion algorithm it will be improved though.
