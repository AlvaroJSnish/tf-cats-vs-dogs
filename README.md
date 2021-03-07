# Cats and dogs image classifier

Python 3.8 -> to install dependencies run ``pip install -r requirements.txt``

The model creation is on ``main.py``. Inferences (predictions) with the model on ``predict.py``

All the images are stored on ``data``, download from https://www.kaggle.com/c/dogs-vs-cats/data. Train images around 9k
for each class, validation around 2.5k. Test images downloaded from unsplash. Example folder
structure: ``data/train/cats`` ``data/train/dogs``

I'm using my GPU that is CUDA compatible, if you don't have GPU or it's not compatible, comment ``main.py:11``
, ``use_gpu()``