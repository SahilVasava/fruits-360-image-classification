# -*- coding: utf-8 -*-

!pip install -Uqq fastbook
import fastbook
fastbook.setup_book()

# !pip install -U -q kaggle
# !mkdir -p ~/.kaggle

# from google.colab import files
# files.upload()

# !cp kaggle.json ~/.kaggle/

# !kaggle datasets list

# !kaggle datasets download -d moltean/fruits -p /content/gdrive/MyDrive/datasets

# !unzip /content/gdrive/MyDrive/datasets/fruits.zip -d /content/gdrive/MyDrive/datasets/fruits

from fastbook import *

from fastai.vision.all import *

path =  Path('/content/gdrive/MyDrive/datasets/fruits/fruits-360')
Path.BASE_PATH = path

fns = get_image_files(path)
fns

(path/'Training/Corn').ls()

fruits = DataBlock(
    blocks = (ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSubsetSplitter(0.2, 0.05),
    get_y = parent_label,
    batch_tfms=aug_transforms(min_scale=0.75)
)
dsets = fruits.datasets(path/'Training')
dls = fruits.dataloaders(path/'Training')

dsets.train, dsets.valid

len(dsets.vocab)

dsets.vocab

fruits.summary(path/'Training')

dls.train.show_batch(max_n=8, nrows=2, unique=True)

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(2)

learn.export('/content/gdrive/MyDrive/fruits1.1.pkl')

learn = cnn_learner(dls, resnet34, metrics=error_rate)
lr_min,lr_steep = learn.lr_find()

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(2, base_lr=3e-3)

learn.export('/content/gdrive/MyDrive/fruits2.1_lrfind.pkl')

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fit_one_cycle(3, 3e-3)

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fit_one_cycle(3, 3e-3)

learn.unfreeze()

learn.lr_find()

learn.fit_one_cycle(8, lr_max=1e-5)

learn.export('/content/gdrive/MyDrive/fruits3.1_customFine_lrfind.pkl')

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fit_one_cycle(3, 3e-3)
learn.unfreeze()
learn.fit_one_cycle(12, lr_max=slice(1e-6,1e-4))

learn.recorder.plot_loss()

learn.export('/content/gdrive/MyDrive/fruits4.1_customFine_dlr.pkl')

from fastai.callback.fp16 import *
learn = cnn_learner(dls, resnet50, metrics=error_rate).to_fp16()
learn.fine_tune(6, freeze_epochs=3)

from fastai.callback.fp16 import *
learn = cnn_learner(dls, resnet50, metrics=error_rate).to_fp16()
learn.fine_tune(6, freeze_epochs=3)

learn.export('/content/gdrive/MyDrive/fruits5.1_resnet50.pkl')

learn = cnn_learner(dls, resnet50, metrics=error_rate).to_fp16()
lr_min,lr_steep = learn.lr_find()

learn = cnn_learner(dls, resnet50, metrics=error_rate).to_fp16()
learn.fine_tune(2, base_lr=3e-3)

learn = cnn_learner(dls, resnet50, metrics=error_rate).to_fp16()
learn.fit_one_cycle(3,3e-3)

learn.unfreeze()

learn.lr_find()

learn.fit_one_cycle(6, lr_max=1e-5)

learn = cnn_learner(dls, resnet50, metrics=error_rate).to_fp16()
learn.fit_one_cycle(3, 3e-3)
learn.unfreeze()
learn.fit_one_cycle(12, lr_max=slice(1e-6,1e-4))

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)

interp.most_confused(min_val=5)

interp.plot_top_losses(4, nrows=2)

