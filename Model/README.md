# How to run?

## Train a new model
1. Make dirs for tf datasets
```
mkdir tfdatasets
mkdir tfdatasets/train
mkdir tfdatasets/test
```
The above commands will convert npy to a checkpoint and make empty directories for generating datasets
2. Make sure you have downloaded imagenet train and val with the development kit. Both the train and val path must have all the images. There should be no folders.
Then run the following
```
python3 tfdata.py <train_path> <val_path> <validation_ground_truth_text> imagenet_synset.txt
```

3. Train it using the following
```
python3 vgg16_train.py --model_dir <> --epochs <> --learn_rate <> --batch_size <>
```

4. Export using the following
```
python3 freeze.py --start_checkpoint <>
```

## Continue training an old model
1. Get the pretrained npy from [here.](https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM) This was taken from the repository: https://github.com/machrisaa/tensorflow-vgg
2. Then run the following 
```
python3 npy_to_ckpt.py
```
3. Repeat the new model steps but in training also add ```--start_checkpoint <>```
