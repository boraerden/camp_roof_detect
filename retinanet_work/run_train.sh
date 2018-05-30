python ~/keras-retinanet/keras_retinanet/bin/train.py \
--gpu 0 \
--batch-size 1 \
--epochs 5 \
--steps 2000 \
--tensorboard-dir ~/retinanet_models/tensorboard \
--snapshot-path ~/retinanet_models/snapshots \
--random-transform \
csv \
~/data/train_annotations.csv \
~/data/classes.csv \
--val-annotations ~/data/val_annotations.csv \


# 
