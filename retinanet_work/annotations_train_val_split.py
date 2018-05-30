import csv
from sklearn.model_selection import train_test_split
from pdb import set_trace as t

annotations_csv_path = 'data/annotations.csv'
train_csv_path = 'data/train_annotations.csv'
val_csv_path = 'data/val_annotations.csv'
 
annotations = []
 
with open(annotations_csv_path) as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        annotations.append(row)


train_annotations, val_annotations = train_test_split(annotations, test_size = 0.05)


with open(train_csv_path, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(train_annotations)
    
with open(val_csv_path, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(val_annotations)