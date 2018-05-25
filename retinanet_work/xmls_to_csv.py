import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import sys
import argparse
from pdb import set_trace as t

def xml_to_csv(path):
    xml_list = []
    xmls_path = os.path.join(path, '*.xml')
    for xml_file in glob.glob(xmls_path):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):

            # have to fix because xmls were made when files were png
            # so changing filename from png to jpg
            filename = root.find('filename').text
            if filename[-3:] == 'png':
                new_filename = 'images/' + filename[:-3] + 'jpg'
            # take this is out if not the case

            value = (new_filename,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text),
                     member[0].text,
                     )
            xml_list.append(value)
    xml_df = pd.DataFrame(xml_list)
    return xml_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml_dir')
    parser.add_argument('--csv_save_dir')
    args = parser.parse_args()

    xml_df = xml_to_csv(args.xml_dir)
    csv_write_path = os.path.join(args.csv_save_dir, 'annotations.csv')
    xml_df.to_csv(csv_write_path, index=None, header=False)
    print('Successfully converted xmls to csv saved at ' + csv_write_path)
