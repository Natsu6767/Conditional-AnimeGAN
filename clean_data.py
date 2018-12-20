import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', default='data/tags_clean.csv', help='Path to csv data file.')
parser.add_argument('-save_path', default='data/clean_labels.csv', help="Path to save cleaned data.")

args = parser.parse_args()

with open(args.save_path, 'w') as g:
    writer = csv.writer(g)
    writer.writerow(["image_name", "eyes", "hair"])

    with open(args.load_path, 'r') as f:
        for ridx, row in enumerate(csv.reader(f)):
            tags = row[1].split('\t')
            attrib = {'eyes': '<UNK>', 'hair': '<UNK>'}

            for t in tags:
                if t != '':
                    tag = t.split(':')[0].strip()
                    s_tag = tag.split()
                    
                    if len(s_tag) != 2:
                        continue

                    w = s_tag[1]
                    if (s_tag[0] == 'long' or s_tag[0] == 'short'):
                        continue
                    if(w == 'eyes' or w == 'hair'):
                        attrib[w] = s_tag[0]
            
            writer.writerow([ridx, attrib['eyes'], attrib['hair']])

