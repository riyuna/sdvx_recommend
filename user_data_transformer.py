import csv
song_id_map = dict()
with open('data/song_id.csv', encoding='utf-8') as f:
    parser = csv.reader(f)
    for row in parser:
        if row[1]!='song_id':
            song_id_map[row[0]]=int(row[1])
print(song_id_map)

with open('data/song_data.csv', encoding='utf-8') as f:
    parser = csv.reader(f)
    #todo