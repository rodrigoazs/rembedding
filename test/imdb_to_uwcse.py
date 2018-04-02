#import os, sys
#sys.path.insert(1, os.path.join(sys.path[0], '..'))
#from rembedding.rembedding import REmbedding
import re

source_settings = '''workedunder(person,person).
female(person).
movie(movie,person).
genre(person,genre).
actor(person).
director(person).
'''

source = REmbedding()
target = REmbedding()

lines = source_settings.split('\n')
s = {}
for line in lines:
    m = re.search('^(\w+)\(([\w, ]+)*\).$', line)
    if m:
        relation = m.group(1).replace(' ', '')
        entities = m.group(2).replace(' ', '').split(',')
        s[relation] = entities
        
source.load_settings(s)

s = []
with open('imdb.pl') as f:
    for line in f:
        m = re.search('^(\w+)\(([\w, ]+)*\).$', line)
        if m:
            relation = m.group(1).replace(' ', '')
            entities = m.group(2).replace(' ', '').split(',')
            s.append((relation, entities))
            
source.load_dataset(s)

target_settings = '''professor(person).
student(person).
hasposition(person,faculty).
taughtby(course,person).
advisedby(person,person).
tempadvisedby(person,person).
ta(course,person).
publication(title,person).
'''

lines = target_settings.split('\n')
sett = {}
for line in lines:
    m = re.search('^(\w+)\(([\w, ]+)*\).$', line)
    if m:
        relation = m.group(1).replace(' ', '')
        entities = m.group(2).replace(' ', '').split(',')
        sett[relation] = entities
        
target.load_settings(sett)

s = []
with open('uwcselearn.pl') as f:
    for line in f:
        m = re.search('^(\w+)\(ai,([\w, ]+)*\).$', line)
        if m:
            relation = m.group(1).replace(' ', '')
            entities = m.group(2).replace(' ', '').split(',')
            if relation in sett:
                s.append((relation, entities))
            
target.load_dataset(s)

source.generate_sentences()
target.generate_sentences()
source.run_embedding()
target.run_embedding()
source.plot_2d(color={'person': 'r', 'movie': 'b', 'genre':'g'}, plot_centroid=True)
target.plot_2d(color={'person': 'r', 'faculty': 'b', 'course': 'g', 'title': 'p'}, plot_centroid=True)