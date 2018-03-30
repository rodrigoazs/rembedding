"""Implementation of the Relational Embedding.


"""

from gensim.models import Word2Vec
import random

class REmbedding(object):
    def __init__(self):
        self.dataset = []
        self.settings = []
        self.sentences = []
        self.graph = self.Graph()

    # {'parent': ['person', 'person'] }
    def load_settings(self, st):
        self.settings = st
    
    # [('parent', ['alexis','rodrigo'])]
    def load_dataset(self, st):
        self.dataset = st
        for tupl in self.dataset:
            type1 = self.settings[tupl[0]][0]
            type2 = self.settings[tupl[0]][1] if len(tupl[1]) > 1 else self.settings[tupl[0]][0]
            self.graph.add_relation(tupl[1][0], type1, tupl[0], tupl[1][1] if len(tupl[1]) > 1 else tupl[1][0], type2)
            
    def generate_sentences(self, max_depth=10, n_sentences=1000000, allow_revisit=False):
        self.sentences = []
        for i in range(10000000):
            node = self.graph.nodes[random.choice(list(self.graph.nodes))]
            sentence = [str(node)]
            i_depth = 1
            while(i_depth < max_depth):
                if len(node.edges) == 0:
                    break
                edge = random.choice(list(node.edges))
                sentence.append(str(edge[0]))
                sentence.append(str(edge[1]))
                node = edge[1]
                i_depth += 1
            self.sentences.append(sentence)

    class Graph(object):
        def __init__(self):
            self.nodes = {}
    
        def add_relation(self, subject, type1, relation, object_, type2):
            if subject not in self.nodes:
                self.nodes[type1 + '_' + subject] = self.Node(subject, type1)
            if object_ not in self.nodes:
                self.nodes[type1 + '_' + object_] = self.Node(object_, type2)
            self.nodes[type1 + '_' + subject].add_edge(relation, self.nodes[type2 + '_' + object_])
            
        class Node(object):
            def __init__(self, name, type_):
                self.name = name
                self.type = type_
                self.edges = set()
            
            def add_edge(self, relation, node):
                self._add_edge(relation, node)
                node._add_edge('_'+relation, self)
            
            def _add_edge(self, relation, node):
                self.edges.add((relation, node))
        
            def __str__(self):
                return str(self.type) + '_' + str(self.name)
                
            def __hash__(self):
                return hash(self.type + '_' + self.name)
                
            def __eq__(self, other):
                return str(self) == str(other)
        
settings = '''parent(person,person).
male(person).
grandmother(person,person).'''
   
data = '''parent(bart,stijn).
parent(bart,pieter).
parent(luc,soetkin).
parent(willem,lieve).
parent(willem,katleen).
parent(rene,willem).
parent(rene,lucy).
parent(leon,rose).
parent(etienne,luc).
parent(etienne,an).
parent(prudent,esther).

parent(katleen,stijn).
parent(katleen,pieter).
parent(lieve,soetkin).
parent(esther,lieve).
parent(esther,katleen).
parent(yvonne,willem).
parent(yvonne,lucy).
parent(alice,rose).
parent(rose,luc).
parent(rose,an).
parent(laura,esther).

male(bart).
male(etienne).
male(leon).
male(luc).
male(pieter).
male(prudent).
male(rene).
male(stijn).
male(willem).

grandmother(esther,soetkin).
grandmother(esther,stijn).
grandmother(esther,pieter).
grandmother(yvonne,lieve).
grandmother(yvonne,katleen).
grandmother(alice,luc).
grandmother(alice,an).
grandmother(rose,soetkin).
grandmother(laura,lieve).
grandmother(laura,katleen).'''

rembedd = REmbedding()

import re
lines = settings.split('\n')
s = {}
for line in lines:
    m = re.search('^(\w+)\(([\w, ]+)*\).$', line)
    if m:
        relation = m.group(1).replace(' ', '')
        entities = m.group(2).replace(' ', '').split(',')
        s[relation] = entities

rembedd.load_settings(s)

lines = data.split('\n')
s = []
for line in lines:
    m = re.search('^(\w+)\(([\w, ]+)*\).$', line)
    if m:
        relation = m.group(1).replace(' ', '')
        entities = m.group(2).replace(' ', '').split(',')
        s.append((relation, entities))
        
rembedd.load_dataset(s)
rembedd.generate_sentences()