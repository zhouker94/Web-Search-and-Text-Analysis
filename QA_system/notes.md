#Question Answering

- Def: automatically determining the answer (set) for a natural language questing

- Primarry approaches:
  — NLIB (Natural Language Interface to Database): automatically construct a query, and answer question relative to fixed Knowledge Bases

  — Direct text matching: answer question via string/text passage in a doc


## Knowledge-rich Restricted-domain QA (1950-1980)

*Focus on NL interface to knowledge base from a particular domain with idiosyncratic "data semantics"*

NL query -> formal representation that can be used to query knowledge base directly

Biggest issues:

- particular domain; doesn't generalise
- scalability


## QA as IR (1999-)
"user information needs":
    - Informational
    - Navigational
    - Transactional
    - Mixed-mode information needs

TREC QA: introduced QA as a "track".

IR approach:

- query the document collection with question q_i, and return document ranking D_i
- analyse q_i to determine the expected answer type
- perform some shallow parsing/NER over (a snippet of) d_i,j to find entities of type t(q_i), and rerank D_i

(Pure IR) combine TF with:

- synonym
- rank
- IDF
- exact match for "important" words
- proximity
- heading match score
- phrasal match

Early commercial-scale QA engines (semi-automatic):

- data curation
- database lookup


## QA as IE (2005-)

IE **components** of a TREC-style system:

- question classification

- entity recognition

### QUESTION CLASSIFICATION

**Task**: predict the entity type of the answer based on wording of the question

Example question classification: {ABBREVIATION, ENTITY, DESCRIPTION, HUMAN, LOCATION, NUMERIC VALUE} (e.g. {city, country, mountain, state, other} -> LOCATION)
Feature engineering-based discriminative model, with features including:

#### Example approach 1:
- BOW features (unigrams, bigrams, trigrams)
- Positional n-gram features
- Quoted n-grams
- POS, chunk, CCG supertag and NE features
- "Target" word features, based on CCG parse tree (word, POS, chunk, NE, …) + positional n-grams

#### Example approach 2:

Deep learning model (the features are automatically constructed)

- Deep CNN over word embeddings

### SEMANTIC PARSING

Automatically translate a NL text into a formal meaning representation

#### Basic approach:

parser model produces a formal meaning representation:

- supervised learning (over query-MR pairs)
- Alignment between some intermediate representation (e.g. dependency graphs) and the query language

Use constraints of query language, and possibly analysis of query results, to prune / refine the parser output

## QA as Deep Learning (2014-)

String-string QA in an end-to-end deep learning architecture:

- "Quiz bowl" QA
- "Episodic" QA

Not currently scalable as a full-on IR solution; focuses on answering questions relative to text passages which contain the answer, answer set ranking

### Example Deep QA Dataset: SQuAD

[Rajpurkar et al., 2016]