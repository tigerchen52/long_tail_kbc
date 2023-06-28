
candidate_templates = {
    "performer": ['the song {a} is performed by which person?'],
    "composer": ['the song {a} is composed by which person?'],
    'founded by': ['the business {a} is founded by which person?'],
    "residence": ['the person {a} lived in which place?'],
    "place of birth": ['the person {a} was born in which place?'],
    "place of death": ['the person {a} died in which place?'],
    "employer": ["the person {a} worked in which place?"],
    "educated at": ["the person {a} graduated from which place?"]
}

corroboration_templates = {
    'performer': ['the song {a} is performed by [ENT] this person [ENT]'],
    'composer': ['the song {a} is composed by [ENT] this person [ENT]'],
    'founded by': ['the business {a} is founded by [ENT] this person [ENT]'],
    'residence': ['the person {a} lived in [ENT] this place [ENT]'],
    'place of birth': ['the person {a} was born in [ENT] this place [ENT]'],
    'place of death': ['the person {a} died in [ENT] this place [ENT]'],
    'employer': ['the person {a} worked in [ENT] this place [ENT]'],
    'educated at': ['the person {a} graduated from [ENT] this place [ENT]']
}

file_names = {
    'performer': 'song',
    'composer': 'song',
    'founded by': 'business',
    'residence': 'person',
    'place of birth': 'person',
    'place of death': 'person',
    'employer': 'person',
    'educated at': 'person',

}
