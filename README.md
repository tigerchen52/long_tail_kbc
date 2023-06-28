# Knowledge Base Completion for Long-Tail Entities
Recent work has shown that pre-trained language models store a large number of relational facts, and then propose "Language Models as Knowledge Bases" [1].  
Existing approaches use cloze-style prompts to query masked language models.
However, they cannot cope with multi-token facts well and suffer from the long-tail issue.
This paper devises a novel method for knowledge base completion (KBC), specifically geared to cope with long-tail entities.
<p align="center">
<img src="figure/framework.png" width="900">
</p>
Our method leverages Transformer-based language models in a new way. 
Most notably, we employ two different LMs in a two-stage pipeline,
as shown in the above Figure. The first stage generates candidate answers to input prompts and gives cues
to retrieve informative sentences from Wikipedia and other sources. The second stage validates (or
falsifies) the candidates and disambiguates the retained answer strings onto entities in the underlying
KG (e.g., mapping “Lhasa” to Lhasa de Sela , and “Bratsch” to Bratsch (band)).
