# Knowledge Base Completion for Long-Tail Entities
In this work, we propose an unsupervised framework for knowledge base completion. The main benefits of our approach are:
* **fully prompt-based**. To extract a new relation, the only thing needed in this framework is to design a prompt.
* **can deal with multi-token and ambiguous entities**
* **work well on long-tail entities**

<p align="center">
<img src="figure/framework.png" width="900">
</p>
Our method employs two different LMs in a two-stage pipeline as shown in the above Figure. 
The first stage generates candidate answers to input prompts and gives cues to retrieve informative sentences from Wikipedia and other sources. 
The second stage validates the candidates and disambiguates the retained answer strings onto entities in the underlying KG (e.g., mapping “Yves Desrosiers” to Yves Desrosiers (guitarist)).

## Usage
### Data Preparation
We developed a new dataset with an emphasis on the long-tail challenge, called [MALT](https://zenodo.org/record/8092562) (for “Multi-token, Ambiguous, Long-Tailed facts”).
After downloading, put the MALT file in the root path.
There are files in the MALT dataset:
* `malt_eval.txt` contains entity IDs for evaluation
* `malt_hold_out.txt` contains entity IDs for adjusting the hyper-parameters
* `gold_wikidata.json` contains the gold facts
* `mal_wiki.json` contains the corresponding Wikipedia pages
### Run Example
Given the input document: 
> Lhasa de Sela said that the song was about inner happiness and
"feeling my feet in the earth, having a place in the world, of things
taking care of themselves.“ In May 2009, her collaboration
with Patrick Watson was released.

We'd like to extract the collaborators for the singer "Lhasa de Sela".
```python
python two_stage_pipeline.py -run_example True
```
After, the output is shown below:
> ( Lhasa de Sela, collaborator, Patrick Watson, 0.4763992584808626 ) <br />
> ( Lhasa de Sela, collaborator, Patrick Watson (musician), 0.3224404241174992 ) <br />
> ( Lhasa de Sela, collaborator, Patrick Watson (producer), 0.2401321410226018 ) <br />
