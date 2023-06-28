import pickle
from GENRE.genre.trie import Trie
from GENRE.genre.hf_model import GENRE

dict_path = "GENRE/data/genre/kilt_titles_trie_dict.pkl"
model_path = "GENRE/data/genre/hf_entity_disambiguation_aidayago"

with open(dict_path, "rb") as f:
    trie = Trie.load_from_dict(pickle.load(f))

model = GENRE.from_pretrained(model_path).eval()


def genre_predict(name, abstract, num_beams=30, topk=10, templates=None):
    results = dict()
    for template in templates:
        sentences = [
            name[1] + ' ' + abstract + ' ' +
            template.format(a=name[0])
        ]
        predicted = model.sample(
            sentences,
            prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
            num_beams=num_beams,
            num_return_sequences=topk
        )

        predicted = [(pred[0]['text'], round(pred[0]['logprob'].item(), 5)) for pred in predicted]
        for name, score in predicted:
            if name not in results:
                results[name] = 0
            results[name] += score
    results = sorted(results.items(), key=lambda e:e[1], reverse=True)[:topk]

    return results


if __name__ == '__main__':
        sentences = """In May 2009, the collaboration of Lhasa de Sela and Patrick Watson was released: the song "Wooden Arms" on his album Wooden Arms. Lhasa de Sela collaborated with [START_ENT] this person [END_ENT]"""
        predicted = model.sample(
            [sentences],
            prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
            num_beams=20,
            num_return_sequences=20
        )
        predicted = [(pred[0]['text'], round(pred[0]['logprob'].item(), 5)) for pred in predicted]
        print(predicted)