from transformers import pipeline


def generate(QA_MODEL, name, proof_sentence, topk=10, tp_number=1, templates=None):
    context = proof_sentence
    pre_results = dict()
    for template in templates[:tp_number]:
        question = template.format(a=name)
        results = QA_MODEL(question=question, context=context, topk=topk)
        if len(results) < 1:continue
        for result in results:
            confidence, pred_name, start, end = result['score'], result['answer'], result['start'], result['end']
            scores = [float(r['score']) for r in results if pred_name in r['answer'] or r['answer'] in pred_name]
            score = sum(scores) / len(scores)
            confidence = score
            if pred_name in pre_results:continue
            pre_results[pred_name] = (confidence, start, end)
    return pre_results


if __name__ == '__main__':

    model_name = 'mrm8488/spanbert-finetuned-squadv2'
    QA_MODEL = pipeline(task='question-answering', model=model_name, tokenizer=model_name)
    # wiki_page = utils.load_wiki_page(path='../pipeline/data/latest_wiki.json')
    # page = wiki_page[0]
    name = 'Tavolevo River'
    pred_name = 'Chile'
    content = 'in Chile.'
    my_template = {'PersonPlaceOfDeath': ['which country does {a} flow through?']}
    #proof = utils.collect_proof_sentences(content, pred_name)
    score = generate(QA_MODEL, name=name, proof_sentence=content, pred=pred_name, my_templates=my_template)
    print(score)
