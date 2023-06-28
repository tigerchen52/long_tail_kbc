import candidate_generation
import corroboration
import utils
import math
from transformers import pipeline
import nltk
import template
import argparse
import sys
import torch

#hyper-parameters
parser = argparse.ArgumentParser(description='contrastive learning framework for word vector')
parser.add_argument('-malt_dataset', help='the file path of the MALT evaluation dataset', type=str, default='MALT/malt_eval.txt')
parser.add_argument('-hold_out_dataset', help='the hold-out dataset', type=str, default='MALT/malt_hold_out.txt')
parser.add_argument('-qa_model', help='the name of the qa model for candidate generation ', type=str, default='mrm8488/spanbert-finetuned-squadv2')
parser.add_argument('-output_path', help='the file of extracted facts', type=str, default='extracted_facts.txt')
parser.add_argument('-wikipedia_dataset', help='Wikipedia pages ', type=str, default='MALT/mal_wiki.json')
parser.add_argument('-topk', help='topk', type=int, default=20)
parser.add_argument('-max_len', help='the maximum length of an input context sentence', type=int, default=1024)
parser.add_argument('-min_can_name_len', help='the minimum length of a candidate name', type=int, default=3)
parser.add_argument('-min_sen_len', help='the minimum length of a sentence', type=int, default=30)
parser.add_argument('-run_example', help='if run the example', type=bool, default=False)


try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)


def load_name():
    train_names = [line.strip().split('\t')[0] for line in
                   open(args.malt_dataset, encoding='utf8')]
    test_names = [line.strip().split('\t')[0] for line in
                  open(args.hold_out_dataset, encoding='utf8')]
    return set(train_names + test_names)


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
corroboration.model.to(device)

tokenizer = args.qa_model
model_name = args.qa_model


model_for_candidate = pipeline(task='question-answering', model=model_name, tokenizer=tokenizer,
                    device=int(device.split(':')[-1]))


def run_on_malt():
    f = open(args.output_path, 'w', encoding='utf8')

    wiki_pages = utils.load_wiki_page(path=args.wikipedia_dataset)

    for relation_type in template.candidate_templates.keys():
        file_type = template.file_names[relation_type]

        cnt = 0
        for index, name in enumerate(wiki_pages):

            wp_page = wiki_pages[name]
            e_type = wp_page['type']
            if e_type != file_type: continue
            cnt += 1
            page_content = wp_page['wikipage']
            print('processing {a} lines'.format(a=cnt))
            print('process name = {a} ......'.format(a=name))

            predict_result = set()
            page_contents = nltk.sent_tokenize(page_content)
            for page_content in page_contents:
                if len(page_content) < args.min_sen_len: continue
                candidates = candidate_generation.generate(model_for_candidate, name, page_content, topk=args.topk,
                                                  templates=template.candidate_templates[relation_type])

                for can_name in candidates:
                    if len(can_name) < args.min_can_name_len: continue
                    if utils.filter(can_name): continue
                    qa_score, start, end = candidates[can_name]
                    sentence = page_content
                    corroborated_results = corroboration.genre_predict((name, can_name), sentence[:args.max_len], topk=args.topk, num_beams=args.topk,
                                                             templates=template.corroboration_templates[relation_type])
                    corroborated_results = dict([(n, math.exp(v)) for n, v in corroborated_results])

                    for sr in corroborated_results:
                        clean_sr = utils.clean_genre(sr)
                        if can_name != clean_sr: continue

                        if clean_sr in predict_result:continue
                        predict_result.add(clean_sr)


                        ed_score = corroborated_results[sr]
                        avg_score = 0.5 * (qa_score + ed_score)
                        w_l = name + '\t' + clean_sr + '\t' + relation_type + '\t' + str(
                            round(avg_score, 8)) + '\t' + str(
                            round(qa_score, 8)) + '\t' + str(
                            round(ed_score, 8)) + '\t' + sentence + '\n'
                        f.write(w_l)
                        f.flush()


def run_example():
    doc = """
        Lhasa de Sela said that the song was about inner happiness and
        "feeling my feet in the earth, having a place in the world, of things
        taking care of themselves.“ In May 2009, her collaboration
        with Patrick Watson was released.
    """
    name = 'Lhasa de Sela'
    candidates = candidate_generation.generate(model_for_candidate, name, doc, topk=args.topk,
                                               my_templates=['the person collaborated with which person?'])

    for can_name in candidates:
        if len(can_name) < args.min_can_name_len: continue
        if utils.filter(can_name): continue
        qa_score, start, end = candidates[can_name]
        sentence = doc
        corroborated_results = corroboration.genre_predict((name, can_name), sentence[:args.max_len], topk=args.topk,
                                                           num_beams=args.topk,
                                                           templates=['the person {a} collaborated with [START_ENT] this person [END_ENT].'])
        corroborated_results = dict([(n, math.exp(v)) for n, v in corroborated_results])

        for sr in corroborated_results:
            clean_sr = utils.clean_genre(sr)
            if can_name != clean_sr: continue

            ed_score = corroborated_results[sr]
            avg_score = 0.5 * (qa_score + ed_score)
            print('( Lhasa de Sela, collaborator, ' + sr + ', ' + str(avg_score) +' )')


if __name__ == '__main__':
    if args.run_example:
        run_example()
    else:
        run_on_malt()