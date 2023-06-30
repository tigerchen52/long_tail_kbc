import json
import numpy as np


mapping = {'residence': ['P551'], 'educated at': ['P69'], 'employer': ['P108'], 'place of birth': ['P19'],
           'place of death': ['P20'], 'founded by': ['P112'],
           'performer': ['P175'], 'composer': ['P86']}

reverse_mapping = {v[0]:k for k,v in mapping.items()}


def cal_f1(gold_facts, pred_facts, name=True, in_rid=None):
    r_cnts, p_cnts, tp_cnts = dict(), dict(), dict()
    visited = set()
    for rid, gold_fact in gold_facts.items():
        # filter
        if rid == 'P17': continue
        if in_rid and rid != in_rid: continue

        for qid in gold_fact:

            gold_names = [str.lower(v) for v in sum(gold_facts[rid][qid]['tail_name'], [])]
            gold_ids = gold_facts[rid][qid]['tail_qid']

            # skip empty facts
            if len(gold_ids) == 0: continue

            preds = []
            if qid in pred_facts and rid in pred_facts[qid]:
                preds = pred_facts[qid][rid]
                visited.add(qid)

            r_cnt = len(gold_ids)
            p_cnt = len(preds)
            if name:
                tp_cnt = len(set(preds) & set(gold_names))
            else:
                tp_cnt = len(set(preds) & set(gold_ids))

            if rid not in r_cnts:
                r_cnts[rid] = 0
            r_cnts[rid] += r_cnt
            if rid not in p_cnts:
                p_cnts[rid] = 0
            p_cnts[rid] += p_cnt
            if rid not in tp_cnts:
                tp_cnts[rid] = 0
            tp_cnts[rid] += tp_cnt

    precisions, recalls, f1s = list(), list(), list()
    for qid in tp_cnts:
        p, r, tp = p_cnts[qid], r_cnts[qid], tp_cnts[qid]
        precision = tp * 1.0 / p if p != 0 else 0
        recall = tp * 1.0 / r if r != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
        print(tp, p, r)
        print('relation = {d}, precision = {a}, recall = {b}, f1 = {c}'.format(a=precision, b=recall, c=f1,
                                                                               d=reverse_mapping[qid]))
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    un_visited = set(pred_facts.keys()) - visited
    print(len(un_visited), list(un_visited)[:10])

    macro_precision, marco_recall, marco_f1 = sum(precisions) / len(precisions), sum(recalls) / len(recalls), sum(
        f1s) / len(f1s)
    return macro_precision, marco_recall, marco_f1


def load_fact(qids, file_name=None):
    facts_by_rtype = dict()
    for line in open(file_name, encoding='utf8'):
        obj = json.loads(line)
        qid = obj['qid']
        if qid not in qids:continue
        fact = obj['fact']
        for rid, values in fact.items():
            if rid in ['P2341','P205']:continue
            if rid not in facts_by_rtype:
                facts_by_rtype[rid] = dict()

            facts_by_rtype[rid][qid] = values

    stat = dict()
    for rid, fact in facts_by_rtype.items():
        for key, values in fact.items():
            if rid not in stat:
                stat[rid] = 0
            stat[rid] += len(values['tail_qid'])
    print('load pred facts :')
    print(stat)
    print('--------------------------------------')
    return facts_by_rtype


def load_qid_type(file_name):
    qid_type, qid_name, name_qid = dict(), dict(), dict()
    for line in open(file_name, encoding='utf8'):
        row = line.strip().split('\t')
        name, e_type, qid = row[0], row[1], row[2]
        qid_type[qid] = e_type
        qid_name[qid] = name
        name_qid[name] = qid
    return qid_type, qid_name, name_qid


def load_framework_result(path, alpha=0.0, type_map=mapping):
    qid_type, qid_name, name_qid = load_qid_type(file_name='MALT/entity_name_qid.txt')
    pred_facts = dict()
    fact_num = 0
    for line in open(path, encoding='utf8'):
        row = line.strip().split('\t')
        if len(row)<4:continue
        name, pred_name, e_type, score = row[0], str.lower(row[1]), row[2], float(row[3])
        if name not in name_qid:
            print(name)
            continue
        if score < alpha:continue

        fact_num += 1
        qid = name_qid[name]
        if e_type not in type_map:continue
        r_qid = type_map[e_type][0]
        if qid not in pred_facts:
            pred_facts[qid] = dict()
        if r_qid not in pred_facts[qid]:
            pred_facts[qid][r_qid] = set()
        pred_facts[qid][r_qid].add(pred_name)
    print('loaded!, path = {a}, entity number = {b}, fact number = {c}'.format(a=path, b=len(pred_facts), c=fact_num))
    return pred_facts


def eval(fact_path, gold_facts, alpha, rid):
    pred_facts = dict()
    for file in fact_path:
        pred_facts.update(load_framework_result(path=file, alpha=alpha, type_map=mapping))

    result = cal_f1(gold_facts, pred_facts, in_rid=rid)
    print(result)
    return result


def get_percentile(file, rid=None):
    scores = list()
    for line in open(file, encoding='utf8'):
        row = line.strip().split('\t')
        if len(row)<4:continue
        if row[2] not in mapping:continue
        relation, score = mapping[row[2]][0], float(row[3])
        if rid and rid!=relation:continue
        scores.append(score)
    scores.sort()
    ps = list()
    if len(scores) < 1:
        for i in range(0, 100, 2):
            ps.append(0)
            return ps
    ps = list()
    for i in range(0, 100, 2):
        v = np.percentile(scores, i)
        ps.append(v)
    return ps


def load_qids(path):
    qids = set([line.strip().split('\t')[1] for line in open(path, encoding='utf8')])
    return qids


def search_threshold(fact_path, malt_path, hold_out_path, gold_path, rid=None):
    train_qids = load_qids(path=malt_path)
    test_qids = load_qids(
        path=hold_out_path)

    train_facts = load_fact(
        train_qids,
        file_name=gold_path)

    test_facts = load_fact(
        test_qids,
        file_name=gold_path)

    percentiles = get_percentile(fact_path[0], rid)
    recalls, precisions, f1s = list(), list(), list()
    max_f1, max_result, best_alpha = 0.0, (0.0, 0.0, 0.0), 0.0
    for alpha in percentiles:
        (precision, recall, f1) = eval(fact_path=fact_path, gold_facts=train_facts, alpha=alpha, rid=rid)
        print('alpha = {d}, precision = {a}, recall = {b}, f1 = {c}'.format(a=precision, b=recall, c=f1, d=alpha))
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        if f1 > max_f1:
            max_f1 = f1
            best_alpha = alpha
            max_result = (precision, recall, f1)

    test_result = eval(fact_path=fact_path, gold_facts=test_facts, alpha=best_alpha, rid=rid)
    print('best alpha = {a}, train result = {b}'.format(a=best_alpha, b=max_result))
    print('test alpha = {a}, test result = {b}'.format(a=best_alpha, b=test_result))
    obj = {
        'precision':precisions,
        'recall':recalls,
        'f1':f1s,
        'best': {'p':max_result[0], 'r':max_result[1], 'f1':max_result[2]}
    }
    return obj


def run_eval(predicted_path, result_path, malt_path, hold_out_path, gold_path):
    result = dict()
    score = search_threshold(predicted_path, malt_path, hold_out_path, gold_path)
    result['macro'] = score

    with open(result_path, 'w', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    run_eval(['fact.txt'], 'result.txt', malt_path='MALT/train_wikidata.txt', hold_out_path='MALT/test_wikidata.txt', gold_path='MALT/filter_gold_wikidata.json')