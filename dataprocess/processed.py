# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/8/27 10:48 上午
==================================="""
import os
import glob
import json

import itertools

def extract_relation():
    """
    处理Duconv 数据中的关系信息
    :return:
    """
    data_path = '../data/Duconv'
    files_path = glob.glob(data_path+'/*.txt')

    set_relation = set()
    for data_p in files_path:
        with open(data_p, 'r', encoding='utf8') as f:
            for line in f:
                line_data = json.loads(line)
                goal = line_data['goal']
                knowledge = line_data['knowledge']
                for goal_ in goal[1:]:
                    set_relation.add(goal_[1])
                for knowledge_ in knowledge:
                    set_relation.add(knowledge_[1])

    with open(data_path+f'/relations{len(set_relation)}.txt', 'w') as f:
        result = {}
        for relation in set_relation:
            result[relation] = ''.join(relation.split(' '))
        f.writelines(str(result))

def process_duconv_data():
    data_path = '../data/Duconv'
    files_path = glob.glob(data_path+'/*.txt')

    relation_dict = {'妻子': '{0}的妻子是{1}', '毕业 院校': '{0}的毕业院校是{1}', '星座': '{0}的星座是{1}', '代表作': '{0}的代表作有{1}', '出生 日期': '{0}的出生日期是{1}', '身高': '{0}的身高是{1}', '体重': '{0}的体重是{1}', '好友': '{0}的好友是{1}', '主要 成就': '{0}的主要成就是{1}',  '国家': '{0}所属国家是{1}', '学历': '{0}的学历是{1}', '丈夫': '{0}的丈夫是{1}',  '导演': '{0}的导演是{1}',  '祖籍': '{0}的祖籍是{1}', '血型': '{0}的血型是{1}', '主要成就': '{0}的主要成就是{1}',  '民族': '{0}的民族是{1}', '性别': '{0}的性别是{1}', '家人': '{0}的家人是{1}',  '职业': '{0}的职业是{1}',  '生肖': '{0}的生肖是{1}'}

    for data_p in files_path:
        with open(data_p, 'r', encoding='utf8') as f:
            print(f"loading data from {data_p}")
            if 'test' in data_p:
                continue
            txt_file = data_p.split('/')[-1].split('.')[0] + 'knowledge.txt'
            with open(data_path+os.sep+txt_file, 'w', encoding='utf-8') as w_f:
                for line in f:
                    line_data = json.loads(line)
                    goal = line_data['goal']
                    knowledge = line_data['knowledge']
                    line_dict = {'knowledge':[], 'conversation': []}
                    for k in knowledge:
                        rel = k[1]
                        if rel in relation_dict:
                            entity_1 = ''.join(k[0].split(' '))
                            entity_2 = ''.join(k[2].split(' '))

                            line_dict['knowledge'].append(relation_dict[rel].format(entity_1, entity_2))
                    line_dict['conversation'] = [''.join(t.split(' ')) for t in line_data['conversation']]
                    w_f.write(json.dumps(line_dict, ensure_ascii=False)+'\n')



def process_emotion_c3kg_data():
    data_path = '../data/C3KG/AllEmotionLabel.json'
    save_path = '../data/processed/c3kg.jsonl'
    emotion_dict = {'惊讶': '惊讶', '悲伤': '忧伤', '开心': '开心', 'others': '平静', '生气': '生气'}
    template = "以下是两个人物的对话：\n\n{0}"
    with open(data_path, 'r', encoding='utf8') as f, open(save_path, 'w', encoding='utf8') as w_f:
        datas = json.load(f)
        for da in datas:
            p1_utterance = []
            p2_utterance = []
            for i, line in enumerate(da):
                label = line['label']

                if i % 2 == 0:
                    sentence = 'P1（{0}）："{1}"'.format(emotion_dict[label], line['sens']) + '\n'
                    p1_utterance.append(sentence)
                else:
                    sentence = 'P2（{0}）："{1}"'.format(emotion_dict[label], line['sens']) + '\n'
                    p2_utterance.append(sentence)
            # 交叉合并两个列表
            sentences = list(itertools.chain(*zip(p1_utterance, p2_utterance)))
            sentences_str = ''.join(sentences)
            sentences_str = sentences_str.strip()

            template_ = template.format(sentences_str)
            texts = {'text': template_}
            w_f.write(json.dumps(texts, ensure_ascii=False)+'\n')


def process_dialog_zh():
    data_path = '../data/multi_turn/dialog_zh.json'
    save_path = '../data/processed/dialog_zh.jsonl'

    template = "以下是两个人物的对话：\n\n{0}"
    with open(data_path, 'r', encoding='utf8') as f, open(save_path, 'w', encoding='utf8') as w_f:
        datas = json.load(f)
        for da in datas:
            p1_utterance = []
            p2_utterance = []
            for i, line in enumerate(da['content']):

                if i % 2 == 0:
                    sentence = 'P1：{0}'.format(line) + '\n'
                    p1_utterance.append(sentence)
                else:
                    sentence = 'P2：{0}'.format(line) + '\n'
                    p2_utterance.append(sentence)
            # 交叉合并两个列表
            sentences = list(itertools.chain(*zip(p1_utterance, p2_utterance)))
            sentences_str = ''.join(sentences)
            sentences_str = sentences_str.strip()

            template_ = template.format(sentences_str)
            texts = {'text': template_}
            w_f.write(json.dumps(texts, ensure_ascii=False)+'\n')


def process_theirs_data():
    data_path = '../data/卡夫卡/0527/processed.txt'
    save_path = '../data/processed/theirs.jsonl'
    template = "以下是两个人物的对话：\n\n{0}"
    with open(data_path, 'r', encoding='utf8') as f, open(save_path, 'w', encoding='utf8') as w_f:
        for line in f:
            temp_template = 'P1："{0}"\nP2："{1}"'
            sentences = line.split(' 回复:')
            try:
                sen = temp_template.format(sentences[0].replace('对话上文:', ''), sentences[1].strip())
                template_ = template.format(sen)
                texts = {'text': template_}
                w_f.write(json.dumps(texts, ensure_ascii=False)+'\n')
            except IndexError as e:
                print(line)
                continue
            template_ = template.format(sen)
            texts = {'text': template_}
            w_f.write(json.dumps(texts, ensure_ascii=False)+'\n')


def process_duconv_data():
    data_path = '../data/Duconv/all.txt'
    save_path = '../data/processed/duconv.jsonl'

    template = '使用以下人物信息进行对话：\n\n{0}\n\n以下是两个人物的对话：\n\n{1}'

    with open(data_path, 'r', encoding='utf8') as f, open(save_path, 'w', encoding='utf8') as w_f:
        for line in f:
            datas = json.loads(line)
            knowledge = datas['knowledge']
            knowledge_ = '；'.join(knowledge)
            p1_utterance = []
            p2_utterance = []
            for i, sen in enumerate(datas['conversation']):

                if i % 2 == 0:
                    sentence = 'P1："{0}"'.format(sen) + '\n'
                    p1_utterance.append(sentence)
                else:
                    sentence = 'P2："{0}"'.format(sen) + '\n'
                    p2_utterance.append(sentence)
            # 交叉合并两个列表
            sentences = list(itertools.chain(*zip(p1_utterance, p2_utterance)))
            sentences_str = ''.join(sentences)
            sentences_str = sentences_str.strip()

            template_ = template.format(knowledge_, sentences_str)
            texts = {'text': template_}
            w_f.write(json.dumps(texts, ensure_ascii=False)+'\n')


def process_dulemon_self_data():
    data_path = '../data/DuLeMon/self'
    files_path = glob.glob(data_path+'/*.json')
    save_path = '../data/processed/dulemon_self.jsonl'

    template = '使用以下人物信息进行对话：\n\n{0}\n{1}\n\n以下是两个人物的对话：\n\n{2}'

    with open(save_path, 'w', encoding='utf8') as w_f:
        for file_path in files_path:
            with open(file_path, 'r', encoding='utf8') as f:
                for line in f:
                    datas = json.loads(line)
                    p1_persona = datas['p1_persona']
                    p2_persona = datas['p2_persona']
                    conversation = datas['conversation']

                    if 'train' in str(files_path):
                        p1_persona_str = _util_for_dulemon_self(p1_persona, True)
                        p2_persona_str = _util_for_dulemon_self(p2_persona, True)
                    else:
                        p1_persona_str = _util_for_dulemon_self(p1_persona)
                        p2_persona_str = _util_for_dulemon_self(p2_persona)

                    p1_utterance = []
                    p2_utterance = []
                    for i, sen in enumerate(conversation):
                        # 处理：我 觉得 很好 喝 ， 就 回家 买 了 一个 咖啡机 自己 做 。\tU5
                        if '\t' in sen:
                            sen = sen.split('\t')[0]
                        sen_str = ''.join(sen.split(' ')[1:])
                        if i % 2 == 0:
                            sentence = 'P1："{0}"'.format(sen_str) + '\n'
                            p1_utterance.append(sentence)
                        else:
                            sentence = 'P2："{0}"'.format(sen_str) + '\n'
                            p2_utterance.append(sentence)
                    # 交叉合并两个列表
                    sentences = list(itertools.chain(*zip(p1_utterance, p2_utterance)))
                    sentences_str = ''.join(sentences)
                    sentences_str = sentences_str.strip()

                    template_ = template.format(p1_persona_str, p2_persona_str, sentences_str)
                    texts = {'text': template_}
                    w_f.write(json.dumps(texts, ensure_ascii=False)+'\n')

def process_lccd_data():
    data_path = '../data/LCCD.json'
    save_path = '../data/processed/LCCD.jsonl'
    all_data = []
    template = "以下是两个人物的对话：\n\n{0}"
    with open(data_path, 'r', encoding='utf8') as f, open(save_path, 'w', encoding='utf8') as w_f:
        data_train = json.load(f)
        for datas in data_train:
            p1_utterance = []
            p2_utterance = []
            for i, sen in enumerate(datas):
                if i % 2 == 0:
                    sentence = 'P1："{0}"'.format(''.join(sen.split(' '))) + '\n'
                    p1_utterance.append(sentence)
                else:
                    sentence = 'P2："{0}"'.format(''.join(sen.split(' '))) + '\n'
                    p2_utterance.append(sentence)
            # 交叉合并两个列表
            sentences = list(itertools.chain(*zip(p1_utterance, p2_utterance)))
            sentences_str = ''.join(sentences)
            sentences_str = sentences_str.strip()

            template_ = template.format(sentences_str)
            texts = {'text': template_}
            w_f.write(json.dumps(texts, ensure_ascii=False) + '\n')

def _util_for_dulemon_self(persona_list, is_train=False):
    """
    将list转换为 str
    :return:
    """
    p1_persona_list = []
    for text in persona_list:
        if is_train:
            p1_persona_list.append(''.join(text.split(' ')[1:]))
        else:
            p1_persona_list.append(''.join(text.split(' ')))

    return '；'.join(p1_persona_list)


if __name__ == '__main__':
    # process_emotion_c3kg_data()
    # process_theirs_data()
    # process_duconv_data()
    # process_dulemon_self_data()
    process_lccd_data()