'''
This is the 1st version of reformat_data.py, which need further reformation and modification
'''
import xlrd
import datetime
import json
from v1_code import utils
import numpy as np
N_CLUSTER = 4


def reformat_source_data():
    read_project = ['angular', 'Appium',
                    'docker', 'dl4j',
                    'ethereum', 'Gitter',
                    'Typescript', 'nodejs']
    workbook = xlrd.open_workbook('excel/source_data.xlsx')
    final_data_dict = dict()
    modify_data_dict = dict()
    for project in read_project:
        worksheet = workbook.sheet_by_name(project)
        data_origin = worksheet.col_values(1, 1, worksheet.nrows)
        data_modify = worksheet.col_values(2, 1, worksheet.nrows)
        disentangled_data_project = list()
        for i in range(len(data_modify)):
            if data_modify[i] == 'delete':
                continue
            elif data_modify[i] == 'Y' or data_modify[i] == 'Y?':
                former_stored_dialogs = get_reconstruct_list(data_origin[i])
            else:
                former_stored_dialogs = get_reconstruct_list(data_modify[i])
            for dialog_stored in former_stored_dialogs:
                disentangled_data_project.append(dialog_stored)
        final_data_dict[project] = disentangled_data_project
        modify_data_dict[project] = data_modify
    return final_data_dict, modify_data_dict


def get_reconstruct_list(data_string):
    list_data_result = data_string.split('\n')
    dialog_sequence_list = list()
    string_data, date_string, text_string = '', '', ''
    mark_new_sentence = False
    former_stored_dialogs = list()
    for line_data in list_data_result:
        if len(line_data) == 0:
            continue
        if line_data.startswith('//'):
            if len(string_data) > 0:
                dialog_sequence_list.append(string_data)
                former_stored_dialogs.append(dialog_sequence_list)
            dialog_sequence_list = []
            string_data = ''
            continue
        if (line_data[0] == '[' and line_data[1] == '2' and line_data[2] == '0'):
            date_string = line_data[0: 21]
            text_string = line_data[21:]
            mark_new_sentence = True
        elif (line_data[0].isdigit() and line_data[1] == ' ' and line_data[2] == '[' and line_data[3] == '2' and
                 line_data[4] == '0'):
            date_string = line_data[2: 23]
            text_string = line_data[23:]
            mark_new_sentence = True
        elif (line_data[0].isdigit() and line_data[1].isdigit() and line_data[2] == ' ' and line_data[3] == '[' and
                 line_data[4] == '2' and line_data[5] == '0'):
            date_string = line_data[3: 24]
            text_string = line_data[24:]
            mark_new_sentence = True
        if mark_new_sentence:
            date_string = date_string.replace('T', ' ')
            # print(date_string)
            if string_data == '':
                string_data = date_string + text_string
            else:
                dialog_sequence_list.append(string_data)
                string_data = date_string + text_string
        else:
            string_data += line_data
    dialog_sequence_list.append(string_data)
    former_stored_dialogs.append(dialog_sequence_list)
    return former_stored_dialogs


def reformat_e2d_data():
    dialog_dict, _ = reformat_source_data()
    e2d_cluster_sentences = list()
    project_e2d_cluster, dic_e2d_cluster = list(), dict()
    for project in dialog_dict.keys():
        dialog_project_list = dialog_dict[project]
        sentences_cluster = list()
        for i in range(len(dialog_project_list)):
            if i > 0 and i % N_CLUSTER == 0:
                sentences_cluster = sorted(sentences_cluster,
                                           key=lambda x: datetime.datetime.strptime(x[1: 20], "%Y-%m-%d %H:%M:%S"))
                json_list_cluster = list()
                for sentence in sentences_cluster:
                    # print(sentence.index('<'))
                    time_name = sentence[sentence.index('[') + 1: sentence.index(']')]
                    user_name = sentence[sentence.index('<') + 1: sentence.index('>')]
                    text_sentence = sentence[sentence.index('>') + 2:]
                    sentence_dict = dict()
                    for k_index in range(len(dialog_project_list)):
                        if sentence in dialog_project_list[k_index]:
                            label_sentence = k_index % 4
                            break
                    print(user_name, text_sentence, label_sentence)
                    sentence_dict['time'] = time_name
                    sentence_dict['speaker'] = user_name
                    sentence_dict['utterance'] = text_sentence
                    sentence_dict['label'] = label_sentence
                    json_list_cluster.append(sentence_dict)
                e2d_cluster_sentences.append(json_list_cluster)
                project_e2d_cluster.append(json_list_cluster)
                sentences_cluster = list()
            sentences_cluster += dialog_project_list[i]
        sentences_cluster = sorted(sentences_cluster,
                                   key=lambda x: datetime.datetime.strptime(x[1: 20], "%Y-%m-%d %H:%M:%S"))
        json_list_cluster = list()
        for sentence in sentences_cluster:
            # print(sentence.index('<'))
            time_name = sentence[sentence.index('[') + 1: sentence.index(']')]
            user_name = sentence[sentence.index('<') + 1: sentence.index('>')]
            text_sentence = sentence[sentence.index('>') + 2:]
            sentence_dict = dict()
            for k_index in range(len(dialog_project_list)):
                if sentence in dialog_project_list[k_index]:
                    label_sentence = k_index % 4
                    break
            print(user_name, text_sentence, label_sentence)
            sentence_dict['time'] = time_name
            sentence_dict['speaker'] = user_name
            sentence_dict['utterance'] = text_sentence
            sentence_dict['label'] = label_sentence
            json_list_cluster.append(sentence_dict)
        e2d_cluster_sentences.append(json_list_cluster)
        project_e2d_cluster.append(json_list_cluster)
        dic_e2d_cluster[project] = project_e2d_cluster
        project_e2d_cluster = []

    # json.dump(e2d_cluster_sentences, open('data/retrain_sample_data_2.json', 'w'), indent=4)
    project_name_list = ['angular', 'Appium', 'docker', 'dl4j', 'ethereum', 'Gitter', 'Typescript', 'nodejs']
    for project_name in project_name_list:
        index_sentence = 0
        ff_result = []
        data_project = dic_e2d_cluster[project_name]
        for cluster_data in data_project:
            # print(cluster_data)
            for i in range(len(cluster_data)):
                sentence = cluster_data[i]
                if i > 0:
                    j_mark = False
                    for j in range(i):
                        if cluster_data[j]['label'] == sentence['label']:
                            j_mark = True
                            sentence_compare = cluster_data[j]
                            sentence_compare_text = '[' + sentence_compare['time'] + '] ' + \
                                        '<' + sentence_compare['speaker'] + '> ' + \
                                        sentence_compare['utterance']
                            sentence_text = '[' + sentence['time'] + '] ' + \
                                            '<' + sentence['speaker'] + '> ' + \
                                            sentence['utterance']
                            index_compare = 0
                            for base_sentence, base_start, base_end in ff_result:
                                if base_sentence == sentence_compare_text:
                                    index_compare = base_start
                                    break
                            ff_result.append((sentence_text, index_sentence, index_compare))
                            break
                    if not j_mark:
                        sentence_text = '[' + sentence['time'] + '] ' + \
                                        '<' + sentence['speaker'] + '> ' + \
                                        sentence['utterance']
                        start_pos = index_sentence
                        end_pos = index_sentence
                        ff_result.append((sentence_text, start_pos, end_pos))
                elif i == 0:
                    sentence_text = '[' + sentence['time'] + '] ' + \
                                    '<' + sentence['speaker'] + '> ' + \
                                    sentence['utterance']
                    start_pos = index_sentence
                    end_pos = index_sentence
                    ff_result.append((sentence_text, start_pos, end_pos))
                index_sentence += 1
        with open(f'data/retrain/{project_name}.ascii.txt', mode='w+', encoding='utf8') as f:
            with open(f'data/retrain/{project_name}.annotation.txt', mode='w+', encoding='utf8') as f1:
                for ff_text, index_now, index_cluster in ff_result:
                    f.write(ff_text)
                    f.write('\n')
                    f1.write(str(index_now) + ' ' + str(index_cluster) + ' ' + '-')
                    f1.write('\n')
        f.close()
        f1.close()

    # print(ff_result)
    return


def reformat_ff_data():
    data_result, _ = reformat_source_data()
    dict_return_metrics = dict()
    for project in data_result.keys():
        project_sentences = list()
        for sentence in data_result[project]:
            project_sentences += sentence
        project_sentences = sorted(project_sentences,
                                   key=lambda x: datetime.datetime.strptime(x[1: 20], "%Y-%m-%d %H:%M:%S"))
        # print(project_sentences)
        # with open(f'data/test_disentangled/{project}_testfile.ascii.txt', mode='w+', encoding='utf8') as f:
        #     for sentence in project_sentences:
        #         f.write(sentence + '\n')
        # f.close()
        data_list_new = list()
        with open(f'data/retrain/message/{project}.annotation.txt', 'r', encoding='utf8') as f:
            data_list_new = f.readlines()
        f.close()
        result_predict_list = list()
        predict_list_each = list()
        for data_new in data_list_new:
            if data_new == '--------------------------------------------------------------------------------------------------\n':
                result_predict_list.append(predict_list_each)
                predict_list_each = list()
            else:
                if ']' in data_new:
                    index_start = data_new.index(']')
                    predict_list_each.append(data_new[index_start + 2:].replace('\n', '').replace(' ', ''))
                else:
                    predict_list_each.append(data_new.replace('\n', '').replace(' ', ''))
        result_original_list = data_result[project]
        predicted_labels, truth_labels = [], []
        # for predict_data in result_predict_list:
        #     predicted_labels.append([0] * len(predict_data))
        #     truth = []
        #     mark_zero_sentence, mark_next_sentence = 0, 0
        #     for j in range(len(result_original_list)):
        #         if predict_data[0] in result_original_list[j]:
        #             mark_zero_sentence = j
        #             break
        #     for i in range(len(predict_data)):
        #         if i == 0:
        #             truth.append(0)
        #         else:
        #             for j in range(len(result_original_list)):
        #                 if predict_data[i] in result_original_list[j]:
        #                     mark_next_sentence = j
        #                     break
        #             if mark_zero_sentence == mark_next_sentence:
        #                 truth.append(0)
        #             else:
        #                 truth.append(2)
        #     truth_labels.append(truth)
        result_original_new, result_origin = [], []
        truths = []
        mark_truth = 0
        for i in range(len(result_original_list)):
            # if (mark_truth + 1) % 2 == 0:
            #     result_origin += result_original_list[i]
            #     truths += [mark_truth] * len(result_original_list[i])
            #
            # else:
            result_origin += result_original_list[i]
            truths += [mark_truth] * len(result_original_list[i])
            mark_truth += 1
            result_original_new.append(result_origin)
            truth_labels.append(truths)
            mark_truth = 0
            truths = []
            result_origin = []
        # result_original_new.append(result_origin)
        # truth_labels.append(truths)
        # mark_truth = 0
        # truths = []
        # result_origin = []
        for original_data in result_original_new:
            # truth_labels.append([0] * len(original_data))
            predicts = []
            mark_next_sentence, mark_zero_sentence = 0, 0
            mark_pos_list, mark_list = [], []
            for j in range(len(result_predict_list)):
                index_start = original_data[0].index(']')
                if original_data[0][index_start + 2:].replace(' ', '') in result_predict_list[j]:
                    mark_zero_sentence = j
                    break
            mark_pos_list.append(mark_zero_sentence)
            mark_list.append(0)
            count_cluster = 0
            for i in range(len(original_data)):
                if i == 0:
                    predicts.append(0)
                else:
                    for j in range(len(result_predict_list)):
                        index_start = original_data[i].index(']')
                        if original_data[i][index_start + 2:].replace(' ', '') in result_predict_list[j]:
                            mark_next_sentence = j
                            break
                    j_find = 0
                    for j in range(len(mark_pos_list)):
                        if mark_pos_list[j] == mark_next_sentence:
                            predicts.append(mark_list[j])
                            mark_pos_list.append(mark_next_sentence)
                            mark_list.append(mark_list[j])
                            break
                        j_find += 1
                    if j_find == len(mark_pos_list):
                        count_cluster += 1
                        mark_pos_list.append(mark_next_sentence)
                        mark_list.append(count_cluster)
                        predicts.append(count_cluster)
            predicted_labels.append(predicts)
        # print('aaa')
        # print(result_predict_list)
        read_project = {'angular': 11, 'Appium': 8,
                        'docker': 9, 'dl4j': 9,
                        'ethereum': 9, 'Gitter': 9,
                        'Typescript': 9, 'nodejs': 9}
        workbook = xlrd.open_workbook('excel/source_annotation.xlsx')
        worksheet = workbook.sheet_by_name(project)
        score_origin = worksheet.col_values(read_project[project], 1, worksheet.nrows)
        predicted_data_list, origin_data_list = [], []
        data_predicted = worksheet.col_values(1, 1, worksheet.nrows)
        data_origin = worksheet.col_values(2, 1, worksheet.nrows)
        for i in range(len(data_origin)):
            if data_origin[i] == 'Y':
                data_origin[i] = data_predicted[i]
        mark_final = []
        nmi_score = []
        ari_score = []
        shen_f_score = []
        dict_result = dict()
        mark_truth_category, score_sum = 0, []
        predicted_reformat_list, truth_reformat_list = [], []
        for origin_data, predicted_data, truth_data in zip(result_original_new, predicted_labels, truth_labels):
            symbol_sentence = origin_data[0]
            symbol_sentence_reformat = symbol_sentence[symbol_sentence.index('<'):].replace(' ', '')
            for origin_search, score_search in zip(data_origin, score_origin):
                origin_search_reformat = origin_search.replace(' ', '')
                if symbol_sentence_reformat in origin_search_reformat and score_search != '':
                    # mark_final.append((score_search - 1) * 0.25)
                    # nmi_score.append(utils.compare([predicted_data], [truth_data], 'NMI'))
                    # ari_score.append(utils.compare([predicted_data], [truth_data], 'ARI'))
                    # shen_f_score.append(utils.compare([predicted_data], [truth_data], 'shen_f'))
                    score_sum.append((score_search - 1) * 0.25)
                    break
            truth_reformat_list += [mark_truth_category] * len(truth_data)
            add_data = 0
            if len(predicted_reformat_list) > 0:
                add_data += (max(predicted_reformat_list) + 1)
            predicted_reformat_list += [predicted + add_data for predicted in predicted_data]
            mark_truth_category += 1
            if mark_truth_category == 4:
                mark_final.append(np.array(score_sum).mean())
                nmi_score.append(utils.compare([predicted_reformat_list], [truth_reformat_list], 'NMI'))
                ari_score.append(utils.compare([predicted_reformat_list], [truth_reformat_list], 'ARI'))
                shen_f_score.append(utils.compare([predicted_reformat_list], [truth_reformat_list], 'shen_f'))
                mark_truth_category, score_sum = 0, []
                predicted_reformat_list, truth_reformat_list = [], []
        dict_result['NMI'] = nmi_score
        dict_result['ARI'] = ari_score
        dict_result['Shen-F'] = shen_f_score
        dict_result['score'] = mark_final
        dict_return_metrics[project] = dict_result
        # Exchange the sequence of truth_labels and predicted_labels
        # print(utils.compare([[0, 0, 0, 1]], [[0, 0, 0, 0]], 'ARI'))
        purity_score = utils.compare(predicted_labels, truth_labels, 'purity')
        nmi_score = utils.compare([predicted_labels[5]], [truth_labels[5]], 'NMI')
        ari_score = utils.compare([predicted_labels[5]], [truth_labels[5]], 'ARI')
        shen_f_score = utils.compare(predicted_labels, truth_labels, 'shen_f')
        log_msg = "project: {}, purity_score: {}, nmi_score: {}, ari_score: {}, shen_f_score: {}".format(
            project, round(purity_score, 4), round(nmi_score, 4), round(ari_score, 4),
            round(shen_f_score, 4))
        # print(log_msg)
    read_project = {'angular': 11, 'Appium': 8,
                    'docker': 9, 'dl4j': 9,
                    'ethereum': 9, 'Gitter': 9,
                    'Typescript': 9, 'nodejs': 9}
    angular_data = dict_return_metrics['angular']
    metrics = ['NMI', 'ARI', 'Shen-F']
    projects_list = read_project.keys()

    nmi_angular = np.array(angular_data['NMI']).mean()
    for metric in metrics:
        for project in projects_list:
            project_data = dict_return_metrics[project]
            metric_project = np.array(project_data[metric]).mean()
            print("Metric: {}, Project: {}, Value: {}".format(metric, project, metric_project))
    return dict_return_metrics


def divide_retrain_data():
    dic_sample = get_json_data()
    project_name_list = ['angular', 'Appium', 'docker', 'dl4j', 'ethereum', 'Gitter', 'Typescript', 'nodejs']
    start_pos_list = [0, 27, 51, 78, 104, 129, 152, 179]
    end_pos_list = [26, 50, 77, 103, 128, 151, 178, 203]
    project_split = {'angular':'train', 'Appium':'train', 'docker':'test',
                     'dl4j':'train', 'ethereum':'train', 'Gitter':'train',
                     'Typescript':'train', 'nodejs':'dev'}
    sample_train, sample_dev, sample_test_new = [], [], []
    for project_name, start_pos, end_pos in zip(project_name_list, start_pos_list, end_pos_list):
        if project_split[project_name] == 'train':
            sample_train += dic_sample[start_pos:end_pos]
        elif project_split[project_name] == 'dev':
            sample_dev += dic_sample[start_pos:end_pos]
        elif project_split[project_name] == 'test':
            sample_test_new += dic_sample[start_pos:end_pos]
    json.dump(sample_train, open('data/retrain_sample_train.json', 'w'), indent=4)
    json.dump(sample_dev, open('data/retrain_sample_dev.json', 'w'), indent=4)
    json.dump(sample_test_new, open('data/retrain_sample_test.json', 'w'), indent=4)


def reformat_data_for_e2d_new_metrics(model_baseline):
    json_data = get_json_data()
    truth_data = dict()
    # , 'Appium', 27, 50
    project_name_list = ['angular', 'Appium', 'docker', 'dl4j', 'ethereum', 'Gitter', 'Typescript', 'nodejs']
    start_pos_list = [0, 27, 51, 78, 104, 129, 152, 179]
    end_pos_list = [26, 50, 77, 103, 128, 151, 178, 203]
    for project_name, start_pos, end_pos in zip(project_name_list, start_pos_list, end_pos_list):
        truth_list = json_data[start_pos:end_pos]
        project_truth_text = []
        for each_truth in truth_list:
            project_truth_text.append(['<' + data['speaker'] + '>' + data['utterance'].replace(' ', '')
                                       for data in each_truth])
        truth_data[project_name] = project_truth_text
    data_result, _ = reformat_source_data()
    for project in data_result.keys():
        # if project == 'Appium':
        #     continue
        result_predict_list = list()
        with open(f'data/retrain/output_retrain_cluster/{model_baseline}_{project}_predicted.txt', mode='r', encoding='utf8') as f:
            label_list = f.readlines()
        # print(label_list)
        truth_label = []
        project_truth_message = truth_data[project]
        # dialog_message_list = []
        for label_data, message_list in zip(label_list, project_truth_message):
            label_split = label_data.split(' ')[:-1]
            label_num = [int(num) for num in label_split]
            truth_label.append(label_num)
            cluster_dialog_pos_each = dict()
            for i in range(len(label_num)):
                # address_index = [x for x in range(len(truth_label)) if truth_label[x] == i]
                if label_num[i] not in cluster_dialog_pos_each.keys():
                    cluster_dialog_pos_each[label_num[i]] = [i]
                else:
                    cluster_dialog_pos_each[label_num[i]].append(i)
            # print(cluster_dialog_pos_each)
            for mark in cluster_dialog_pos_each.keys():
                sentence_list = []
                sentence_i = cluster_dialog_pos_each[mark]
                for i in sentence_i:
                    if i < len(message_list):
                        sentence_list.append(message_list[i])
                if len(sentence_list) > 0:
                    result_predict_list.append(sentence_list)
            # dict_address = dict(list_price_positoin_address)
        result_original_list = data_result[project]
        sum_leven_score = 0.0
        leven_dist_list = []
        recall_list = []
        for original_data in result_original_list:
            original_return = get_dialog_reformat(original_data)
            symbolic_sentence = original_return[0]
            correspond_predicted_data = []
            for predicted_data in result_predict_list:
                if symbolic_sentence in predicted_data:
                    correspond_predicted_data = predicted_data
                    break
            # print(correspond_predicted_data)
            leven_score = calculate_leven_score(original_return, correspond_predicted_data)
            leven_dist_list.append(calculate_leven_dist(original_return, correspond_predicted_data))
            recall_list.append(calculate_PR(original_return, correspond_predicted_data))
            sum_leven_score += leven_score
        avg_leven_score = sum_leven_score / len(result_original_list)
        recall_data = recall_list.count(1)/len(recall_list)
        leven_score_data = np.mean(leven_dist_list)

        result_original_list_reformat = []
        for original_data in result_original_list:
            original_sentences = []
            for original_sentence in original_data:
                original_sentences.append(original_sentence[original_sentence.index('<'):].replace(' ', ''))
            result_original_list_reformat.append(original_sentences)
        precision_list = []
        for predicted_data in result_predict_list:
            predicted_return = get_dialog_reformat(predicted_data)
            symbolic_sentence = predicted_return[0]
            correspond_origin_data = []
            for original_data in result_original_list_reformat:
                if symbolic_sentence in original_data:
                    correspond_origin_data = original_data
                    break
            # print(correspond_predicted_data)
            leven_score = calculate_leven_score(predicted_return, correspond_origin_data)
            leven_dist_list.append(calculate_leven_dist(predicted_return, correspond_origin_data))
            precision_list.append(calculate_PR(predicted_return, correspond_origin_data))
            sum_leven_score += leven_score
        precision_data = precision_list.count(1)/len(precision_list)

        f_data = (2 * precision_data * recall_data)/(precision_data + recall_data)
        print("Project Name:{}, P Score: {}".format(project, precision_data))
        print("Project Name:{}, R Score: {}".format(project, recall_data))
        print("Project Name:{}, F Score: {}".format(project, f_data))
        with open(f'img_data/retrain/P_R_F/{model_baseline}_{project}_PRF.txt', mode='a+', encoding='utf8') as f:
            f.write("Project Name:{}, P Score: {}".format(project, precision_data))
            f.write('\n')
            f.write("Project Name:{}, R Score: {}".format(project, recall_data))
            f.write('\n')
            f.write("Project Name:{}, F Score: {}".format(project, f_data))
            f.write('\n')
        f.close()
    return


def calculate_PR(source_data, target_data):
    if len(set(source_data) - set(target_data)) == 0:
        return 1
    else:
        return 0


def get_predicted_list():
    read_project = ['angular', 'Appium',
                    'docker', 'dl4j',
                    'ethereum', 'Gitter',
                    'Typescript', 'nodejs']
    workbook = xlrd.open_workbook('excel/source_data.xlsx')
    final_data_dict = dict()
    for project in read_project:
        worksheet = workbook.sheet_by_name(project)
        data_origin = worksheet.col_values(1, 1, worksheet.nrows)
        data_modify = worksheet.col_values(2, 1, worksheet.nrows)
        predicted_data_project = list()
        # for i in range(len(data_modify)):
        #     if data_modify[i] == 'delete':
        #         continue
        #     elif data_modify[i] == 'Y' or data_modify[i] == 'Y?':
        #         former_stored_dialogs = get_reconstruct_list(data_origin[i])
        #     else:
        #         former_stored_dialogs = get_reconstruct_list(data_modify[i])
        #     for dialog_stored in former_stored_dialogs:
        #         disentangled_data_project.append(dialog_stored)
        # final_data_dict[project] = disentangled_data_project
        # modify_data_dict[project] = data_modify
        for predicted_data, modified_data in zip(data_origin, data_modify):
            if modified_data == 'delete':
                continue
            data_split = predicted_data.split('\n')
            data_new = []
            for data_sentence in data_split:
                if '[' in data_sentence:
                    index_left_time = data_sentence.index('[')
                    if data_sentence[index_left_time + 1] == '2':
                        data_result = data_sentence[data_sentence.index('<'):].replace(' ', '')
                        data_new.append(data_result)
            if data_new != []:
                predicted_data_project.append(data_new)
        final_data_dict[project] = predicted_data_project
    return final_data_dict


def reformat_data_for_ff_new_metrics():
    data_result, modify_dict = reformat_source_data()
    zscore_data_dict = dict()
    dic_data_dld = dict()
    dic_data_dlr = dict()
    for project in data_result.keys():
        # data_list_new = list()
        # with open(f'data/retrain/message/{project}.annotation.txt', 'r', encoding='utf8') as f:
        #     data_list_new = f.readlines()
        # f.close()
        # result_predict_list = list()
        # predict_list_each = list()
        # for data_new in data_list_new:
        #     if data_new == '--------------------------------------------------------------------------------------------------\n':
        #         result_predict_list.append(predict_list_each)
        #         predict_list_each = list()
        #     else:
        #         if ']' in data_new:
        #             index_start = data_new.index(']')
        #             predict_list_each.append(data_new[index_start + 2:].replace('\n', '').replace(' ', ''))
        #         else:
        #             predict_list_each.append(data_new.replace('\n', '').replace(' ', ''))
        result_predict_list = get_predicted_list()[project]

        result_original_list = data_result[project]
        leven_dist_list = []
        leven_dist_ratio_list = []
        sum_leven_score = 0.0
        for original_data in result_original_list:
            original_return = get_dialog_reformat(original_data)
            symbolic_sentence = original_return[0]
            correspond_predicted_data = []
            for predicted_data in result_predict_list:
                if symbolic_sentence in predicted_data:
                    correspond_predicted_data = predicted_data
                    break
            # print(correspond_predicted_data)
            leven_score = calculate_leven_score(original_return, correspond_predicted_data)
            leven_dist_ratio_list.append(leven_score)
            leven_dist_list.append(calculate_leven_dist(original_return, correspond_predicted_data))
            sum_leven_score += leven_score
        avg_leven_score = sum_leven_score/len(result_original_list)
        leven_score_data = np.mean(leven_dist_list)
        # max_leven_dist = max(leven_dist_list)
        # min_leven_dist = min(leven_dist_list)
        leven_dist_list_new = [sigmod_new_data(leven_dist) for leven_dist in leven_dist_list]
        # dic_data_dld[project] = leven_dist_list
        dic_data_dld[project] = leven_dist_list_new
        dic_data_dlr[project] = leven_dist_ratio_list
        array_leven_dist = np.array(leven_dist_list)
        leven_score_zscore = Z_ScoreNormalization(array_leven_dist, array_leven_dist.mean(), array_leven_dist.std())
        list_leven_zscore = list(leven_score_zscore)
        leven_score_zero_one = [sigmod_normalization(leven_score) for leven_score in list_leven_zscore]
        zscore_data_dict[project] = leven_score_zero_one
        score_mean = np.array(leven_score_zero_one).mean()


        recall_list = []
        for original_data in result_original_list:
            original_return = get_dialog_reformat(original_data)
            symbolic_sentence = original_return[0]
            correspond_predicted_data = []
            for predicted_data in result_predict_list:
                if symbolic_sentence in predicted_data:
                    correspond_predicted_data = predicted_data
                    break
            # print(correspond_predicted_data)
            leven_score = calculate_leven_score(original_return, correspond_predicted_data)
            leven_dist_list.append(calculate_leven_dist(original_return, correspond_predicted_data))
            recall_list.append(calculate_PR(original_return, correspond_predicted_data))
            sum_leven_score += leven_score
        avg_leven_score = sum_leven_score / len(result_original_list)
        recall_data = recall_list.count(1)/len(recall_list)
        leven_score_data = np.mean(leven_dist_list)

        # precision_list = []
        # data_modify = modify_dict[project]
        # for each_modify in data_modify:
        #     if each_modify == 'Y' or each_modify == 'Y?':
        #         precision_list.append(1)
        #     elif each_modify == 'delete':
        #         continue
        #     else:
        #         precision_list.append(0)
        # precision_data = precision_list.count(1)/len(precision_list)

        result_original_list_reformat = []
        for original_data in result_original_list:
            original_sentences = []
            for original_sentence in original_data:
                original_sentences.append(original_sentence[original_sentence.index('<'):].replace(' ', ''))
            result_original_list_reformat.append(original_sentences)
        precision_list = []
        for predicted_data in result_predict_list:
            predicted_return = get_dialog_reformat(predicted_data)
            symbolic_sentence = predicted_return[0]
            correspond_origin_data = []
            for original_data in result_original_list_reformat:
                if symbolic_sentence in original_data:
                    correspond_origin_data = original_data
                    break
            # print(correspond_predicted_data)
            leven_score = calculate_leven_score(predicted_return, correspond_origin_data)
            leven_dist_list.append(calculate_leven_dist(predicted_return, correspond_origin_data))
            precision_list.append(calculate_PR(predicted_return, correspond_origin_data))
            sum_leven_score += leven_score
        precision_data = precision_list.count(1) / len(precision_list)

        # # sigmod_leven = sigmod_normalization(leven_score_array)
        # # print("Project Name:{}, Leven Score: {}".format(project, avg_leven_score))
        # print("Project Name:{}, Leven Score: {}".format(project, score_mean))
        # with open(f'img_data/P_R_F/FF_{project}_PRF.txt', mode='a+', encoding='utf8') as f:
        #     f.write("Project Name:{}, Leven Score: {}".format(project, score_mean))
        #     f.write('\n')
        # f.close()

        # f_data = (2 * precision_data * recall_data)/(precision_data + recall_data)
        # print("Project Name:{}, P Score: {}".format(project, precision_data))
        # print("Project Name:{}, R Score: {}".format(project, recall_data))
        # print("Project Name:{}, F Score: {}".format(project, f_data))
        # with open(f'img_data/P_R_F/FF_{project}_PRF.txt', mode='a+', encoding='utf8') as f:
        #     f.write("Project Name:{}, P Score: {}".format(project, precision_data))
        #     f.write('\n')
        #     f.write("Project Name:{}, R Score: {}".format(project, recall_data))
        #     f.write('\n')
        #     f.write("Project Name:{}, F Score: {}".format(project, f_data))
        #     f.write('\n')
        # f.close()
    return dic_data_dld, dic_data_dlr


def Z_ScoreNormalization(x,mu,sigma):
    x = (x - mu) / sigma
    return x

def sigmod_normalization(x):
    return 1.0 / (1 + np.exp(float(x)))

def sigmod_normalization_reverse(x):
    return 1.0 / (1 + np.exp(-float(x)))

def sigmod_new_data(x):
    if x == 0:
        return 1
    else:
        return 1.0 / (1 + np.exp(float(x - 5)))

def get_dialog_reformat(original_data):
    original_return = []
    for sentence in original_data:
        if ']' in sentence:
            index_start = sentence.index(']')
            original_return.append(sentence[index_start + 2:].replace('\n', '').replace(' ', ''))
        else:
            original_return.append(sentence.replace('\n', '').replace(' ', ''))
    return original_return


def calculate_leven_score(original_return, correspond_predicted_data):
    # leven_data = len(set(original_return) - set(correspond_predicted_data)) + \
    #              len(set(correspond_predicted_data) - set(original_return))
    # # print(leven_data)
    # return 1 - float(leven_data)/(len(set(original_return)) + len(set(correspond_predicted_data)))
    delete_size = len(set(original_return) - set(correspond_predicted_data))
    add_size = len(set(correspond_predicted_data) - set(original_return))
    division_data = 0
    # if delete_size > 0:
    #     division_data += len(set(original_return))
    # if add_size > 0:
    #     division_data += len(set(correspond_predicted_data))
    division_data = len(set(original_return)) + len(set(correspond_predicted_data))
    if division_data == 0:
        return 1.0
    elif len(set(original_return) - set(correspond_predicted_data)) + len(set(correspond_predicted_data) - set(original_return)) == 0:
        return 1.0
    else:
        # print(division_data)
        return len(set(original_return) & set(correspond_predicted_data))/division_data


def calculate_leven_dist(original_return, correspond_predicted_data):
    delete_size = len(set(original_return) - set(correspond_predicted_data))
    add_size = len(set(correspond_predicted_data) - set(original_return))
    # if delete_size + add_size == 0:
    #     return 1
    # else:
    #     return 1/(delete_size + add_size)
    return delete_size + add_size


def get_json_data():
    with open('data/retrain_sample_data_2.json', 'r') as load_f:
        load_dict = json.load(load_f)
        # print(load_dict)
    # print(len(load_dict))
    return load_dict


def get_truth_annotation():
    read_project = {'angular': 11, 'Appium': 8,
                    'docker': 9, 'dl4j': 9,
                    'ethereum': 9, 'Gitter': 9,
                    'Typescript': 9, 'nodejs': 9}
    workbook = xlrd.open_workbook('excel/source_annotation.xlsx')
    final_data_dict = dict()
    for project in read_project.keys():
        worksheet = workbook.sheet_by_name(project)
        data_origin = worksheet.col_values(read_project[project], 1, worksheet.nrows)
        data_result = [(data - 1) * 0.25 for data in data_origin if data != '']
        data_std = Z_ScoreNormalization(np.array(data_result), np.array(data_result).mean(), np.array(data_result).std())
        data_zero_one = [sigmod_normalization_reverse(x) for x in list(data_std)]
        # print(np.array(data_zero_one).mean())
        print(data_result)
        final_data_dict[project] = data_result
    return final_data_dict

def get_bad_cases():
    read_project = {'angular': 12, 'Appium': 9,
                    'docker': 10, 'dl4j': 10,
                    'ethereum': 10, 'Gitter': 10,
                    'Typescript': 10, 'nodejs': 10}
    workbook = xlrd.open_workbook('excel/source_annotation.xlsx')
    final_data_dict = dict()
    for project in read_project.keys():
        worksheet = workbook.sheet_by_name(project)
        data_origin = worksheet.col_values(read_project[project], 1, worksheet.nrows)
        final_data_dict[project] = data_origin
    redundant_list, multiple_list, context_list, rules_list = [], [], [], []
    for project in final_data_dict.keys():
        data_list = final_data_dict[project]
        count_redundant, count_multiple, count_context, count_rules = 0, 0, 0, 0
        for data in data_list:
            if 'Redundant' in data:
                count_redundant += 1
            elif 'Multiple' in data:
                count_multiple += 1
            elif 'Context-aware' in data and 'Potential' in data:
                count_context += 1
            elif 'Rules' in data or 'Follow-up' in data or 'Further' in data:
                count_rules += 1
        redundant_list.append(count_redundant)
        multiple_list.append(count_multiple)
        context_list.append(count_context)
        rules_list.append(count_rules)
    return redundant_list, multiple_list, context_list, rules_list


# def store_dld_value(dld_dic_zscore, dic_truth):
#     project_name_list = ['angular', 'Appium', 'docker', 'dl4j', 'ethereum', 'Gitter', 'Typescript', 'nodejs']
#     for project in project_name_list:
#         with open('img_data/DLD_Violin/dld.txt', mode='a+', encoding='utf8') as f:
#             f.write(dld_dic_zscore[project])
#             f.write('\n')
#         f.close()
#     return


def get_predicted_truth_pair():
    read_project = {'angular': 11, 'Appium': 8,
                    'docker': 9, 'dl4j': 9,
                    'ethereum': 9, 'Gitter': 9,
                    'Typescript': 9, 'nodejs': 9}
    workbook = xlrd.open_workbook('excel/source_annotation.xlsx')
    final_data_dict = dict()
    final_data_list = []
    dict_metrics_new = dict()
    for project in read_project.keys():
        worksheet = workbook.sheet_by_name(project)
        score_origin = worksheet.col_values(read_project[project], 1, worksheet.nrows)
        predicted_data_list, origin_data_list = [], []
        data_predicted = worksheet.col_values(1, 1, worksheet.nrows)
        data_origin = worksheet.col_values(2, 1, worksheet.nrows)
        for predicted_data, origin_data in zip(data_predicted, data_origin):
            if origin_data == 'Y' or origin_data == 'Y?':
                origin_data = predicted_data
            if origin_data == 'delete':
                origin_data = '<delete'
            predicted_temp = predicted_data.split('\n')
            origin_temp = origin_data.split('\n')
            predicted_temp = [predicted[predicted.index('<'):].replace(' ', '') for predicted in predicted_temp if '<' in predicted]
            # origin_temp_new = []
            # for origin_each,

            origin_temp = [origin[origin.index('<'):].replace(' ', '') for origin in origin_temp if '<' in origin]
            if not predicted_temp:
                predicted_data_list.append(['temp'])
            else:
                predicted_data_list.append(predicted_temp)
            if not origin_temp:
                origin_data_list.append(['temp'])
            else:
                origin_data_list.append(origin_temp)
            if len(predicted_data_list) != len(origin_data_list):
                print('test')
        data_result = [(predicted, origin, (score - 1) * 0.25) for score, predicted, origin in
                       zip(score_origin, predicted_data_list, origin_data_list) if score != '']
        final_data_dict[project] = data_result
        f1_project_list, dl_project_list, score_project_list = [], [], []
        dict_result = dict()
        p_temp_list, r_temp_list, dl_temp_list, score_temp_list = [], [], [], []
        mark_data = 0
        for predicted, origin, score in data_result:
            if len(set(predicted) - set(origin)) == 0:
                p_temp_list.append(1)
                # f1_project_list.append(1.0)
            else:
                p_temp_list.append(0)
            if len(set(origin) - set(predicted)) == 0:
                r_temp_list.append(1)
            else:
                r_temp_list.append(0)
            dl_temp_list.append(0.8 * calculate_leven_score(origin, predicted) +
                                   0.2 * sigmod_new_data(calculate_leven_dist(origin, predicted)))
            score_temp_list.append(score)
            mark_data += 1
            if mark_data == 4:
                p_data = p_temp_list.count(1)/len(p_temp_list)
                r_data = r_temp_list.count(1)/len(r_temp_list)
                if p_data + r_data == 0:
                    f1_project_list.append(0)
                else:
                    f1_project_list.append(2 * p_data * r_data/(p_data + r_data))
                dl_project_list.append(np.array(dl_temp_list).mean())
                score_project_list.append(np.array(score_temp_list).mean())
                p_temp_list, r_temp_list, dl_temp_list, score_temp_list = [], [], [], []
                mark_data = 0
        final_data_list += data_result
        dict_result['F1'] = f1_project_list
        dict_result['DLD'] = dl_project_list
        dict_result['truth'] = score_project_list
        dict_metrics_new[project] = dict_result
    f1_mean_list = []
    precision_list = [len(set(predicted) & set(origin))/len(set(predicted)) for predicted, origin, score in final_data_list]
    recall_list = [len(set(origin) & set(predicted))/len(set(origin)) for predicted, origin, score in final_data_list]
    for i in range(len(final_data_list)):
        temp_metric = precision_list[i] + recall_list[i]
        truth_metric = final_data_list[i]
        if truth_metric[2] == 1.0 and temp_metric < 1.0:
            print('test')
    f1_list = []
    for precision, recall in zip(precision_list, recall_list):
        if precision + recall == 0:
            f1_list.append(0.0)
        else:
            f1_list.append(2 * precision * recall / (precision + recall))
        # f1_list = [2 * precision * recall / (precision + recall) for precision, recall in zip(precision_list, recall_list)]

    leven_dist_ratio_list = [calculate_leven_score(origin, predicted) for predicted, origin, score in final_data_list]
    leven_dist_revise_list = [sigmod_new_data(calculate_leven_dist(origin, predicted)) for predicted, origin, score in final_data_list]
    leven_dist_list = [0.8 * leven_ratio + 0.2 * leven_revise for leven_ratio, leven_revise in zip(leven_dist_ratio_list, leven_dist_revise_list)]
    score_list = [score for predicted, origin, score in final_data_list]
    dict_leven = {0.0: [], 0.25: [], 0.5: [], 0.75: [], 1.0: []}
    dict_f1 = {0.0: [], 0.25: [], 0.5: [], 0.75: [], 1.0: []}
    for data, leven, f1 in zip(final_data_list, leven_dist_list, f1_list):
        if f1 > 0.5:
            dict_f1[data[2]].append(1)
        else:
            dict_f1[data[2]].append(0)
        if leven > 0.5:
            dict_leven[data[2]].append(1)
        else:
            dict_leven[data[2]].append(0)
        if data[2] == 1.0 and leven != 1.0:
            print(data[0])
            print(data[1])
    leven_return_list, f1_return_list = [], []
    for data in dict_leven.keys():
        leven_final_list = dict_leven[data]
        f1_final_list = dict_f1[data]
        leven_return_list.append(float(leven_final_list.count(1))/len(leven_final_list))
        f1_return_list.append(float(f1_final_list.count(1))/len(f1_final_list))
    return leven_return_list, f1_return_list, score_list, leven_dist_list, f1_list, dict_metrics_new


if __name__ == '__main__':
    # get_predicted_list()
    # reformat_data_for_ff_new_metrics()
    # reformat_data_for_e2d_new_metrics('E2E_Online_Liu')

    # sample_data_to_ff_annotation()

    # reformat_e2d_data()

    # dld_dic_zscore = reformat_data_for_ff_new_metrics()
    # dic_truth = get_truth_annotation()

    # store_dld_value(dld_dic_zscore, dic_truth)

    # reformat_e2d_data()
    # divide_retrain_data()
    # print(utils.compare([[0, 1, 2], [1, 0, 1]], [[2, 3, 1], [0, 0, 3]],'purity'))

    # get_bad_cases()
    # get_predicted_truth_pair()
    reformat_ff_data()