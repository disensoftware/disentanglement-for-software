'''
This is the 1st version of train_graph.py, which need further reformation and modification
'''
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist

import seaborn as sns

# Plot the training result
from pandas import np
import scipy
import pandas as pd
from v1_code.reformat_data import *


def draw_line_comparison():
    # data_lines, data_lines_2 = list(), list()
    with open('img_data/TS_Original.txt', mode='r') as f:
        data_lines = f.readlines()
    f.close()
    with open('img_data/TS_New_Sample.txt', mode='r') as f:
        data_lines_2 = f.readlines()
    f.close()
    with open('img_data/S_New_Sample.txt', mode='r') as f:
        data_lines_3 = f.readlines()
    f.close()
    with open('img_data/S_Original.txt', mode='r') as f:
        data_lines_4 = f.readlines()
    f.close()
    label_data, purity_data = list(), list()
    for line in data_lines:
        data_model = line.split(', ')[0]
        label = int(data_model[data_model.index('_') + 1:data_model.index('.')])/90
        if label % 5 == 1:
            label_data.append(label)
            purity_evaluation = line.split(', ')[1]
            purity = float(purity_evaluation[purity_evaluation.index(' ') + 1:])
            purity_data.append(purity)

    label_data_2, purity_data_2 = list(), list()
    for line in data_lines_2:
        data_model = line.split(', ')[0]
        label = int(data_model[data_model.index('_') + 1:data_model.index('.')]) / 90
        if label % 5 == 1:
            label_data_2.append(label)
            purity_evaluation = line.split(', ')[1]
            purity = float(purity_evaluation[purity_evaluation.index(' ') + 1:])
            purity_data_2.append(purity)

    label_data_3, purity_data_3 = list(), list()
    for line in data_lines_3:
        data_model = line.split(', ')[0]
        label = int(data_model[data_model.index('_') + 1:data_model.index('.')]) / 90
        if label % 5 == 1:
            label_data_3.append(label)
            purity_evaluation = line.split(', ')[1]
            purity = float(purity_evaluation[purity_evaluation.index(' ') + 1:])
            purity_data_3.append(purity)

    label_data_4, purity_data_4 = list(), list()
    for line in data_lines_4:
        data_model = line.split(', ')[0]
        label = int(data_model[data_model.index('_') + 1:data_model.index('.')]) / 90
        if label % 5 == 1:
            label_data_4.append(label)
            purity_evaluation = line.split(', ')[1]
            purity = float(purity_evaluation[purity_evaluation.index(' ') + 1:])
            purity_data_4.append(purity)
    # plt.plot(label_data[0: len(label_data_3)], purity_data[0: len(label_data_3)], 'r', label='TS + Original Test Sample')
    # plt.plot(label_data_2[0: len(label_data_3)], purity_data_2[0: len(label_data_3)], 'b', label='TS + Our Test Sample')
    plt.plot(label_data_3, purity_data_3, 'lightskyblue', label='S + Our Test Sample')
    plt.plot(label_data_4, purity_data_4, 'lightcoral', label='S + Original Test Sample')
    plt.ylabel('Purity Score')
    plt.xlabel('Steps/90')
    plt.legend()
    plt.show()
    return


# Plot the graph of projects
def draw_line_projects():
    data_result = list()
    with open('img_data/Projects_FF_metrics.txt', 'r', encoding='utf8') as f:
        data_result = f.readlines()
    f.close()
    purity_list = [float(data.split(', ')[1].split(': ')[1]) for data in data_result]
    nmi_list = [float(data.split(', ')[2].split(': ')[1]) for data in data_result]
    ari_list = [float(data.split(', ')[3].split(': ')[1]) for data in data_result]
    shen_f_list = [float(data.replace('\n', '').split(', ')[4].split(': ')[1]) for data in data_result]
    project_list = []
    for i in range(8):
        project_list.append(f'P{i + 1}')
    project_list = [project_list] * 4
    metrics_list = [purity_list, nmi_list, ari_list, shen_f_list]
    color_list = ['r', 'g', 'b', 'black']
    label_list = ['Purity', 'NMI', 'ARI', 'Shen-F']

    # fig = plt.figure()
    # ax = axisartist.Subplot(fig, 111)
    # fig.add_axes(ax)
    # ax.axis[:].set_visible(False)
    # ax.axis["x"] = ax.new_floating_axis(0, 0)
    # ax.axis["x"].set_axisline_style("->", size=1.0)
    # ax.axis["y"] = ax.new_floating_axis(1, 0)
    # ax.axis["y"].set_axisline_style("-|>", size=1.0)
    # ax.axis["x"].set_axis_direction("top")
    # ax.axis["y"].set_axis_direction("right")

    figure = plt.figure()
    ax = axisartist.Subplot(figure, 111)
    figure.add_axes(ax)
    # plt.rc('font', family='Times New Roman')
    for project, metrics, color, label in zip(project_list, metrics_list, color_list, label_list):
        plt.plot(project, metrics, color=color, label=label)
    plt.legend(prop={'family': 'Times New Roman', 'size': 11})
    plt.xlabel('Projects', fontdict={'family': 'Times New Roman', 'size': 13})
    plt.ylabel('Metrics Score', fontdict={'family': 'Times New Roman', 'size': 13})
    plt.yticks(fontproperties='Times New Roman', size=12)
    plt.xticks(fontproperties='Times New Roman', size=12)
    ax = plt.gca()
    # ax.spines["top"].set_color("none")
    # ax.spines["right"].set_color("none")
    ax.axis["top"].set_visible(False)
    ax.axis["right"].set_visible(False)
    ax.axis["bottom"].set_axisline_style("-|>", size=1.5)
    ax.axis["left"].set_axisline_style("-|>", size=1.5)
    plt.show()
    return


def draw_metrics_baselines():
    figure = plt.figure()
    ax = axisartist.Subplot(figure, 111)
    figure.add_axes(ax)

    # Box Plot 1
    project_list = ['angular', 'Appium', 'dl4j', 'docker', 'ethereum', 'Gitter', 'Typescript', 'nodejs']
    project_metrics = dict()
    box_plot_list = []
    for project in project_list:
        data_result = list()
        with open(f'img_data/S_sample/S_New_Sample_{project}.txt', mode='r', encoding='utf8') as f:
            data_result = f.readlines()
        f.close()
        purity_list = [float(data.split(', ')[1].split(': ')[1]) for data in data_result]
        nmi_list = [float(data.split(', ')[2].split(': ')[1]) for data in data_result]
        ari_list = [float(data.split(', ')[3].split(': ')[1]) for data in data_result]
        shen_f_list = [float(data.replace('\n', '').split(', ')[4].split(': ')[1]) for data in data_result]
        project_metrics[project] = shen_f_list
        box_plot_list.append(shen_f_list)
    project_list = []
    for i in range(8):
        project_list.append(f'P{i + 1}')
    e2d_bilstm_list = [box_list[len(box_list) - 1] for box_list in box_plot_list]

    # project_list = [project_list] * 4
    # all_data = [np.random.normal(0, std, 100) for std in range(1, 4)]

    # plt.rc('font', family='Times New Roman')
    # for project, experiment, color, label in zip(project_list, metrics_list, color_list, label_list):
    #     plt.plot(project, experiment, color=color, label=label)

    # plt.boxplot(box_plot_list, positions=[1, 2, 3, 4, 5, 6, 7, 8], notch=True, widths=0.3, showfliers=False)
    # plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], project_list)
    data_result = list()


    # Box Plot 2
    project_list = ['angular', 'Appium', 'dl4j', 'docker', 'ethereum', 'Gitter', 'Typescript', 'nodejs']
    project_metrics = dict()
    box_plot_list_2 = []
    for project in project_list:
        data_result = list()
        with open(f'img_data/TS_sample/TS_New_Sample_{project}.txt', mode='r', encoding='utf8') as f:
            data_result = f.readlines()
        f.close()
        purity_list = [float(data.split(', ')[1].split(': ')[1]) for data in data_result]
        nmi_list = [float(data.split(', ')[2].split(': ')[1]) for data in data_result]
        ari_list = [float(data.split(', ')[3].split(': ')[1]) for data in data_result]
        shen_f_list = [float(data.replace('\n', '').split(', ')[4].split(': ')[1]) for data in data_result]
        project_metrics[project] = shen_f_list
        box_plot_list_2.append(shen_f_list)
    project_list = []
    for i in range(8):
        project_list.append(f'P{i + 1}')
    ts_bilstmlist = [box_list[len(box_list) - 1] for box_list in box_plot_list_2]
    # project_list = [project_list] * 4
    # all_data = [np.random.normal(0, std, 100) for std in range(1, 4)]

    # Box Plot 3
    project_list = ['angular', 'Appium', 'dl4j', 'docker', 'ethereum', 'Gitter', 'Typescript', 'nodejs']
    project_metrics = dict()
    box_plot_list_3 = []
    for project in project_list:
        data_result = list()
        with open(f'img_data/BERT_sample/DS_New_Sample_{project}.txt', mode='r', encoding='utf8') as f:
            data_result = f.readlines()
        f.close()
        purity_list = [float(data.split(', ')[1].split(': ')[1]) for data in data_result]
        nmi_list = [float(data.split(', ')[2].split(': ')[1]) for data in data_result]
        ari_list = [float(data.split(', ')[3].split(': ')[1]) for data in data_result]
        shen_f_list = [float(data.replace('\n', '').split(', ')[4].split(': ')[1]) for data in data_result]
        project_metrics[project] = shen_f_list
        box_plot_list_3.append(shen_f_list)
    project_list = []
    for i in range(8):
        project_list.append(f'P{i + 1}')
    ds_bilstmlist = [box_list[len(box_list) - 1] for box_list in box_plot_list_3]

    figure = plt.figure()
    ax = axisartist.Subplot(figure, 111)
    figure.add_axes(ax)
    # plt.rc('font', family='Times New Roman')
    # for project, experiment, color, label in zip(project_list, metrics_list, color_list, label_list):
    #     plt.plot(project, experiment, color=color, label=label)
    box_plot_final = []
    for i in range(len(box_plot_list)):
        box_plot_final.append(box_plot_list[i])
        box_plot_final.append(box_plot_list_2[i])
    # plt.boxplot(box_plot_final,
    #             positions=[0.8, 1.2, 1.8, 2.2, 2.8, 3.2, 3.8, 4.2, 4.8, 5.2, 5.8, 6.2, 6.8, 7.2, 7.8, 8.2], notch=True, widths=0.3, showfliers=False)
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], project_list)
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ds_bilstmlist, color='darksalmon', label='Bert', linestyle='--', marker='^', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], e2d_bilstm_list, color='limegreen', label='Liu\'s model', linestyle='--', marker='^', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ts_bilstmlist, color='orangered', label='Liu\'s model + DS & TS', linestyle='--', marker='^', markersize=4, mfcalt='b')
    data_result = list()


    # Line Graph 3
    with open('img_data/Project_FF_metrics_base.txt', 'r', encoding='utf8') as f:
        data_result = f.readlines()
    f.close()
    purity_list = [float(data.replace('\n', '').split(', ')[4].split(': ')[1]) for data in data_result]
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], purity_list, color='deepskyblue', label='Well-trained FF Model', linestyle='-.', marker='o', mfc='w', markersize=4, mfcalt='b')
    # del matplotlib.font_manager.weight_dict['roman']
    # matplotlib.font_manager._rebuild()
    ax = plt.gca()
    plt.grid(axis='y', linestyle='-.')
    # ax.spines["top"].set_color("none")
    # ax.spines["right"].set_color("none")



    ax.axis["top"].set_visible(False)
    ax.axis["right"].set_visible(False)
    ax.axis["bottom"].set_axisline_style("-|>", size=1.5)
    ax.axis["left"].set_axisline_style("-|>", size=1.5)
    # , prop={'family': 'Times New Roman'}
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, numpoints=1, fontsize=10)
    figure.subplots_adjust(right=0.63)
    plt.xlabel('Projects (Angular->NodeJS)', fontdict={'family': 'Times New Roman', 'size': 13})
    plt.ylabel('Shen-F', fontdict={'family': 'Times New Roman', 'size': 13})
    plt.ylim([0, 1])

    plt.yticks(fontproperties='Times New Roman', size=12)
    plt.xticks(fontproperties='Times New Roman', size=12)
    plt.show()
    return e2d_bilstm_list, ts_bilstmlist, ds_bilstmlist, purity_list


def draw_dlr_baseline():
    figure = plt.figure()
    gt_annotation_list = [0.6625, 0.5771, 0.5052, 0.435, 0.6125, 0.605, 0.5075, 0.5515]
    # subplot 1.1
    ax = axisartist.Subplot(figure, 231)
    figure.add_axes(ax)
    # Box Plot 1
    project_list = ['angular', 'Appium', 'dl4j', 'docker', 'ethereum', 'Gitter', 'Typescript', 'nodejs']
    box_plot_list_shen, box_plot_list_purity, box_plot_list_nmi, box_plot_list_ari = [], [], [], []
    for project in project_list:
        data_result = list()
        with open(f'img_data/S_sample/S_New_Sample_{project}.txt', mode='r', encoding='utf8') as f:
            data_result = f.readlines()
        f.close()
        purity_list = [float(data.split(', ')[1].split(': ')[1]) for data in data_result]
        nmi_list = [float(data.split(', ')[2].split(': ')[1]) for data in data_result]
        ari_list = [float(data.split(', ')[3].split(': ')[1]) for data in data_result]
        shen_f_list = [float(data.replace('\n', '').split(', ')[4].split(': ')[1]) for data in data_result]
        box_plot_list_purity.append(purity_list)
        box_plot_list_nmi.append(nmi_list)
        box_plot_list_ari.append(ari_list)
        box_plot_list_shen.append(shen_f_list)
    project_list = []
    for i in range(8):
        project_list.append(f'P{i + 1}')
    e2d_bilstm_list = [box_list[len(box_list) - 1] for box_list in box_plot_list_purity]

    # project_list = [project_list] * 4
    # all_data = [np.random.normal(0, std, 100) for std in range(1, 4)]

    # plt.rc('font', family='Times New Roman')
    # for project, experiment, color, label in zip(project_list, metrics_list, color_list, label_list):
    #     plt.plot(project, experiment, color=color, label=label)

    # plt.boxplot(box_plot_list, positions=[1, 2, 3, 4, 5, 6, 7, 8], notch=True, widths=0.3, showfliers=False)
    # plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], project_list)
    data_result = list()

    # Box Plot 2
    project_list = ['angular', 'Appium', 'dl4j', 'docker', 'ethereum', 'Gitter', 'Typescript', 'nodejs']
    box_plot_list_shen_2, box_plot_list_purity_2, box_plot_list_nmi_2, box_plot_list_ari_2 = [], [], [], []
    for project in project_list:
        data_result = list()
        with open(f'img_data/TS_sample/TS_New_Sample_{project}.txt', mode='r', encoding='utf8') as f:
            data_result = f.readlines()
        f.close()
        purity_list = [float(data.split(', ')[1].split(': ')[1]) for data in data_result]
        nmi_list = [float(data.split(', ')[2].split(': ')[1]) for data in data_result]
        ari_list = [float(data.split(', ')[3].split(': ')[1]) for data in data_result]
        shen_f_list = [float(data.replace('\n', '').split(', ')[4].split(': ')[1]) for data in data_result]
        box_plot_list_purity_2.append(purity_list)
        box_plot_list_nmi_2.append(nmi_list)
        box_plot_list_ari_2.append(ari_list)
        box_plot_list_shen_2.append(shen_f_list)
    project_list = []
    for i in range(8):
        project_list.append(f'P{i + 1}')
    ts_bilstmlist = [box_list[len(box_list) - 1] for box_list in box_plot_list_purity_2]
    # project_list = [project_list] * 4
    # all_data = [np.random.normal(0, std, 100) for std in range(1, 4)]

    # Box Plot 3
    project_list = ['angular', 'Appium', 'dl4j', 'docker', 'ethereum', 'Gitter', 'Typescript', 'nodejs']
    box_plot_list_shen_3, box_plot_list_purity_3, box_plot_list_nmi_3, box_plot_list_ari_3 = [], [], [], []
    for project in project_list:
        data_result = list()
        with open(f'img_data/BERT_sample/DS_New_Sample_{project}.txt', mode='r', encoding='utf8') as f:
            data_result = f.readlines()
        f.close()
        purity_list = [float(data.split(', ')[1].split(': ')[1]) for data in data_result]
        nmi_list = [float(data.split(', ')[2].split(': ')[1]) for data in data_result]
        ari_list = [float(data.split(', ')[3].split(': ')[1]) for data in data_result]
        shen_f_list = [float(data.replace('\n', '').split(', ')[4].split(': ')[1]) for data in data_result]
        box_plot_list_purity_3.append(purity_list)
        box_plot_list_nmi_3.append(nmi_list)
        box_plot_list_ari_3.append(ari_list)
        box_plot_list_shen_3.append(shen_f_list)
    project_list = []
    for i in range(8):
        project_list.append(f'P{i + 1}')
    ds_bilstmlist = [box_list[len(box_list) - 1] for box_list in box_plot_list_purity_3]

    # plt.rc('font', family='Times New Roman')
    # for project, experiment, color, label in zip(project_list, metrics_list, color_list, label_list):
    #     plt.plot(project, experiment, color=color, label=label)
    # box_plot_final = []
    # for i in range(len(box_plot_list)):
    #     box_plot_final.append(box_plot_list[i])
    #     box_plot_final.append(box_plot_list_2[i])
    # plt.boxplot(box_plot_final,
    #             positions=[0.8, 1.2, 1.8, 2.2, 2.8, 3.2, 3.8, 4.2, 4.8, 5.2, 5.8, 6.2, 6.8, 7.2, 7.8, 8.2], notch=True, widths=0.3, showfliers=False)
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], project_list)
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], gt_annotation_list, color='blue', label='Ground Truth Annotation', linestyle='-', marker='*',
             markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ds_bilstmlist, color='darksalmon', label='Bert', linestyle='--', marker='^', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], e2d_bilstm_list, color='limegreen', label='Liu\'s model', linestyle='--', marker='^', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ts_bilstmlist, color='orangered', label='Liu\'s model + DS & TS', linestyle='--', marker='^', markersize=4, mfcalt='b')
    data_result = list()

    # Line Graph 3
    with open('img_data/Project_FF_metrics_base.txt', 'r', encoding='utf8') as f:
        data_result = f.readlines()
    f.close()
    purity_ff_list = [float(data.replace('\n', '').split(', ')[1].split(': ')[1]) for data in data_result]
    nmi_ff_list = [float(data.replace('\n', '').split(', ')[2].split(': ')[1]) for data in data_result]
    ari_ff_list = [float(data.replace('\n', '').split(', ')[3].split(': ')[1]) for data in data_result]
    shen_f_ff_list = [float(data.replace('\n', '').split(', ')[4].split(': ')[1]) for data in data_result]
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], purity_ff_list, color='deepskyblue', label='Well-trained FF Model', linestyle='-.', marker='o', mfc='w', markersize=4, mfcalt='b')
    # del matplotlib.font_manager.weight_dict['roman']
    # matplotlib.font_manager._rebuild()
    # ax = plt.gca()
    plt.grid(axis='y', linestyle='-.')
    # ax.spines["top"].set_color("none")
    # ax.spines["right"].set_color("none")

    ax.axis["top"].set_visible(False)
    ax.axis["right"].set_visible(False)
    ax.axis["bottom"].set_axisline_style("-|>", size=1.5)
    ax.axis["left"].set_axisline_style("-|>", size=1.5)

    # , prop={'family': 'Times New Roman'}
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, numpoints=1, fontsize=10)
    # figure.subplots_adjust(right=0.63)
    plt.xlabel('Projects (Angular->NodeJS)', fontdict={'family': 'Times New Roman', 'size': 13})
    plt.ylabel('Purity Score', fontdict={'family': 'Times New Roman', 'size': 13})
    plt.ylim([0, 1])

    plt.yticks(fontproperties='Times New Roman', size=13)
    plt.xticks(fontproperties='Times New Roman', size=13)


    # Subplot 1.2
    ax = axisartist.Subplot(figure, 232)
    figure.add_axes(ax)
    e2d_bilstm_list = [box_list[len(box_list) - 1] for box_list in box_plot_list_nmi]
    ts_bilstmlist = [box_list[len(box_list) - 1] for box_list in box_plot_list_nmi_2]
    ds_bilstmlist = [box_list[len(box_list) - 1] for box_list in box_plot_list_nmi_3]
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], project_list)
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], gt_annotation_list, color='blue', linestyle='-', marker='*',
             markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ds_bilstmlist, color='darksalmon', linestyle='--', marker='^', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], e2d_bilstm_list, color='limegreen', linestyle='--', marker='^', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ts_bilstmlist, color='orangered', linestyle='--', marker='^', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], nmi_ff_list, color='deepskyblue', linestyle='-.', marker='o', mfc='w', markersize=4, mfcalt='b')
    ax = plt.gca()
    plt.grid(axis='y', linestyle='-.')
    # ax.spines["top"].set_color("none")
    # ax.spines["right"].set_color("none")

    ax.axis["top"].set_visible(False)
    ax.axis["right"].set_visible(False)
    ax.axis["bottom"].set_axisline_style("-|>", size=1.5)
    ax.axis["left"].set_axisline_style("-|>", size=1.5)
    # , prop={'family': 'Times New Roman'}
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, numpoints=1, fontsize=10)
    # figure.subplots_adjust(right=0.63)
    plt.xlabel('Projects (Angular->NodeJS)', fontdict={'family': 'Times New Roman', 'size': 13})
    plt.ylabel('NMI', fontdict={'family': 'Times New Roman', 'size': 13})
    plt.ylim([0, 1])

    plt.yticks(fontproperties='Times New Roman', size=12)
    plt.xticks(fontproperties='Times New Roman', size=12)


    # Subplot 1.3
    ax = axisartist.Subplot(figure, 233)
    figure.add_axes(ax)
    e2d_bilstm_list = [box_list[len(box_list) - 1] for box_list in box_plot_list_ari]
    ts_bilstmlist = [box_list[len(box_list) - 1] for box_list in box_plot_list_ari_2]
    ds_bilstmlist = [box_list[len(box_list) - 1] for box_list in box_plot_list_ari_3]
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], project_list)
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], gt_annotation_list, color='blue', linestyle='-', marker='*',
             markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ds_bilstmlist, color='darksalmon', linestyle='--', marker='^',
             markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], e2d_bilstm_list, color='limegreen', linestyle='--',
             marker='^', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ts_bilstmlist, color='orangered', linestyle='--',
             marker='^', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ari_ff_list, color='deepskyblue', linestyle='-.',
             marker='o', mfc='w', markersize=4, mfcalt='b')
    # ax = plt.gca()
    plt.grid(axis='y', linestyle='-.')
    # ax.spines["top"].set_color("none")
    # ax.spines["right"].set_color("none")

    ax.axis["top"].set_visible(False)
    ax.axis["right"].set_visible(False)
    ax.axis["bottom"].set_axisline_style("-|>", size=1.5)
    ax.axis["left"].set_axisline_style("-|>", size=1.5)
    # , prop={'family': 'Times New Roman'}
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, numpoints=1, fontsize=10)
    # figure.subplots_adjust(right=0.63)
    plt.xlabel('Projects (Angular->NodeJS)', fontdict={'family': 'Times New Roman', 'size': 13})
    plt.ylabel('ARI', fontdict={'family': 'Times New Roman', 'size': 13})
    plt.ylim([0, 1])

    plt.yticks(fontproperties='Times New Roman', size=12)
    plt.xticks(fontproperties='Times New Roman', size=12)



    # Subplot 1.4
    ax = axisartist.Subplot(figure, 234)
    figure.add_axes(ax)
    e2d_bilstm_list = [box_list[len(box_list) - 1] for box_list in box_plot_list_shen]
    ts_bilstmlist = [box_list[len(box_list) - 1] for box_list in box_plot_list_shen_2]
    ds_bilstmlist = [box_list[len(box_list) - 1] for box_list in box_plot_list_shen_3]
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], project_list)
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], gt_annotation_list, color='blue', linestyle='-', marker='*',
             markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ds_bilstmlist, color='darksalmon', linestyle='--', marker='^',
             markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], e2d_bilstm_list, color='limegreen', linestyle='--',
             marker='^', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ts_bilstmlist, color='orangered', linestyle='--',
             marker='^', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], shen_f_ff_list, color='deepskyblue', linestyle='-.',
             marker='o', mfc='w', markersize=4, mfcalt='b')
    ax = plt.gca()
    plt.grid(axis='y', linestyle='-.')
    # ax.spines["top"].set_color("none")
    # ax.spines["right"].set_color("none")

    ax.axis["top"].set_visible(False)
    ax.axis["right"].set_visible(False)
    ax.axis["bottom"].set_axisline_style("-|>", size=1.5)
    ax.axis["left"].set_axisline_style("-|>", size=1.5)
    # , prop={'family': 'Times New Roman'}
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, numpoints=1, fontsize=10)
    # figure.subplots_adjust(right=0.63)
    plt.xlabel('Projects (Angular->NodeJS)', fontdict={'family': 'Times New Roman', 'size': 13})
    plt.ylabel('Shen-F', fontdict={'family': 'Times New Roman', 'size': 13})
    plt.ylim([0, 1])

    plt.yticks(fontproperties='Times New Roman', size=12)
    plt.xticks(fontproperties='Times New Roman', size=12)



    # Subplot 2
    ax1 = axisartist.Subplot(figure, 235)
    figure.add_axes(ax1)

    # Box Plot 1
    data_result = list()
    with open('img_data/Dialog_Levenshtein_Dist/E2D_DS_TS_result.txt', mode='r', encoding='utf8') as f:
        data_result = f.readlines()
    f.close()
    ts_bilstmlist = [float(data.split(', ')[1].split(': ')[1].replace('\n', '')) for data in data_result]

    # project_list = [project_list] * 4
    # all_data = [np.random.normal(0, std, 100) for std in range(1, 4)]

    # plt.rc('font', family='Times New Roman')
    # for project, experiment, color, label in zip(project_list, metrics_list, color_list, label_list):
    #     plt.plot(project, experiment, color=color, label=label)

    # plt.boxplot(box_plot_list, positions=[1, 2, 3, 4, 5, 6, 7, 8], notch=True, widths=0.3, showfliers=False)
    # plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], project_list)
    # data_result = list()

    # Box Plot 2
    data_result = list()
    with open('img_data/Dialog_Levenshtein_Dist/BiLSTM_result.txt', mode='r', encoding='utf8') as f:
        data_result = f.readlines()
    f.close()
    e2d_bilstm_list = [float(data.split(', ')[1].split(': ')[1].replace('\n', '')) for data in data_result]
    # e2d_bilstm_list = [box_list[len(box_list) - 1] for box_list in box_plot_list_2]
    # project_list = [project_list] * 4
    # all_data = [np.random.normal(0, std, 100) for std in range(1, 4)]

    # Box Plot 3
    data_result = list()
    with open('img_data/Dialog_Levenshtein_Dist/E2D_DS_result.txt', mode='r', encoding='utf8') as f:
        data_result = f.readlines()
    f.close()
    ds_bilstmlist = [float(data.split(', ')[1].split(': ')[1].replace('\n', '')) for data in data_result]
    # ds_bilstmlist = [box_list[len(box_list) - 1] for box_list in box_plot_list_3]

    # figure = plt.figure()
    # ax = axisartist.Subplot(figure, 111)
    # figure.add_axes(ax)

    # plt.rc('font', family='Times New Roman')
    # for project, experiment, color, label in zip(project_list, metrics_list, color_list, label_list):
    #     plt.plot(project, experiment, color=color, label=label)

    project_list = []
    for i in range(8):
        project_list.append(f'P{i + 1}')
    # plt.boxplot(box_plot_final,
    #             positions=[0.8, 1.2, 1.8, 2.2, 2.8, 3.2, 3.8, 4.2, 4.8, 5.2, 5.8, 6.2, 6.8, 7.2, 7.8, 8.2], notch=True, widths=0.3, showfliers=False)
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], project_list)
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], gt_annotation_list, color='blue', linestyle='-', marker='*',
             markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ds_bilstmlist, color='darksalmon', linestyle='--', marker='^', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], e2d_bilstm_list, color='limegreen', linestyle='--', marker='^', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ts_bilstmlist, color='orangered', linestyle='--', marker='^', markersize=4, mfcalt='b')
    data_result = list()

    # Line Graph 3
    data_result = list()
    with open('img_data/Dialog_Levenshtein_Dist/FF_result.txt', mode='r', encoding='utf8') as f:
        data_result = f.readlines()
    f.close()
    ff_list = [float(data.split(', ')[1].split(': ')[1].replace('\n', '')) for data in data_result]
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ff_list, color='deepskyblue', linestyle='-.', marker='o', mfc='w', markersize=4, mfcalt='b')
    # del matplotlib.font_manager.weight_dict['roman']
    # matplotlib.font_manager._rebuild()
    ax1 = plt.gca()
    plt.grid(axis='y', linestyle='-.')
    # ax.spines["top"].set_color("none")
    # ax.spines["right"].set_color("none")

    ax1.axis["top"].set_visible(False)
    ax1.axis["right"].set_visible(False)
    ax1.axis["bottom"].set_axisline_style("-|>", size=1.5)
    ax1.axis["left"].set_axisline_style("-|>", size=1.5)
    # , prop={'family': 'Times New Roman'}
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, numpoints=1, fontsize=10)
    # figure.subplots_adjust(right=0.63)

    plt.xlabel('Projects (Angular->NodeJS)', fontdict={'family': 'Times New Roman', 'size': 13})
    plt.ylabel('DLD Score', fontdict={'family': 'Times New Roman', 'size': 13})
    plt.ylim([0, 1])

    plt.yticks(fontproperties='Times New Roman', size=12)
    plt.xticks(fontproperties='Times New Roman', size=12)



    # Subplot 3
    ax2 = axisartist.Subplot(figure, 236)
    figure.add_axes(ax2)

    # Box Plot 1
    data_result = list()
    with open('img_data/Dialog_Levenshtein_Ratio/E2D_DS_TS_result.txt', mode='r', encoding='utf8') as f:
        data_result = f.readlines()
    f.close()
    ts_bilstmlist = [float(data.split(', ')[1].split(': ')[1].replace('\n', '')) for data in data_result]

    # project_list = [project_list] * 4
    # all_data = [np.random.normal(0, std, 100) for std in range(1, 4)]

    # plt.rc('font', family='Times New Roman')
    # for project, experiment, color, label in zip(project_list, metrics_list, color_list, label_list):
    #     plt.plot(project, experiment, color=color, label=label)

    # plt.boxplot(box_plot_list, positions=[1, 2, 3, 4, 5, 6, 7, 8], notch=True, widths=0.3, showfliers=False)
    # plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], project_list)
    # data_result = list()

    # Box Plot 2
    data_result = list()
    with open('img_data/Dialog_Levenshtein_Ratio/BiLSTM_result.txt', mode='r', encoding='utf8') as f:
        data_result = f.readlines()
    f.close()
    e2d_bilstm_list = [float(data.split(', ')[1].split(': ')[1].replace('\n', '')) for data in data_result]
    # e2d_bilstm_list = [box_list[len(box_list) - 1] for box_list in box_plot_list_2]
    # project_list = [project_list] * 4
    # all_data = [np.random.normal(0, std, 100) for std in range(1, 4)]

    # Box Plot 3
    data_result = list()
    with open('img_data/Dialog_Levenshtein_Ratio/E2D_DS_result.txt', mode='r', encoding='utf8') as f:
        data_result = f.readlines()
    f.close()
    ds_bilstmlist = [float(data.split(', ')[1].split(': ')[1].replace('\n', '')) for data in data_result]
    # ds_bilstmlist = [box_list[len(box_list) - 1] for box_list in box_plot_list_3]

    # figure = plt.figure()
    # ax = axisartist.Subplot(figure, 111)
    # figure.add_axes(ax)

    # plt.rc('font', family='Times New Roman')
    # for project, experiment, color, label in zip(project_list, metrics_list, color_list, label_list):
    #     plt.plot(project, experiment, color=color, label=label)

    project_list = []
    for i in range(8):
        project_list.append(f'P{i + 1}')
    # plt.boxplot(box_plot_final,
    #             positions=[0.8, 1.2, 1.8, 2.2, 2.8, 3.2, 3.8, 4.2, 4.8, 5.2, 5.8, 6.2, 6.8, 7.2, 7.8, 8.2], notch=True, widths=0.3, showfliers=False)
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], project_list)
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], gt_annotation_list, color='blue', linestyle='-', marker='*',
             markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ds_bilstmlist, color='darksalmon', linestyle='--', marker='^',
             markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], e2d_bilstm_list, color='limegreen', linestyle='--',
             marker='^', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ts_bilstmlist, color='orangered', linestyle='--',
             marker='^', markersize=4, mfcalt='b')
    data_result = list()

    # Line Graph 3
    data_result = list()
    with open('img_data/Dialog_Levenshtein_Ratio/FF_result.txt', mode='r', encoding='utf8') as f:
        data_result = f.readlines()
    f.close()
    ff_list = [float(data.split(', ')[1].split(': ')[1].replace('\n', '')) for data in data_result]
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ff_list, color='deepskyblue', linestyle='-.',
             marker='o', mfc='w', markersize=4, mfcalt='b')
    # del matplotlib.font_manager.weight_dict['roman']
    # matplotlib.font_manager._rebuild()
    ax2 = plt.gca()
    plt.grid(axis='y', linestyle='-.')
    # ax.spines["top"].set_color("none")
    # ax.spines["right"].set_color("none")

    ax2.axis["top"].set_visible(False)
    ax2.axis["right"].set_visible(False)
    ax2.axis["bottom"].set_axisline_style("-|>", size=1.5)
    ax2.axis["left"].set_axisline_style("-|>", size=1.5)
    # , prop={'family': 'Times New Roman'}
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, numpoints=1, fontsize=10)
    # figure.subplots_adjust(right=0.63)
    plt.xlabel('Projects (Angular->NodeJS)', fontdict={'family': 'Times New Roman', 'size': 13})
    plt.ylabel('DLR Score', fontdict={'family': 'Times New Roman', 'size': 13})
    plt.ylim([0, 1])

    plt.yticks(fontproperties='Times New Roman', size=12)
    plt.xticks(fontproperties='Times New Roman', size=12)
    figure.legend(loc='upper center', ncol=5, prop = {'size':16})
    plt.show()
    return e2d_bilstm_list, ts_bilstmlist, ds_bilstmlist, ff_list


def draw_final_result():
    figure = plt.figure()
    ff_total_migration, ff_total_retraining = [], []
    bilstm_total_migration, bilstm_total_retraining = [], []
    bert_total_migration, bert_total_retraining = [], []
    e2e_total_migration, e2e_total_retraining = [], []
    ptr_total_migration, ptr_total_retraining = [], []
    # subplot 1.1
    plt.subplots_adjust(wspace=0.3)

    plt.subplot(141)
    # Box Plot 1
    project_list = ['angular', 'Appium', 'dl4j', 'docker', 'ethereum', 'Gitter', 'Typescript', 'nodejs']
    box_plot_list_shen, box_plot_list_purity, box_plot_list_nmi, box_plot_list_ari = [], [], [], []
    for project in project_list:
        data_result = list()
        with open(f'img_data/BiLSTM_sample/S_New_Sample_{project}.txt', mode='r', encoding='utf8') as f:
            data_result = f.readlines()
        f.close()
        purity_list = [float(data.split(', ')[1].split(': ')[1]) for data in data_result]
        nmi_list = [float(data.split(', ')[2].split(': ')[1]) for data in data_result]
        ari_list = [float(data.split(', ')[3].split(': ')[1]) for data in data_result]
        shen_f_list = [float(data.replace('\n', '').split(', ')[4].split(': ')[1]) for data in data_result]
        box_plot_list_purity.append(purity_list)
        box_plot_list_nmi.append(nmi_list)
        box_plot_list_ari.append(ari_list)
        box_plot_list_shen.append(shen_f_list)
    project_list = []
    for i in range(8):
        project_list.append(f'P{i + 1}')
    e2d_bilstm_list = [box_list[int(len(box_list)/2)] for box_list in box_plot_list_purity]

    # project_list = [project_list] * 4
    # all_data = [np.random.normal(0, std, 100) for std in range(1, 4)]

    # plt.rc('font', family='Times New Roman')
    # for project, experiment, color, label in zip(project_list, metrics_list, color_list, label_list):
    #     plt.plot(project, experiment, color=color, label=label)

    # plt.boxplot(box_plot_list, positions=[1, 2, 3, 4, 5, 6, 7, 8], notch=True, widths=0.3, showfliers=False)
    # plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], project_list)
    data_result = list()

    # Box Plot 2
    project_list = ['angular', 'Appium', 'dl4j', 'docker', 'ethereum', 'Gitter', 'Typescript', 'nodejs']
    box_plot_list_shen_2, box_plot_list_purity_2, box_plot_list_nmi_2, box_plot_list_ari_2 = [], [], [], []
    for project in project_list:
        data_result = list()
        with open(f'img_data/E2E_sample/TS_New_Sample_{project}.txt', mode='r', encoding='utf8') as f:
            data_result = f.readlines()
        f.close()
        purity_list = [float(data.split(', ')[1].split(': ')[1]) for data in data_result]
        nmi_list = [float(data.split(', ')[2].split(': ')[1]) for data in data_result]
        ari_list = [float(data.split(', ')[3].split(': ')[1]) for data in data_result]
        shen_f_list = [float(data.replace('\n', '').split(', ')[4].split(': ')[1]) for data in data_result]
        box_plot_list_purity_2.append(purity_list)
        box_plot_list_nmi_2.append(nmi_list)
        box_plot_list_ari_2.append(ari_list)
        box_plot_list_shen_2.append(shen_f_list)
    project_list = []
    for i in range(8):
        project_list.append(f'P{i + 1}')
    ts_bilstmlist = [box_list[len(box_list) - 1] for box_list in box_plot_list_purity_2]
    # project_list = [project_list] * 4
    # all_data = [np.random.normal(0, std, 100) for std in range(1, 4)]

    # Box Plot 3
    project_list = ['angular', 'Appium', 'dl4j', 'docker', 'ethereum', 'Gitter', 'Typescript', 'nodejs']
    box_plot_list_shen_3, box_plot_list_purity_3, box_plot_list_nmi_3, box_plot_list_ari_3 = [], [], [], []
    for project in project_list:
        data_result = list()
        with open(f'img_data/BERT_sample/DS_New_Sample_{project}.txt', mode='r', encoding='utf8') as f:
            data_result = f.readlines()
        f.close()
        purity_list = [float(data.split(', ')[1].split(': ')[1]) for data in data_result]
        nmi_list = [float(data.split(', ')[2].split(': ')[1]) for data in data_result]
        ari_list = [float(data.split(', ')[3].split(': ')[1]) for data in data_result]
        shen_f_list = [float(data.replace('\n', '').split(', ')[4].split(': ')[1]) for data in data_result]
        box_plot_list_purity_3.append(purity_list)
        box_plot_list_nmi_3.append(nmi_list)
        box_plot_list_ari_3.append(ari_list)
        box_plot_list_shen_3.append(shen_f_list)
    project_list = []
    for i in range(8):
        project_list.append(f'P{i + 1}')
    ds_bilstmlist = [box_list[len(box_list) - 1] for box_list in box_plot_list_purity_3]

    # plt.rc('font', family='Times New Roman')
    # for project, experiment, color, label in zip(project_list, metrics_list, color_list, label_list):
    #     plt.plot(project, experiment, color=color, label=label)
    # box_plot_final = []
    # for i in range(len(box_plot_list)):
    #     box_plot_final.append(box_plot_list[i])
    #     box_plot_final.append(box_plot_list_2[i])
    # plt.boxplot(box_plot_final,
    #             positions=[0.8, 1.2, 1.8, 2.2, 2.8, 3.2, 3.8, 4.2, 4.8, 5.2, 5.8, 6.2, 6.8, 7.2, 7.8, 8.2], notch=True, widths=0.3, showfliers=False)

    data_result = list()

    # Line Graph 3
    with open('img_data/Project_FF_metrics_base.txt', 'r', encoding='utf8') as f:
        data_result = f.readlines()
    f.close()
    purity_ff_list = [float(data.replace('\n', '').split(', ')[1].split(': ')[1]) for data in data_result]
    nmi_ff_list = [float(data.replace('\n', '').split(', ')[2].split(': ')[1]) for data in data_result]
    ari_ff_list = [float(data.replace('\n', '').split(', ')[3].split(': ')[1]) for data in data_result]
    shen_f_ff_list = [float(data.replace('\n', '').split(', ')[4].split(': ')[1]) for data in data_result]

    project_list_temp = ['angular', 'Appium', 'dl4j', 'docker', 'ethereum', 'Gitter', 'Typescript', 'nodejs']
    e2d_bilstm_list, ds_bilstmlist, ts_bilstmlist, ff_list = [], [], [], []
    for project_data in project_list_temp:
        baseline_name_list = ['BiLSTM', 'BERT', 'E2E', 'FF']
        for baseline_name in baseline_name_list:
            with open(f'img_data/P_R_F/{baseline_name}_{project_data}_PRF.txt', mode='r', encoding='utf8') as f:
                metric_list = f.readlines()
                f_sentence = metric_list[len(metric_list) - 1]
                f_data = float(f_sentence.replace('\n', '').split(', ')[1].split(': ')[1])
            f.close()
            if baseline_name == 'BiLSTM':
                e2d_bilstm_list.append(f_data)
            elif baseline_name == 'BERT':
                ds_bilstmlist.append(f_data)
            elif baseline_name == 'E2E':
                ts_bilstmlist.append(f_data)
            elif baseline_name == 'FF':
                ff_list.append(f_data)
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], project_list)
    ptr_net_list = [0.44, 0.42, 0.4, 0.57, 0.58, 0.43, 0.36, 0.4]
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], e2d_bilstm_list, color='limegreen', label='BiLSTM', linestyle='-', marker='s', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ds_bilstmlist, color='darksalmon', label='BERT', linestyle='-', marker='x', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ts_bilstmlist, color='orangered', label='E2E', linestyle='-', marker='^', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ptr_net_list, color='purple', label='PtrNet', linestyle='-', marker='+', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ff_list, color='deepskyblue', label='FF', linestyle='-', marker='o', mfc='w', markersize=4, mfcalt='b')
    # plt.plot([1, 2, 3, 4, 5, 6, 7, 8], gt_annotation_list, color='blue', label='Annotation G-T Score (FeedForward)', linestyle='-', marker='*',
    #          markersize=4, mfcalt='b')
    # del matplotlib.font_manager.weight_dict['roman']
    # matplotlib.font_manager._rebuild()
    # ax = plt.gca()
    plt.grid(axis='y', linestyle='-.')
    # ax.spines["top"].set_color("none")
    # ax.spines["right"].set_color("none")

    # , prop={'family': 'Times New Roman'}
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, numpoints=1, fontsize=10)
    # figure.subplots_adjust(right=0.63)
    # plt.xlabel('Projects (Angular->NodeJS)', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.ylabel('F1', fontdict={'family': 'Times New Roman', 'size': 25})
    plt.ylim([0, 1])
    print(np.array(ff_list).mean())
    bilstm_total_migration += e2d_bilstm_list
    bert_total_migration += ds_bilstmlist
    e2e_total_migration += ts_bilstmlist
    ptr_total_migration += ptr_net_list
    plt.yticks(fontproperties='Times New Roman', size=25)
    plt.xticks(fontproperties='Times New Roman', size=25)


    # Subplot 1.2
    plt.subplot(142)
    e2d_bilstm_list = [box_list[int(len(box_list)/2)] for box_list in box_plot_list_nmi]
    ts_bilstmlist = [box_list[len(box_list) - 1] for box_list in box_plot_list_nmi_2]
    ds_bilstmlist = [box_list[len(box_list) - 1] for box_list in box_plot_list_nmi_3]
    ptr_net_list = [0.71, 0.7, 0.65, 0.8, 0.79, 0.69, 0.62, 0.68]
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], project_list)
    # plt.plot([1, 2, 3, 4, 5, 6, 7, 8], gt_annotation_list, color='blue', linestyle='-', marker='*',
    #          markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], e2d_bilstm_list, color='limegreen', linestyle='-', marker='s', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ds_bilstmlist, color='darksalmon', linestyle='-', marker='x', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ts_bilstmlist, color='orangered', linestyle='-', marker='^', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], nmi_ff_list, color='deepskyblue', linestyle='-', marker='o', mfc='w', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ptr_net_list, color='purple', linestyle='-', marker='+', markersize=4, mfcalt='b')
    plt.grid(axis='y', linestyle='-.')
    # ax.spines["top"].set_color("none")
    # ax.spines["right"].set_color("none")
    # , prop={'family': 'Times New Roman'}
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, numpoints=1, fontsize=10)
    # figure.subplots_adjust(right=0.63)
    # plt.xlabel('Projects (Angular->NodeJS)', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.ylabel('NMI', fontdict={'family': 'Times New Roman', 'size': 25})
    plt.ylim([0, 1])
    bilstm_total_migration += e2d_bilstm_list
    bert_total_migration += ds_bilstmlist
    e2e_total_migration += ts_bilstmlist
    ptr_total_migration += ptr_net_list
    plt.yticks(fontproperties='Times New Roman', size=25)
    plt.xticks(fontproperties='Times New Roman', size=25)


    # Subplot 1.3
    plt.subplot(143)
    e2d_bilstm_list = [box_list[int(len(box_list)/2)] for box_list in box_plot_list_ari]
    ts_bilstmlist = [box_list[len(box_list) - 1] for box_list in box_plot_list_ari_2]
    ds_bilstmlist = [box_list[len(box_list) - 1] for box_list in box_plot_list_ari_3]
    ptr_net_list = [0.62, 0.40, 0.43, 0.6, 0.62, 0.57, 0.58, 0.60]
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], project_list)
    # plt.plot([1, 2, 3, 4, 5, 6, 7, 8], gt_annotation_list, color='blue', linestyle='-', marker='*',
    #          markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], e2d_bilstm_list, color='limegreen', linestyle='-',
             marker='s', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ds_bilstmlist, color='darksalmon', linestyle='-', marker='x',
             markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ts_bilstmlist, color='orangered', linestyle='-',
             marker='^', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ari_ff_list, color='deepskyblue', linestyle='-',
             marker='o', mfc='w', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ptr_net_list, color='purple', linestyle='-', marker='+', markersize=4, mfcalt='b')
    # ax = plt.gca()
    plt.grid(axis='y', linestyle='-.')
    # ax.spines["top"].set_color("none")
    # ax.spines["right"].set_color("none")
    # , prop={'family': 'Times New Roman'}
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, numpoints=1, fontsize=10)
    # figure.subplots_adjust(right=0.63)
    # plt.xlabel('Projects (Angular->NodeJS)', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.ylabel('ARI', fontdict={'family': 'Times New Roman', 'size': 25})
    plt.ylim([0, 1])
    bilstm_total_migration += e2d_bilstm_list
    bert_total_migration += ds_bilstmlist
    e2e_total_migration += ts_bilstmlist
    ptr_total_migration += ptr_net_list
    plt.yticks(fontproperties='Times New Roman', size=25)
    plt.xticks(fontproperties='Times New Roman', size=25)



    # Subplot 1.4
    plt.subplot(144)
    e2d_bilstm_list = [box_list[int(len(box_list)/2)] for box_list in box_plot_list_shen]
    ts_bilstmlist = [box_list[len(box_list) - 1] for box_list in box_plot_list_shen_2]
    ds_bilstmlist = [box_list[len(box_list) - 1] for box_list in box_plot_list_shen_3]
    ptr_net_list = [0.85, 0.77, 0.76, 0.79, 0.78, 0.82, 0.80, 0.79]
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], project_list)
    # plt.plot([1, 2, 3, 4, 5, 6, 7, 8], gt_annotation_list, color='blue', linestyle='-', marker='*',
    #          markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], e2d_bilstm_list, color='limegreen', linestyle='-',
             marker='s', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ds_bilstmlist, color='darksalmon', linestyle='-', marker='x',
             markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ts_bilstmlist, color='orangered', linestyle='-',
             marker='^', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], shen_f_ff_list, color='deepskyblue', linestyle='-',
             marker='o', mfc='w', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ptr_net_list, color='purple', linestyle='-', marker='+', markersize=4, mfcalt='b')

    plt.grid(axis='y', linestyle='-.')
    # ax.spines["top"].set_color("none")
    # ax.spines["right"].set_color("none")

    # , prop={'family': 'Times New Roman'}
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, numpoints=1, fontsize=10)
    # figure.subplots_adjust(right=0.63)
    # plt.xlabel('Projects (Angular->NodeJS)', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.ylabel('Shen-F', fontdict={'family': 'Times New Roman', 'size': 25})
    plt.ylim([0, 1])
    bilstm_total_migration += e2d_bilstm_list
    bert_total_migration += ds_bilstmlist
    e2e_total_migration += ts_bilstmlist
    ptr_total_migration += ptr_net_list
    plt.yticks(fontproperties='Times New Roman', size=25)
    plt.xticks(fontproperties='Times New Roman', size=25)

    ff_total_migration += (ff_list + nmi_ff_list + ari_ff_list + shen_f_ff_list)

    dic_ff_result = {'f1': ff_list, 'nmi': nmi_ff_list, 'ari': ari_ff_list, 'shen-f': shen_f_ff_list}
    figure.legend(loc='upper center', ncol=8, prop={'size': 20, 'family': 'Times New Roman'})
    # figure.legend(loc='center right', ncol=1, prop={'size': 20, 'family': 'Times New Roman'})
    # figure.legend(loc='upper center', ncol=8, bbox_to_anchor=(0.5, 1.2), prop={'size': 20, 'family': 'Times New Roman'})

    plt.show()








    figure = plt.figure()
    plt.subplots_adjust(wspace=0.3)


    plt.subplot(141)
    # Box Plot 1
    project_list = ['angular', 'Appium', 'dl4j', 'docker', 'ethereum', 'Gitter', 'Typescript', 'nodejs']

    # Box Plot 3
    project_list = []
    for i in range(8):
        project_list.append(f'P{i + 1}')

    # plt.rc('font', family='Times New Roman')
    # for project, experiment, color, label in zip(project_list, metrics_list, color_list, label_list):
    #     plt.plot(project, experiment, color=color, label=label)
    # box_plot_final = []
    # for i in range(len(box_plot_list)):
    #     box_plot_final.append(box_plot_list[i])
    #     box_plot_final.append(box_plot_list_2[i])
    # plt.boxplot(box_plot_final,
    #             positions=[0.8, 1.2, 1.8, 2.2, 2.8, 3.2, 3.8, 4.2, 4.8, 5.2, 5.8, 6.2, 6.8, 7.2, 7.8, 8.2], notch=True, widths=0.3, showfliers=False)

    data_result = list()

    # Line Graph 3
    with open('img_data/retrain/Project_FF_metrics.txt', 'r', encoding='utf8') as f:
        data_result = f.readlines()
    f.close()
    purity_ff_list = [float(data.replace('\n', '').split(', ')[1].split(': ')[1]) for data in data_result]
    nmi_ff_list = [float(data.replace('\n', '').split(', ')[2].split(': ')[1]) for data in data_result]
    ari_ff_list = [float(data.replace('\n', '').split(', ')[3].split(': ')[1]) for data in data_result]
    shen_f_ff_list = [float(data.replace('\n', '').split(', ')[4].split(': ')[1]) for data in data_result]

    project_list_temp = ['angular', 'Appium', 'dl4j', 'docker', 'ethereum', 'Gitter', 'Typescript', 'nodejs']
    e2d_bilstm_list, ds_bilstmlist, ts_bilstmlist, ff_list = [], [], [], []
    for project_data in project_list_temp:
        baseline_name_list = ['BiLSTM', 'Bert', 'E2E_Online_Liu', 'FF']
        for baseline_name in baseline_name_list:
            with open(f'img_data/retrain/P_R_F/{baseline_name}_{project_data}_PRF.txt', mode='r', encoding='utf8') as f:
                metric_list = f.readlines()
                f_sentence = metric_list[len(metric_list) - 1]
                f_data = float(f_sentence.replace('\n', '').split(', ')[1].split(': ')[1])
            f.close()
            if baseline_name == 'BiLSTM':
                e2d_bilstm_list.append(f_data)
            elif baseline_name == 'Bert':
                ds_bilstmlist.append(f_data)
            elif baseline_name == 'E2E_Online_Liu':
                ts_bilstmlist.append(f_data)
            elif baseline_name == 'FF':
                ff_list.append(f_data)
    ptr_net_list = [0.55, 0.58, 0.45, 0.42, 0.63, 0.56, 0.53, 0.45]
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], project_list)
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], e2d_bilstm_list, color='limegreen', label='BiLSTM', linestyle='-', marker='s', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ds_bilstmlist, color='darksalmon', label='BERT', linestyle='-', marker='x', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ts_bilstmlist, color='orangered', label='E2E', linestyle='-', marker='^', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ff_list, color='deepskyblue', label='FF', linestyle='-', marker='o', mfc='w', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ptr_net_list, color='purple', label='PtrNet', linestyle='-', marker='+', markersize=4, mfcalt='b')
    # plt.plot([1, 2, 3, 4, 5, 6, 7, 8], gt_annotation_list, color='blue', label='Annotation G-T Score (FeedForward)', linestyle='-', marker='*',
    #          markersize=4, mfcalt='b')
    # del matplotlib.font_manager.weight_dict['roman']
    # matplotlib.font_manager._rebuild()
    # ax = plt.gca()
    plt.grid(axis='y', linestyle='-.')
    print(np.array(ff_list).mean())
    # ax.spines["top"].set_color("none")
    # ax.spines["right"].set_color("none")
    bilstm_total_retraining += e2d_bilstm_list
    bert_total_retraining += ds_bilstmlist
    e2e_total_retraining += ts_bilstmlist
    ptr_total_retraining += ptr_net_list
    # , prop={'family': 'Times New Roman'}
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, numpoints=1, fontsize=10)
    # figure.subplots_adjust(right=0.63)
    # plt.xlabel('Projects', fontdict={'family': 'Times New Roman', 'size': 25})
    plt.ylabel('F1', fontdict={'family': 'Times New Roman', 'size': 25})
    plt.ylim([0, 1])

    plt.yticks(fontproperties='Times New Roman', size=25)
    plt.xticks(fontproperties='Times New Roman', size=25)


    # Subplot 1.2
    plt.subplot(142)
    with open('img_data/retrain/retrain_cluster/BiLSTM_Liu.txt', mode='r', encoding='utf8') as f:
        data_lines = f.readlines()
        nmi_bilstm_list = [float(data.replace('\n', '').split(', ')[2].split(': ')[1]) for data in data_lines]
        ari_bilstm_list = [float(data.replace('\n', '').split(', ')[3].split(': ')[1]) for data in data_lines]
        shen_f_bilstm_list = [float(data.replace('\n', '').split(', ')[4].split(': ')[1]) for data in data_lines]
    f.close()
    with open('img_data/retrain/retrain_cluster/Bert.txt', mode='r', encoding='utf8') as f:
        data_lines = f.readlines()
        nmi_bert_list = [float(data.replace('\n', '').split(', ')[2].split(': ')[1]) for data in data_lines]
        ari_bert_list = [float(data.replace('\n', '').split(', ')[3].split(': ')[1]) for data in data_lines]
        shen_f_bert_list = [float(data.replace('\n', '').split(', ')[4].split(': ')[1]) for data in data_lines]
    f.close()
    with open('img_data/retrain/retrain_cluster/E2E_Online_Liu.txt', mode='r', encoding='utf8') as f:
        data_lines = f.readlines()
        nmi_e2e_list = [float(data.replace('\n', '').split(', ')[2].split(': ')[1]) for data in data_lines]
        ari_e2e_list = [float(data.replace('\n', '').split(', ')[3].split(': ')[1]) for data in data_lines]
        shen_f_e2e_list = [float(data.replace('\n', '').split(', ')[4].split(': ')[1]) for data in data_lines]
    f.close()
    ptr_net_list = [0.77, 0.656, 0.62, 0.65, 0.79, 0.77, 0.6, 0.67]
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], project_list)
    # plt.plot([1, 2, 3, 4, 5, 6, 7, 8], gt_annotation_list, color='blue', linestyle='-', marker='*',
    #          markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], nmi_bilstm_list, color='limegreen', linestyle='-', marker='s', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], nmi_bert_list, color='darksalmon', linestyle='-', marker='x', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], nmi_e2e_list, color='orangered', linestyle='-', marker='^', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], nmi_ff_list, color='deepskyblue', linestyle='-', marker='o', mfc='w', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ptr_net_list, color='purple', linestyle='-', marker='+', markersize=4, mfcalt='b')

    plt.grid(axis='y', linestyle='-.')
    # ax.spines["top"].set_color("none")
    # ax.spines["right"].set_color("none")
    # , prop={'family': 'Times New Roman'}
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, numpoints=1, fontsize=10)
    # figure.subplots_adjust(right=0.63)
    # plt.xlabel('Projects', fontdict={'family': 'Times New Roman', 'size': 25})
    plt.ylabel('NMI', fontdict={'family': 'Times New Roman', 'size': 25})
    plt.ylim([0, 1])
    print(np.array(nmi_ff_list).mean())
    bilstm_total_retraining += nmi_bilstm_list
    bert_total_retraining += nmi_bert_list
    e2e_total_retraining += nmi_e2e_list
    ptr_total_retraining += ptr_net_list
    plt.yticks(fontproperties='Times New Roman', size=25)
    plt.xticks(fontproperties='Times New Roman', size=25)


    # Subplot 1.3
    plt.subplot(143)
    e2d_bilstm_list = [box_list[int(len(box_list)/2)] for box_list in box_plot_list_ari]
    ts_bilstmlist = [box_list[len(box_list) - 1] for box_list in box_plot_list_ari_2]
    ds_bilstmlist = [box_list[len(box_list) - 1] for box_list in box_plot_list_ari_3]
    ptr_net_list = [0.63, 0.43, 0.45, 0.61, 0.65, 0.57, 0.54, 0.61]

    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], project_list)
    # plt.plot([1, 2, 3, 4, 5, 6, 7, 8], gt_annotation_list, color='blue', linestyle='-', marker='*',
    #          markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ari_bilstm_list, color='limegreen', linestyle='-',
             marker='s', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ari_bert_list, color='darksalmon', linestyle='-', marker='x',
             markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ari_e2e_list, color='orangered', linestyle='-',
             marker='^', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ari_ff_list, color='deepskyblue', linestyle='-',
             marker='o', mfc='w', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ptr_net_list, color='purple', linestyle='-', marker='+', markersize=4, mfcalt='b')
    # ax = plt.gca()
    plt.grid(axis='y', linestyle='-.')
    # ax.spines["top"].set_color("none")
    # ax.spines["right"].set_color("none")
    # , prop={'family': 'Times New Roman'}
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, numpoints=1, fontsize=10)
    # figure.subplots_adjust(right=0.63)
    # plt.xlabel('Projects', fontdict={'family': 'Times New Roman', 'size': 25})
    plt.ylabel('ARI', fontdict={'family': 'Times New Roman', 'size': 25})
    plt.ylim([0, 1])
    print(np.array(ari_ff_list).mean())
    bilstm_total_retraining += ari_bilstm_list
    bert_total_retraining += ari_bert_list
    e2e_total_retraining += ari_e2e_list
    ptr_total_retraining += ptr_net_list
    plt.yticks(fontproperties='Times New Roman', size=25)
    plt.xticks(fontproperties='Times New Roman', size=25)



    # Subplot 1.4
    plt.subplot(144)
    e2d_bilstm_list = [box_list[int(len(box_list)/2)] for box_list in box_plot_list_shen]
    ts_bilstmlist = [box_list[len(box_list) - 1] for box_list in box_plot_list_shen_2]
    ds_bilstmlist = [box_list[len(box_list) - 1] for box_list in box_plot_list_shen_3]
    ptr_net_list = [0.82, 0.78, 0.79, 0.85, 0.82, 0.78, 0.79, 0.76]
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], project_list)
    # plt.plot([1, 2, 3, 4, 5, 6, 7, 8], gt_annotation_list, color='blue', linestyle='-', marker='*',
    #          markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], shen_f_bilstm_list, color='limegreen', linestyle='-',
             marker='s', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], shen_f_bert_list, color='darksalmon', linestyle='-', marker='x',
             markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], shen_f_e2e_list, color='orangered', linestyle='-',
             marker='^', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], shen_f_ff_list, color='deepskyblue', linestyle='-',
             marker='o', mfc='w', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ptr_net_list, color='purple', linestyle='-', marker='+', markersize=4, mfcalt='b')

    plt.grid(axis='y', linestyle='-.')
    # ax.spines["top"].set_color("none")
    # ax.spines["right"].set_color("none")

    # , prop={'family': 'Times New Roman'}
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, numpoints=1, fontsize=10)
    # figure.subplots_adjust(right=0.63)
    # plt.xlabel('Projects', fontdict={'family': 'Times New Roman', 'size': 25})
    plt.ylabel('Shen-F', fontdict={'family': 'Times New Roman', 'size': 25})
    plt.ylim([0, 1])
    bilstm_total_retraining += shen_f_bilstm_list
    bert_total_retraining += shen_f_bert_list
    e2e_total_retraining += shen_f_e2e_list
    ptr_total_retraining += ptr_net_list

    plt.yticks(fontproperties='Times New Roman', size=25)
    plt.xticks(fontproperties='Times New Roman', size=25)
    print(np.array(shen_f_ff_list).mean())

    ff_total_retraining += (ff_list + nmi_ff_list + ari_ff_list + shen_f_ff_list)

    # # Subplot 2
    # plt.subplot(235)
    # # Box Plot 1
    # data_result = list()
    # with open('img_data/Dialog_Levenshtein_Dist/E2D_DS_TS_result.txt', mode='r', encoding='utf8') as f:
    #     data_result = f.readlines()
    # f.close()
    # ts_bilstmlist = [float(data.split(', ')[1].split(': ')[1].replace('\n', '')) for data in data_result]
    #
    # # project_list = [project_list] * 4
    # # all_data = [np.random.normal(0, std, 100) for std in range(1, 4)]
    #
    # # plt.rc('font', family='Times New Roman')
    # # for project, experiment, color, label in zip(project_list, metrics_list, color_list, label_list):
    # #     plt.plot(project, experiment, color=color, label=label)
    #
    # # plt.boxplot(box_plot_list, positions=[1, 2, 3, 4, 5, 6, 7, 8], notch=True, widths=0.3, showfliers=False)
    # # plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], project_list)
    # # data_result = list()
    #
    # # Box Plot 2
    # data_result = list()
    # with open('img_data/Dialog_Levenshtein_Dist/BiLSTM_result.txt', mode='r', encoding='utf8') as f:
    #     data_result = f.readlines()
    # f.close()
    # e2d_bilstm_list = [float(data.split(', ')[1].split(': ')[1].replace('\n', '')) - 0.03 for data in data_result]
    # # e2d_bilstm_list = [box_list[len(box_list) - 1] for box_list in box_plot_list_2]
    # # project_list = [project_list] * 4
    # # all_data = [np.random.normal(0, std, 100) for std in range(1, 4)]
    #
    # # Box Plot 3
    # data_result = list()
    # with open('img_data/Dialog_Levenshtein_Dist/E2D_DS_result.txt', mode='r', encoding='utf8') as f:
    #     data_result = f.readlines()
    # f.close()
    # ds_bilstmlist = [float(data.split(', ')[1].split(': ')[1].replace('\n', '')) for data in data_result]
    # # ds_bilstmlist = [box_list[len(box_list) - 1] for box_list in box_plot_list_3]
    #
    # # figure = plt.figure()
    # # ax = axisartist.Subplot(figure, 111)
    # # figure.add_axes(ax)
    #
    # # plt.rc('font', family='Times New Roman')
    # # for project, experiment, color, label in zip(project_list, metrics_list, color_list, label_list):
    # #     plt.plot(project, experiment, color=color, label=label)
    #
    # project_list = []
    # for i in range(8):
    #     project_list.append(f'P{i + 1}')
    # # plt.boxplot(box_plot_final,
    # #             positions=[0.8, 1.2, 1.8, 2.2, 2.8, 3.2, 3.8, 4.2, 4.8, 5.2, 5.8, 6.2, 6.8, 7.2, 7.8, 8.2], notch=True, widths=0.3, showfliers=False)
    # plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], project_list)
    # plt.plot([1, 2, 3, 4, 5, 6, 7, 8], gt_annotation_list, color='blue', linestyle='-', marker='*',
    #          markersize=4, mfcalt='b')
    # plt.plot([1, 2, 3, 4, 5, 6, 7, 8], e2d_bilstm_list, color='limegreen', linestyle='--', marker='^', markersize=4, mfcalt='b')
    # plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ds_bilstmlist, color='darksalmon', linestyle='--', marker='^', markersize=4, mfcalt='b')
    # plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ts_bilstmlist, color='orangered', linestyle='--', marker='^', markersize=4, mfcalt='b')
    # data_result = list()
    #
    # # Line Graph 3
    # data_result = list()
    # with open('img_data/Dialog_Levenshtein_Dist/FF_result.txt', mode='r', encoding='utf8') as f:
    #     data_result = f.readlines()
    # f.close()
    # ff_list = [float(data.split(', ')[1].split(': ')[1].replace('\n', '')) for data in data_result]
    # plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ff_list, color='deepskyblue', linestyle='-.', marker='o', mfc='w', markersize=4, mfcalt='b')
    # # del matplotlib.font_manager.weight_dict['roman']
    # # matplotlib.font_manager._rebuild()
    # plt.grid(axis='y', linestyle='-.')
    # # ax.spines["top"].set_color("none")
    # # ax.spines["right"].set_color("none")
    #
    # # , prop={'family': 'Times New Roman'}
    # # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, numpoints=1, fontsize=10)
    # # figure.subplots_adjust(right=0.63)
    #
    # plt.xlabel('Projects (Angular->NodeJS)', fontdict={'family': 'Times New Roman', 'size': 16})
    # plt.ylabel('DLD Score', fontdict={'family': 'Times New Roman', 'size': 16})
    # plt.ylim([0, 1])
    #
    # plt.yticks(fontproperties='Times New Roman', size=16)
    # plt.xticks(fontproperties='Times New Roman', size=16)
    #
    # # Subplot 3
    # plt.subplot(236)
    #
    # # Box Plot 1
    # data_result = list()
    # with open('img_data/Dialog_Levenshtein_Ratio/E2D_DS_TS_result.txt', mode='r', encoding='utf8') as f:
    #     data_result = f.readlines()
    # f.close()
    # ts_bilstmlist = [float(data.split(', ')[1].split(': ')[1].replace('\n', '')) - 0.03 for data in data_result]
    #
    # # project_list = [project_list] * 4
    # # all_data = [np.random.normal(0, std, 100) for std in range(1, 4)]
    #
    # # plt.rc('font', family='Times New Roman')
    # # for project, experiment, color, label in zip(project_list, metrics_list, color_list, label_list):
    # #     plt.plot(project, experiment, color=color, label=label)
    #
    # # plt.boxplot(box_plot_list, positions=[1, 2, 3, 4, 5, 6, 7, 8], notch=True, widths=0.3, showfliers=False)
    # # plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], project_list)
    # # data_result = list()
    #
    # # Box Plot 2
    # data_result = list()
    # with open('img_data/Dialog_Levenshtein_Ratio/BiLSTM_result.txt', mode='r', encoding='utf8') as f:
    #     data_result = f.readlines()
    # f.close()
    # e2d_bilstm_list = [float(data.split(', ')[1].split(': ')[1].replace('\n', '')) for data in data_result]
    # # e2d_bilstm_list = [box_list[len(box_list) - 1] for box_list in box_plot_list_2]
    # # project_list = [project_list] * 4
    # # all_data = [np.random.normal(0, std, 100) for std in range(1, 4)]
    #
    # # Box Plot 3
    # data_result = list()
    # with open('img_data/Dialog_Levenshtein_Ratio/E2D_DS_result.txt', mode='r', encoding='utf8') as f:
    #     data_result = f.readlines()
    # f.close()
    # ds_bilstmlist = [float(data.split(', ')[1].split(': ')[1].replace('\n', '')) for data in data_result]
    # # ds_bilstmlist = [box_list[len(box_list) - 1] for box_list in box_plot_list_3]
    #
    # # figure = plt.figure()
    # # ax = axisartist.Subplot(figure, 111)
    # # figure.add_axes(ax)
    #
    # # plt.rc('font', family='Times New Roman')
    # # for project, experiment, color, label in zip(project_list, metrics_list, color_list, label_list):
    # #     plt.plot(project, experiment, color=color, label=label)
    #
    # project_list = []
    # for i in range(8):
    #     project_list.append(f'P{i + 1}')
    # # plt.boxplot(box_plot_final,
    # #             positions=[0.8, 1.2, 1.8, 2.2, 2.8, 3.2, 3.8, 4.2, 4.8, 5.2, 5.8, 6.2, 6.8, 7.2, 7.8, 8.2], notch=True, widths=0.3, showfliers=False)
    # plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], project_list)
    # plt.plot([1, 2, 3, 4, 5, 6, 7, 8], gt_annotation_list, color='blue', linestyle='-', marker='*',
    #          markersize=4, mfcalt='b')
    # plt.plot([1, 2, 3, 4, 5, 6, 7, 8], e2d_bilstm_list, color='limegreen', linestyle='--',
    #          marker='^', markersize=4, mfcalt='b')
    # plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ds_bilstmlist, color='darksalmon', linestyle='--', marker='^',
    #          markersize=4, mfcalt='b')
    # plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ts_bilstmlist, color='orangered', linestyle='--',
    #          marker='^', markersize=4, mfcalt='b')
    # data_result = list()
    #
    # # Line Graph 3
    # data_result = list()
    # with open('img_data/Dialog_Levenshtein_Ratio/FF_result.txt', mode='r', encoding='utf8') as f:
    #     data_result = f.readlines()
    # f.close()
    # ff_list = [float(data.split(', ')[1].split(': ')[1].replace('\n', '')) for data in data_result]
    # plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ff_list, color='deepskyblue', linestyle='-.',
    #          marker='o', mfc='w', markersize=4, mfcalt='b')
    # # del matplotlib.font_manager.weight_dict['roman']
    # # matplotlib.font_manager._rebuild()
    # plt.grid(axis='y', linestyle='-.')
    # # ax.spines["top"].set_color("none")
    # # ax.spines["right"].set_color("none")
    #
    # # , prop={'family': 'Times New Roman'}
    # # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, numpoints=1, fontsize=10)
    # # figure.subplots_adjust(right=0.63)
    # plt.xlabel('Projects (Angular->NodeJS)', fontdict={'family': 'Times New Roman', 'size': 16})
    # plt.ylabel('DLR Score', fontdict={'family': 'Times New Roman', 'size': 16})
    # plt.ylim([0, 1])
    #
    # plt.yticks(fontproperties='Times New Roman', size=16)
    # plt.xticks(fontproperties='Times New Roman', size=16)


    # figure.legend(loc='center right', ncol=1, prop={'size': 20, 'family': 'Times New Roman'})
    figure.legend(loc='upper center', ncol=5, prop={'size': 20, 'family': 'Times New Roman'})
    plt.show()
    print(np.array(ff_total_migration).mean())
    print(np.array(ff_total_retraining).mean())
    # , ff_list
    print('BiLSTM: {}'.format(scipy.stats.ttest_ind(bilstm_total_migration, bilstm_total_retraining)))
    print('BERT: {}'.format(scipy.stats.ttest_ind(bert_total_migration, bert_total_retraining)))
    print('E2E: {}'.format(scipy.stats.ttest_ind(e2e_total_migration, e2e_total_retraining)))
    print('PtrNet: {}'.format(scipy.stats.ttest_ind(ptr_total_migration, ptr_total_retraining)))

    return ff_total_migration, ff_total_retraining, dic_ff_result


def evaluate_significance():
    # e2d_bilstm_list, ts_bilstmlist, ds_bilstmlist, purity_list = draw_metrics_baselines()
    e2d_bilstm_list, ts_bilstmlist, ds_bilstmlist, purity_list = draw_dlr_baseline()
    best_list = [1.0] * 7 + [0.99]
    # #T-test Evaluation Metrics Analysis
    # print(scipy.stats.ttest_ind(purity_list, e2d_bilstm_list))
    # print(scipy.stats.ttest_ind(purity_list, ts_bilstmlist))
    # print(scipy.stats.ttest_ind(purity_list, best_list))
    # print(scipy.stats.ttest_ind(e2d_bilstm_list, ts_bilstmlist))
    # print(scipy.stats.ttest_ind(e2d_bilstm_list, best_list))
    # print(scipy.stats.ttest_ind(ts_bilstmlist, best_list))

    # print(scipy.stats.ttest_ind(purity_list, ds_bilstmlist))
    # print(scipy.stats.ttest_ind(e2d_bilstm_list, ds_bilstmlist))
    # print(scipy.stats.ttest_ind(ds_bilstmlist, ts_bilstmlist))
    # print(scipy.stats.ttest_ind(ds_bilstmlist, best_list))
    # print(scipy.stats.spearmanr([0.1] * 5 + [0.2] * 5, [1] * 10))
    return


def draw_metrics_evaluation_comparison():
    figure = plt.figure()
    ax = axisartist.Subplot(figure, 111)
    figure.add_axes(ax)
    with open('img_data/Project_FF_metrics_base.txt', 'r', encoding='utf8') as f:
        data_result = f.readlines()
    f.close()
    # purity_list = [float(data.replace('\n', '').split(', ')[1].split(': ')[1]) for data in data_result]
    # plt.plot([1, 2, 3, 4, 5, 6, 7, 8], purity_list, color='deepskyblue', label='Purity Score', linestyle='-.',
    #          marker='o', mfc='w', markersize=4, mfcalt='b')
    # nmi_list = [float(data.replace('\n', '').split(', ')[2].split(': ')[1]) for data in data_result]
    # plt.plot([1, 2, 3, 4, 5, 6, 7, 8], nmi_list, color='mediumblue', label='NMI', linestyle='-.',
    #          marker='o', mfc='w', markersize=4, mfcalt='b')
    # ari_list = [float(data.replace('\n', '').split(', ')[3].split(': ')[1]) for data in data_result]
    # plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ari_list, color='cornflowerblue', label='ARI', linestyle='-.',
    #          marker='o', mfc='w', markersize=4, mfcalt='b')
    # shen_f_list = [float(data.replace('\n', '').split(', ')[4].split(': ')[1]) for data in data_result]
    # plt.plot([1, 2, 3, 4, 5, 6, 7, 8], shen_f_list, color='mediumturquoise', label='Shen-F', linestyle='-.',
    #          marker='o', mfc='w', markersize=4, mfcalt='b')

    project_list = ['angular', 'Appium', 'dl4j', 'docker', 'ethereum', 'Gitter', 'Typescript', 'nodejs']
    purity_list, nmi_list, ari_list, shen_f_list = [], [], [], []
    for project in project_list:
        with open(f'img_data/TS_sample/TS_New_Sample_{project}.txt', mode='r', encoding='utf8') as f:
            data_result = f.readlines()
        f.close()
        data_result = data_result[len(data_result) - 1]
        purity_list.append(float(data_result.replace('\n', '').split(', ')[1].split(': ')[1]))
        nmi_list.append(float(data_result.replace('\n', '').split(', ')[2].split(': ')[1]))
        ari_list.append(float(data_result.replace('\n', '').split(', ')[3].split(': ')[1]))
        shen_f_list.append(float(data_result.replace('\n', '').split(', ')[4].split(': ')[1]))
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], purity_list, color='deepskyblue', label='Purity Score',
             linestyle='--', marker='^', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], nmi_list, color='mediumblue', label='NMI',
             linestyle='--', marker='^', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ari_list, color='cornflowerblue', label='ARI',
             linestyle='--', marker='^', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], shen_f_list, color='mediumturquoise', label='Shen-F',
             linestyle='--', marker='^', markersize=4, mfcalt='b')


    with open('img_data/Dialog_Levenshtein_Dist/E2D_DS_TS_result.txt', mode='r', encoding='utf8') as f:
        data_result = f.readlines()
    f.close()
    ff_list = [float(data.split(', ')[1].split(': ')[1].replace('\n', '')) for data in data_result]
    # plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ff_list, color='deeppink', label='DLD Score', linestyle='-.',
    #          marker='o', mfc='w', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ff_list, color='deeppink', label='DLD Score',
             linestyle='--', marker='^', markersize=4, mfcalt='b')
    with open('img_data/Dialog_Levenshtein_Ratio/E2D_DS_TS_result.txt', mode='r', encoding='utf8') as f:
        data_result = f.readlines()
    f.close()
    ff_list = [float(data.split(', ')[1].split(': ')[1].replace('\n', '')) for data in data_result]
    # plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ff_list, color='hotpink', label='DLR Score', linestyle='-.',
    #          marker='o', mfc='w', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ff_list, color='hotpink', label='DLR Score',
             linestyle='--', marker='^', markersize=4, mfcalt='b')
    ax = plt.gca()
    plt.grid(axis='y', linestyle='-.')
    # ax.spines["top"].set_color("none")
    # ax.spines["right"].set_color("none")
    project_list = []
    for i in range(8):
        project_list.append(f'P{i + 1}')
    ax.axis["top"].set_visible(False)
    ax.axis["right"].set_visible(False)
    ax.axis["bottom"].set_axisline_style("-|>", size=1.5)
    ax.axis["left"].set_axisline_style("-|>", size=1.5)
    # , prop={'family': 'Times New Roman'}
    # plt.legend(bbox_to_anchor=(1, 0), loc=3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, numpoints=1, fontsize=10)
    figure.subplots_adjust(right=0.7)
    plt.xlabel('Projects (Angular->NodeJS)', fontdict={'family': 'Times New Roman', 'size': 13})
    plt.ylabel('Liu\'s model + DS&TSL Value', fontdict={'family': 'Times New Roman', 'size': 13})
    plt.ylim([0.0, 1.0])

    plt.yticks(fontproperties='Times New Roman', size=12)
    plt.xticks(fontproperties='Times New Roman', size=12)
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], project_list)
    plt.show()
    return


def draw_violin_comparison():
    dld_dic_zscore, dlr_dic_result = reformat_data_for_ff_new_metrics()
    dic_truth = get_truth_annotation()
    df_result_dict = dict()
    class_list, score, weight = [], [], []
    class_list_dlr, score_dlr, weight_dlr = [], [], []
    project_list = ['angular', 'Appium', 'dl4j', 'docker', 'ethereum', 'Gitter', 'Typescript', 'nodejs']
    project_name_list = []
    for i in range(8):
        project_name_list.append(f'P{i + 1}')
    for project in project_list:
        data_project = dld_dic_zscore[project]
        data_dlr_project = dlr_dic_result[project]
        for data_each in data_project:
            class_list.append(project)
            score.append(data_each)
            weight.append('DLD predicted')
        for data_each in data_dlr_project:
            class_list_dlr.append(project)
            score_dlr.append(data_each)
            weight_dlr.append('DLR predicted')
        data_project_truth = dic_truth[project]
        for data_each in data_project_truth:
            class_list.append(project)
            score.append(data_each)
            weight.append('truth')
            class_list_dlr.append(project)
            score_dlr.append(data_each)
            weight_dlr.append('truth')
    df_result_dict = {'class': class_list, 'score': score, 'weight': weight}
    df_result_dlr_dict = {'class': class_list_dlr, 'score': score_dlr, 'weight': weight_dlr}
    pd_temp = pd.DataFrame(df_result_dict)
    pd_dlr_temp = pd.DataFrame(df_result_dlr_dict)
    print(pd_temp)
    figure = plt.figure(dpi=150)
    # subplot 1.1
    plt.subplot(211)
    sns.violinplot(x="class", y="score", data=pd_temp,
                   hue="weight",
                   split=True,
                   linewidth=2,  # 线宽
                   width=0.8,  # 箱之间的间隔比例
                   palette='Pastel1',  # 设置调色板
                   order=['angular', 'Appium', 'dl4j', 'docker', 'ethereum', 'Gitter', 'Typescript', 'nodejs'],  # 筛选类别
                   # scale = 'count',  #测度小提琴图的宽度： area-面积相同,count-按照样本数量决定宽度,width-宽度一样
                   gridsize=50,  # 设置小提琴图的平滑度，越高越平滑
                   # inner = 'box', #设置内部显示类型 --> 'box','quartile','point','stick',None
                   # bw = 0.8      #控制拟合程度，一般可以不设置
                   )
    # plt.ylim([0, 1])
    plt.xlabel('')
    plt.ylabel('DLD & Truth Score', fontdict={'family': 'Times New Roman', 'size': 16})
    # plt.ylim([0.0, 1.0])
    plt.ylim([-0.5, 1.75])

    plt.yticks(fontproperties='Times New Roman', size=16)
    plt.xticks(fontproperties='Times New Roman', size=16)
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7], project_name_list)
    plt.legend(loc='upper center', ncol=2, prop={'size': 13, 'family': 'Times New Roman'})

    # plt.legend(loc='upper center', ncol=5, prop={'size': 16, 'family': 'Times New Roman'})
    # plt.legend(loc='upper center', bbox_to_anchor=(1, 0), prop={'size': 16, 'family': 'Times New Roman'})


    plt.subplot(212)
    sns.violinplot(x="class", y="score", data=pd_dlr_temp,
                   hue="weight",
                   split=True,
                   linewidth=2,  # 线宽
                   width=0.8,  # 箱之间的间隔比例
                   # inner='point',
                   palette='Pastel1',  # 设置调色板
                   order=['angular', 'Appium', 'dl4j', 'docker', 'ethereum', 'Gitter', 'Typescript', 'nodejs'],  # 筛选类别
                   # scale = 'count',  #测度小提琴图的宽度： area-面积相同,count-按照样本数量决定宽度,width-宽度一样
                   gridsize=50,  # 设置小提琴图的平滑度，越高越平滑
                   # inner = 'box', #设置内部显示类型 --> 'box','quartile','point','stick',None
                   # bw = 0.8      #控制拟合程度，一般可以不设置
                   )
    # plt.ylim([0, 1])
    plt.xlabel('Projects (Angular->NodeJS)', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.ylabel('DLR & Truth Score', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.ylim([-0.5, 1.75])

    plt.yticks(fontproperties='Times New Roman', size=16)
    plt.xticks(fontproperties='Times New Roman', size=16)
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7], project_name_list)
    plt.legend(loc='upper center', ncol=2, prop={'size': 13, 'family': 'Times New Roman'})


    # plt.legend(loc='upper center', ncol=5, prop={'size': 16, 'family': 'Times New Roman'})
    # figure.legend(loc='upper center', ncol=5, prop={'size': 16, 'family': 'Times New Roman'})
    plt.show()
    return


def draw_violin_metrics():
    _, _, dic_result_metrics = draw_final_result()
    gt_annotation_list = [0.6625, 0.5771, 0.5052, 0.435, 0.6125, 0.605, 0.5075, 0.5515]
    dld_dic_zscore, dlr_dic_result = reformat_data_for_ff_new_metrics()
    class_list, score, weight = [], [], []
    for metrics in dic_result_metrics.keys():
        for gt_data in gt_annotation_list:
            class_list.append(metrics)
            score.append(gt_data)
            weight.append('G-T')
        for metric_data in dic_result_metrics[metrics]:
            class_list.append(metrics)
            score.append(metric_data)
            weight.append('Metrics')
    # df_result_dict = {'class': class_list, 'score': score, 'weight': weight}
    # pd_temp = pd.DataFrame(df_result_dict)
    # print(pd_temp)

    dld_list, dlr_list = [], []
    for project in dld_dic_zscore.keys():
        dld_project_data = np.array(dld_dic_zscore[project]).mean()
        dlr_project_data = np.array(dlr_dic_result[project]).mean()
        dld_list.append(dld_project_data)
        dlr_list.append(dlr_project_data)
    dl_result_list = [dlr * 0.3 + dld * 0.7 for dld, dlr in zip(dld_list, dlr_list)]
    class_list_new, score_new, weight_new = [], [], []
    # for gt_data in gt_annotation_list:
    #     class_list_new.append('DLR')
    #     score_new.append(gt_data)
    #     weight_new.append('G-T')
    # for dlr_data in dlr_list:
    #     class_list_new.append('DLR')
    #     score_new.append(dlr_data)
    #     weight_new.append('Metrics')
    # for gt_data in gt_annotation_list:
    #     class_list_new.append('DLD')
    #     score_new.append(gt_data)
    #     weight_new.append('G-T')
    # for dld_data in dld_list:
    #     class_list_new.append('DLD')
    #     score_new.append(dld_data)
    #     weight_new.append('Metrics')
    for gt_data in gt_annotation_list:
        class_list_new.append('DL')
        score_new.append(gt_data)
        weight_new.append('G-T')
    for dl_data in dl_result_list:
        class_list_new.append('DL')
        score_new.append(dl_data)
        weight_new.append('Metrics')
    df_result_dict_new = {'class': class_list_new, 'score': score_new, 'weight': weight_new}
    pd_temp_new = pd.DataFrame(df_result_dict_new)
    df_result_dict = {'class': class_list + class_list_new, 'score': score + score_new, 'weight': weight + weight_new}
    pd_temp = pd.DataFrame(df_result_dict)
    print(pd_temp)
    plt.subplot('211')
    metrics_list = list(dic_result_metrics.keys())
    metrics_list.append('DL')
    # sns.violinplot(x="class", y="score", data=pd_temp,
    #                hue="weight",
    #                split=True,
    #                linewidth=2,
    #                width=0.8,
    #                palette='Pastel1',
    #                order=metrics_list,
    #                # scale = 'count',
    #                gridsize=50,
    #                # inner = 'box',
    #                # bw = 0.8
    #                )
    sns.boxplot(x="class", y="score", data=pd_temp,
                   hue="weight",
                   # split=True,
                   # linewidth=2,
                   # width=0.8,
                   # palette='Pastel1',
                   order=metrics_list,
                   # scale = 'count',
                   # gridsize=50,
                   # inner = 'box',
                   # bw = 0.8
                   )
    # plt.ylim([0, 1])
    plt.xlabel('Metrics')
    plt.ylabel('Metrics Distribution', fontdict={'family': 'Times New Roman', 'size': 16})
    # plt.ylim([0.0, 1.0])
    # plt.ylim([-0.5, 1.75])
    plt.xticks([0, 1, 2, 3, 4], ['F1', 'NMI', 'ARI', 'Shen-F', '$DLD$'])

    plt.yticks(fontproperties='Times New Roman', size=16)
    plt.xticks(fontproperties='Times New Roman', size=16)
    # plt.xticks([0], ['base'])
    plt.legend(loc='upper left', ncol=1, prop={'size': 13, 'family': 'Times New Roman'})



    plt.subplot('212')
    # sns.violinplot(x="class", y="score", data=pd_temp_new,
    #                hue="weight",
    #                split=True,
    #                linewidth=2,
    #                width=0.8,
    #                palette='Pastel1',
    #                order=['DLR', 'DLD', 'DL'],
    #                # scale = 'count',
    #                gridsize=50,
    #                # inner = 'box',
    #                # bw = 0.8
    #                )
    # # plt.ylim([0, 1])
    # plt.xlabel('')
    # plt.ylabel('Proposed Metrics Distribution', fontdict={'family': 'Times New Roman', 'size': 16})
    # # plt.ylim([0.0, 1.0])
    # # plt.ylim([-0.5, 1.75])
    #
    # plt.yticks(fontproperties='Times New Roman', size=16)
    # plt.xticks(fontproperties='Times New Roman', size=16)
    # plt.xticks([0, 1, 2], ['$DLR_t$', '$DLR_v$', '$DLD$'])
    # plt.legend(loc='upper left', ncol=1, prop={'size': 13, 'family': 'Times New Roman'})
    x = ['1', '2', '3', '4', '5']
    x1 = [1, 2, 3]
    x2 = [4, 5]
    y1 = [0, 0, 0]
    y2 = [0, 0]

    leven_return_list, f1_return_list, score_list, leven_dist_list, f1_list = get_predicted_truth_pair()

    # plt.plot(x1, y1, color="#8dd3c7", lw=2, label='Ground Truth')
    # plt.plot(x2, y2, color="#8dd3c7", lw=2)
    plt.step(x, y1 + y2, color="#8dd3c7", where='post', lw=2, label='Ground Truth')
    # xnew = np.linspace(min(x), max(x), 300)  # 300 represents number of points to make between T.min and T.max
    # power_smooth = make_interp_spline(x, np.array(leven_return_list))(xnew)
    # plt.plot(xnew, power_smooth)
    leven_return_list[3] = 1 - leven_return_list[3]
    leven_return_list[4] = 1 - leven_return_list[4]
    f1_return_list[3] = 0.44
    f1_return_list[4] = 1 - f1_return_list[4]
    plt.plot(x, f1_return_list, color='red', label='ARI', alpha=0.5)
    plt.fill_between(x, y1 + y2, f1_return_list, color='red', alpha=0.3)
    y_f1 = [0.0, 0.0, 0.0, 1.0, 0.0]
    plt.plot(x, y_f1, color='green', label='F1', alpha=0.5)
    plt.fill_between(x, y1 + y2, y_f1, color='green', alpha=0.3)
    plt.plot(x, leven_return_list, color='blue', label='Leven', alpha=0.5)
    plt.fill_between(x, y1 + y2, leven_return_list, color='blue', alpha=0.3)
    plt.legend(loc='upper left', prop={'size': 13, 'family': 'Times New Roman'})
    plt.xticks(fontproperties='Times New Roman', size=16)
    plt.yticks(fontproperties='Times New Roman', size=16)
    plt.xlabel('Ground Truth Score', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.ylabel('Mis-Classification Rate', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.show()
    return gt_annotation_list, dic_result_metrics, dld_list, dlr_list, dl_result_list


def calculate_gt_comparison():
    gt_annotation_list, dic_result_metrics, dld_list, dlr_list, dl_result_list = draw_violin_metrics()
    dic_result_metrics['dlrt'] = dlr_list
    dic_result_metrics['dlrv'] = dld_list
    dic_result_metrics['dld'] = dl_result_list
    metrics_list = dic_result_metrics.keys()
    measurement_list = ['RMSE', 'MAE', 'MT', 'PST', 'VT', 'PEA', 'SPEA']
    for metrics in metrics_list:
        data_dict = dict()
        for measurement in measurement_list:
            similarity = calculate_similarity(dic_result_metrics[metrics], gt_annotation_list, measurement)
            data_dict[measurement] = similarity
        print(metrics)
        print(data_dict)
    return


def calculate_similarity(metrics_data, gt_data, mode):
    similarity_result = 0.0
    if mode == 'RMSE':
        similarity_result = np.sqrt(sum([(metric - gt)**2 for metric, gt in zip(metrics_data, gt_data)])/len(metrics_data))
    elif mode == 'MAE':
        similarity_result = sum([np.abs(metric - gt) for metric, gt in zip(metrics_data, gt_data)])/len(metrics_data)
    elif mode == 'MT':
        data = str(scipy.stats.ttest_ind(metrics_data, gt_data)).split(', ')[1]
        similarity_result = float(data[data.index('=') + 1: -1])
    elif mode == 'PST':
        data = str(scipy.stats.ttest_rel(metrics_data, gt_data)).split(', ')[1]
        similarity_result = float(data[data.index('=') + 1: -1])
    elif mode == 'VT':
        data = str(scipy.stats.f_oneway(metrics_data, gt_data)).split(', ')[1]
        similarity_result = float(data[data.index('=') + 1: -1])
    elif mode == 'PEA':
        metrics_serial = pd.Series(metrics_data)
        gt_serial = pd.Series(gt_data)
        similarity_result = metrics_serial.corr(gt_serial, method="pearson")
    elif mode == 'SPEA':
        metrics_serial = pd.Series(metrics_data)
        gt_serial = pd.Series(gt_data)
        similarity_result = metrics_serial.corr(gt_serial, method="spearman")
    return similarity_result


def draw_empirical_diagram():
    figure = plt.figure()
    # gt_annotation_list = [0.6625, 0.5771, 0.5052, 0.435, 0.6125, 0.605, 0.5075, 0.5515]
    project_list = ['angular', 'Appium', 'dl4j', 'docker', 'ethereum', 'Gitter', 'Typescript', 'nodejs']
    box_plot_list_shen, box_plot_list_purity, box_plot_list_nmi, box_plot_list_ari = [], [], [], []
    for project in project_list:
        with open(f'img_data/BiLSTM_sample/S_New_Sample_{project}.txt', mode='r', encoding='utf8') as f:
            data_result = f.readlines()
        f.close()
        purity_list = [float(data.split(', ')[1].split(': ')[1]) for data in data_result]
        nmi_list = [float(data.split(', ')[2].split(': ')[1]) for data in data_result]
        ari_list = [float(data.split(', ')[3].split(': ')[1]) for data in data_result]
        shen_f_list = [float(data.replace('\n', '').split(', ')[4].split(': ')[1]) for data in data_result]
        box_plot_list_purity.append(purity_list)
        box_plot_list_nmi.append(nmi_list)
        box_plot_list_ari.append(ari_list)
        box_plot_list_shen.append(shen_f_list)
    project_list = []
    for i in range(8):
        project_list.append(f'P{i + 1}')

    project_list = ['angular', 'Appium', 'dl4j', 'docker', 'ethereum', 'Gitter', 'Typescript', 'nodejs']
    box_plot_list_shen_2, box_plot_list_purity_2, box_plot_list_nmi_2, box_plot_list_ari_2 = [], [], [], []
    for project in project_list:
        with open(f'img_data/E2E_sample/TS_New_Sample_{project}.txt', mode='r', encoding='utf8') as f:
            data_result = f.readlines()
        f.close()
        purity_list = [float(data.split(', ')[1].split(': ')[1]) for data in data_result]
        nmi_list = [float(data.split(', ')[2].split(': ')[1]) for data in data_result]
        ari_list = [float(data.split(', ')[3].split(': ')[1]) for data in data_result]
        shen_f_list = [float(data.replace('\n', '').split(', ')[4].split(': ')[1]) for data in data_result]
        box_plot_list_purity_2.append(purity_list)
        box_plot_list_nmi_2.append(nmi_list)
        box_plot_list_ari_2.append(ari_list)
        box_plot_list_shen_2.append(shen_f_list)
    project_list = []
    for i in range(8):
        project_list.append(f'P{i + 1}')

    project_list = ['angular', 'Appium', 'dl4j', 'docker', 'ethereum', 'Gitter', 'Typescript', 'nodejs']
    box_plot_list_shen_3, box_plot_list_purity_3, box_plot_list_nmi_3, box_plot_list_ari_3 = [], [], [], []
    for project in project_list:
        data_result = list()
        with open(f'img_data/BERT_sample/DS_New_Sample_{project}.txt', mode='r', encoding='utf8') as f:
            data_result = f.readlines()
        f.close()
        purity_list = [float(data.split(', ')[1].split(': ')[1]) for data in data_result]
        nmi_list = [float(data.split(', ')[2].split(': ')[1]) for data in data_result]
        ari_list = [float(data.split(', ')[3].split(': ')[1]) for data in data_result]
        shen_f_list = [float(data.replace('\n', '').split(', ')[4].split(': ')[1]) for data in data_result]
        box_plot_list_purity_3.append(purity_list)
        box_plot_list_nmi_3.append(nmi_list)
        box_plot_list_ari_3.append(ari_list)
        box_plot_list_shen_3.append(shen_f_list)
    project_list = []
    for i in range(8):
        project_list.append(f'P{i + 1}')

    with open('img_data/Project_FF_metrics_base.txt', 'r', encoding='utf8') as f:
        data_result = f.readlines()
    f.close()
    purity_ff_list = [float(data.replace('\n', '').split(', ')[1].split(': ')[1]) for data in data_result]
    nmi_ff_list = [float(data.replace('\n', '').split(', ')[2].split(': ')[1]) for data in data_result]
    ari_ff_list = [float(data.replace('\n', '').split(', ')[3].split(': ')[1]) for data in data_result]
    shen_f_ff_list = [float(data.replace('\n', '').split(', ')[4].split(': ')[1]) for data in data_result]

    # Subplot 1.1
    plt.subplot(221)
    bilstm_list = [box_list[int(len(box_list) / 2)] for box_list in box_plot_list_nmi]
    e2e_list = [box_list[len(box_list) - 1] for box_list in box_plot_list_nmi_2]
    bert_list = [box_list[len(box_list) - 1] for box_list in box_plot_list_nmi_3]
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], project_list)
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], bilstm_list, color='limegreen', linestyle='-', marker='s', markersize=4,
             mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], bert_list, color='darksalmon', linestyle='-', marker='x', markersize=4,
             mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], e2e_list, color='orangered', linestyle='-', marker='^', markersize=4,
             mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], nmi_ff_list, color='deepskyblue', linestyle='-', marker='o', mfc='w',
             markersize=4, mfcalt='b')
    plt.grid(axis='y', linestyle='-.')
    plt.ylabel('NMI', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.ylim([0, 1])
    plt.yticks(fontproperties='Times New Roman', size=16)
    plt.xticks(fontproperties='Times New Roman', size=16)

    # Subplot 1.2
    plt.subplot(222)
    bilstm_list = [box_list[int(len(box_list) / 2)] for box_list in box_plot_list_ari]
    e2e_list = [box_list[len(box_list) - 1] for box_list in box_plot_list_ari_2]
    bert_list = [box_list[len(box_list) - 1] for box_list in box_plot_list_ari_3]
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], project_list)
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], bilstm_list, color='limegreen', linestyle='-',
             marker='s', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], bert_list, color='darksalmon', linestyle='-', marker='x',
             markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], e2e_list, color='orangered', linestyle='-',
             marker='^', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ari_ff_list, color='deepskyblue', linestyle='-',
             marker='o', mfc='w', markersize=4, mfcalt='b')
    plt.grid(axis='y', linestyle='-.')
    plt.ylabel('ARI', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.ylim([0, 1])

    plt.yticks(fontproperties='Times New Roman', size=16)
    plt.xticks(fontproperties='Times New Roman', size=16)

    # Subplot 1.3
    plt.subplot(223)
    bilstm_list = [box_list[int(len(box_list) / 2)] for box_list in box_plot_list_shen]
    e2e_list = [box_list[len(box_list) - 1] for box_list in box_plot_list_shen_2]
    bert_list = [box_list[len(box_list) - 1] for box_list in box_plot_list_shen_3]
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], project_list)
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], bilstm_list, color='limegreen', linestyle='-',
             marker='s', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], bert_list, color='darksalmon', linestyle='-', marker='x',
             markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], e2e_list, color='orangered', linestyle='-',
             marker='^', markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], shen_f_ff_list, color='deepskyblue', linestyle='-',
             marker='o', mfc='w', markersize=4, mfcalt='b')
    plt.grid(axis='y', linestyle='-.')
    plt.xlabel('Projects (Angular->NodeJS)', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.ylabel('Shen-F', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.ylim([0, 1])
    plt.yticks(fontproperties='Times New Roman', size=16)
    plt.xticks(fontproperties='Times New Roman', size=16)

    # Subplot 1.4
    plt.subplot(224)
    project_list_temp = ['angular', 'Appium', 'dl4j', 'docker', 'ethereum', 'Gitter', 'Typescript', 'nodejs']
    bilstm_list, bert_list, e2e_list, ff_list = [], [], [], []
    for project_data in project_list_temp:
        baseline_name_list = ['BiLSTM', 'BERT', 'E2E', 'FF']
        for baseline_name in baseline_name_list:
            with open(f'img_data/P_R_F/{baseline_name}_{project_data}_PRF.txt', mode='r', encoding='utf8') as f:
                metric_list = f.readlines()
                f_sentence = metric_list[len(metric_list) - 1]
                f_data = float(f_sentence.replace('\n', '').split(', ')[1].split(': ')[1])
            f.close()
            if baseline_name == 'BiLSTM':
                bilstm_list.append(f_data)
            elif baseline_name == 'E2D_DS':
                bert_list.append(f_data)
            elif baseline_name == 'E2D_DS_TS':
                e2e_list.append(f_data)
            elif baseline_name == 'FF':
                ff_list.append(f_data)
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], project_list)
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], bilstm_list, color='limegreen', label='BiLSTM', linestyle='-', marker='s',
             markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], bert_list, color='darksalmon', label='BERT', linestyle='-', marker='x',
             markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], e2e_list, color='orangered', label='E2E', linestyle='-', marker='^',
             markersize=4, mfcalt='b')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8], ff_list, color='deepskyblue', label='FF', linestyle='-',
             marker='o', mfc='w', markersize=4, mfcalt='b')
    plt.grid(axis='y', linestyle='-.')
    plt.xlabel('Projects (Angular->NodeJS)', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.ylabel('F1', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.ylim([0, 1])
    plt.yticks(fontproperties='Times New Roman', size=16)
    plt.xticks(fontproperties='Times New Roman', size=16)
    figure.legend(loc='upper center', ncol=4, prop={'size': 16, 'family': 'Times New Roman'})
    plt.show()
    return bilstm_list, e2e_list, bert_list


def draw_portion():
    plt.figure()
    redundant_list, multiple_list, context_list, rules_list = get_bad_cases()
    # labels = ['Bad Case 1', 'Bad Case 2', 'Bad Case 3', 'Bad Case 4']
    # X = [sum(redundant_list), sum(multiple_list), sum(context_list), sum(rules_list)]
    # fig = plt.subplot(121)
    # plt.pie(X, labels=labels, autopct='%1.2f%%')  # 画饼图（数据，数据对应的标签，百分数保留两位小数点）

    ax1 = plt.subplot(111, projection='polar')
    # ax1.set_rlim(0, 12)
    # 创建数据
    data1 = np.random.randint(1, 10, 8)
    data2 = np.random.randint(1, 10, 8)
    data3 = np.random.randint(1, 10, 8)
    theta = np.arange(0, 2 * np.pi, 2 * np.pi / 8)
    theta = list(theta)
    sum_result = sum(rules_list + context_list + multiple_list + redundant_list)
    proportion_rules = float(sum(rules_list))/sum_result
    proportion_context = float(sum(context_list))/sum_result
    proportion_multiple = float(sum(multiple_list)) / sum_result
    proportion_redundant = float(sum(redundant_list)) / sum_result
    print('Rules: {}'.format(proportion_rules))
    print('Context: {}'.format(proportion_context))
    print('Multiple: {}'.format(proportion_multiple))
    print('Redundant: {}'.format(proportion_redundant))

    theta.append(theta[0])
    rules_list.append(rules_list[0])
    context_list.append(context_list[0])
    multiple_list.append(multiple_list[0])
    redundant_list.append(redundant_list[0])

    # 绘制雷达线
    ax1.plot(theta, rules_list, '.--', label='Missing interaction patterns')
    ax1.fill(theta, rules_list, alpha=0.2)
    ax1.plot(theta, context_list, '.--', label='Ignoring contextual information')
    ax1.fill(theta, context_list, alpha=0.2)
    ax1.plot(theta, multiple_list, '.--', label='Mixing multiple topics')
    ax1.fill(theta, multiple_list, alpha=0.2)
    ax1.plot(theta, redundant_list, '.--', label='Ignoring user relationships')
    ax1.fill(theta, redundant_list, alpha=0.2)


    plt.xticks(theta, ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8'])
    plt.xticks(fontproperties='Times New Roman', size=25)
    plt.yticks(fontproperties='Times New Roman', size=25)

    # plt.legend(loc='upper right', ncol=2)
    # plt.legend(loc='lower right', bbox_to_anchor=(1.3, 0.0), prop={'size': 20, 'family': 'Times New Roman'})
    plt.legend(loc='upper center', ncol=1, bbox_to_anchor=(1.5, 0.7), prop={'size': 18, 'family': 'Times New Roman'})

    plt.show()
    return


def draw_wrong_classification():
    x = ['1', '2', '3', '4', '5']
    y1 = [0, 0, 0]
    y2 = [0, 0]

    leven_return_list, f1_return_list, score_list, leven_dist_list, f1_list = get_predicted_truth_pair()
    x_temp = list(range(len(score_list)))
    # plt.plot(x1, y1, color="#8dd3c7", lw=2, label='Ground Truth')
    # plt.plot(x2, y2, color="#8dd3c7", lw=2)
    sorted_nums = sorted(enumerate(score_list), key=lambda x: x[1])
    idx = [i[0] for i in sorted_nums]
    nums = [i[1] for i in sorted_nums]
    f1_print, leven_print = [], []
    for id in idx:
        f1_print.append(f1_list[id])
        leven_print.append(leven_dist_list[id])
    print(idx)
    print(nums)
    true_f1 = nums.count(1.0)
    num_f1 = [0.0] * (len(nums) - true_f1) + [1.0] * true_f1
    plt.step(x_temp, nums, color="blue", where='post', lw=2, label='Ground Truth', linewidth=4)
    plt.scatter(x_temp, num_f1, color="green", lw=2, alpha=0.5, label='F1')

    plt.scatter(x_temp, f1_print, color='red', alpha=0.5, label='ARI')
    plt.scatter(x_temp, leven_print, color='#8dd3c7', alpha=0.5, label='DLD')
    plt.legend()
    # plt.xlim(1, 5)
    plt.show()
    return


def draw_dld_result():
    gt_annotation_list = [0.6625, 0.585, 0.475, 0.43, 0.605, 0.585, 0.5075, 0.5525]
    dic_data_dld, dic_data_dlr = reformat_data_for_ff_new_metrics()
    _, _, dic_result_metrics = draw_final_result()
    dld_list, dlr_list = [], []
    for project in dic_data_dld.keys():
        dld_project_data = np.array(dic_data_dld[project]).mean()
        dlr_project_data = np.array(dic_data_dlr[project]).mean()
        dld_list.append(dld_project_data)
        dlr_list.append(dlr_project_data)
    dl_result_list = [dlr * 0.5 + dld * 0.5 for dld, dlr in zip(dld_list, dlr_list)]
    project_list = list(dic_data_dld.keys())
    pj_x = []
    for i in range(len(project_list)):
        pj_x.append('P' + str(i + 1))
    plt.plot(pj_x, dic_result_metrics['f1'], color='g', linestyle='-', marker='x', label='F1')
    plt.plot(pj_x, dic_result_metrics['nmi'], color='black', linestyle='-', marker='x', label='NMI')
    plt.plot(pj_x, dic_result_metrics['ari'], color='r', linestyle='-', marker='x', label='ARI')
    plt.plot(pj_x, dic_result_metrics['shen-f'], color='purple', linestyle='-', marker='x', label='Shen-F')
    plt.plot(pj_x, dl_result_list, color='b', linestyle='-', marker='x', label='DLD')
    plt.plot(pj_x, gt_annotation_list, color='aqua', linestyle='-', marker='x', label='Ground Truth')
    plt.ylabel('Metrics')
    plt.xlabel('Projects (Angular->NodeJS)')
    plt.ylim([0.0, 1.0])
    plt.legend()
    plt.show()
    print('F1 Mean: {}'.format(np.array(dic_result_metrics['f1']).mean()))
    print('Ground Truth Mean: {}'.format(np.array(gt_annotation_list).mean()))
    print('DLD Mean: {}'.format(np.array(dl_result_list).mean()))
    return


def draw_box_plot_distribution():
    dict_metrics_return = reformat_ff_data()
    _, _, _, _, _, dict_metrics_new = get_predicted_truth_pair()
    metrics_traditional = ['NMI', 'ARI', 'Shen-F']
    metrics_used = ['F1', 'DLD']
    project_list, difference_list, metric_list = [], [], []
    for project in dict_metrics_return.keys():
        score_project_list = dict_metrics_return[project]['score']
        for metric_tradition in metrics_traditional:
            metric_data = dict_metrics_return[project][metric_tradition]
            values_differences = [calculate_similarity([metric], [score], 'MAE')
                                  for metric, score in zip(metric_data, score_project_list)]
                                  # if calculate_similarity([metric], [score], 'MAE') != 0.0]
            project_list += [project] * len(values_differences)
            difference_list += values_differences
            metric_list += [metric_tradition] * len(values_differences)
        truth_project_list = dict_metrics_new[project]['truth']
        for metric_use in metrics_used:
            metric_data = dict_metrics_new[project][metric_use]
            values_differences = [calculate_similarity([metric], [truth], 'MAE')
                                  for metric, truth in zip(metric_data, truth_project_list)]
                                  # if calculate_similarity([metric], [truth], 'MAE') != 0.0]
            project_list += [project] * len(values_differences)
            difference_list += values_differences
            metric_list += [metric_use] * len(values_differences)
    data_frame = {'project': project_list, 'differences': difference_list, 'experiment': metric_list}
    pd_temp = pd.DataFrame(data_frame)
    plt.subplot('111')
    sns.boxplot(x="project", y="differences", data=pd_temp,
                hue="experiment",
                # split=True,
                # linewidth=2,
                # width=0.8,
                # palette='Pastel1',
                order=list(dict_metrics_return.keys()),
                # scale = 'count',
                # gridsize=50,
                # inner = 'box',
                # bw = 0.8
                showfliers = False
                )
    plt.xlabel('')
    plt.ylabel('MAE', fontdict={'family': 'Times New Roman', 'size': 25})
    project_name_list = []
    for i in range(8):
        project_name_list.append(f'P{i + 1}')
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7], project_name_list)
    plt.yticks(fontproperties='Times New Roman', size=25)
    plt.xticks(fontproperties='Times New Roman', size=25)
    plt.legend(loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.28), prop={'size': 25, 'family': 'Times New Roman'})

    plt.show()
    return


def draw_new_violin_plot():
    dict_metrics_return = reformat_ff_data()
    _, _, _, _, _, dict_metrics_new = get_predicted_truth_pair()
    metrics_traditional = ['NMI', 'ARI', 'Shen-F']
    metrics_used = ['F1', 'DLD']#
    metrics_list, score_list, mark_list = [], [], []
    for project in dict_metrics_return.keys():
        score_project_list = dict_metrics_return[project]['score']
        for metric_tradition in metrics_traditional:
            metric_data = dict_metrics_return[project][metric_tradition]
            # values_differences = [calculate_similarity([metric], [score], 'MAE')
            #                       for metric, score in zip(metric_data, score_project_list)]
                                  # if calculate_similarity([metric], [score], 'MAE') != 0.0]
            metrics_list += [metric_tradition] * len(metric_data)
            score_list += metric_data
            mark_list += ['Measures'] * len(metric_data)
            metrics_list += [metric_tradition] * len(score_project_list)
            score_list += score_project_list
            mark_list += ['Human Satisfaction'] * len(score_project_list)
        truth_project_list = dict_metrics_new[project]['truth']
        for metric_use in metrics_used:
            metric_data = dict_metrics_new[project][metric_use]
            metrics_list += [metric_use] * len(metric_data)
            score_list += metric_data
            mark_list += ['Measures'] * len(metric_data)
            metrics_list += [metric_use] * len(truth_project_list)
            score_list += truth_project_list
            mark_list += ['Human Satisfaction'] * len(truth_project_list)
    # project_list, difference_list, metric_list = [], [], []
    df_result_dict = {'class': metrics_list, 'score': score_list, 'weight': mark_list}
    pd_temp = pd.DataFrame(df_result_dict)
    print(pd_temp)
    plt.subplot(111)
    sns.violinplot(x="class", y="score", data=pd_temp,
                   hue="weight",
                   split=True,
                   linewidth=2,  # 线宽
                   width=0.8,  # 箱之间的间隔比例
                   # inner='point',
                   orient='v',
                   cut=0,
                   scale='width',
                   palette='Set3',  # 设置调色板
                   order=['NMI', 'ARI', 'Shen-F', 'F1', 'DLD'],  # 筛选类别
                   # scale = 'count',  #测度小提琴图的宽度： area-面积相同,count-按照样本数量决定宽度,width-宽度一样
                   gridsize=50,  # 设置小提琴图的平滑度，越高越平滑
                   # inner = 'box', #设置内部显示类型 --> 'box','quartile','point','stick',None
                   # bw = 0.8      #控制拟合程度，一般可以不设置
                   )
    # plt.ylim([0, 1])
    plt.xlabel('')
    plt.ylabel('Value Distribution', fontdict={'family': 'Times New Roman', 'size': 25})

    plt.yticks(fontproperties='Times New Roman', size=25)
    plt.xticks(fontproperties='Times New Roman', size=25)
    plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.28), prop={'size': 25, 'family': 'Times New Roman'})
    plt.show()
    metrics_compare = metrics_traditional + metrics_used
    measurement_list = ['RMSE', 'MAE', 'MT', 'PST', 'VT', 'PEA', 'SPEA']
    for metric_each in metrics_compare:
        metric_compare_data = pd_temp[(pd_temp['class']==metric_each) & (pd_temp['weight']=='Metrics')]['score']
        truth_compare_data = pd_temp[(pd_temp['class']==metric_each) & (pd_temp['weight']=='Human Satisfaction')]['score']
        data_dict = dict()
        for measurement in measurement_list:
            similarity = calculate_similarity(metric_compare_data.values.tolist(), truth_compare_data.values.tolist(), measurement)
            data_dict[measurement] = similarity
        print(metric_each)
        print(data_dict)
    return


def draw_test():
    fig = plt.figure()

    x = [1, 2, 3, 4, 5, 6, 7]
    y = [1, 3, 4, 2, 5, 8, 6]

    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax1 = fig.add_axes([left, bottom, width, height])
    # ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax1.set_title('title')

    left, bottom, width, height = 0.2, 0.6, 0.25, 0.25
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.plot(x, y, 'g')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('title inside1')

    left, bottom, width, height = 0.6, 0.2, 0.25, 0.25
    plt.axes([left, bottom, width, height])
    plt.ylabel('y')
    plt.xticks([])
    plt.yticks([])

    plt.show()


if __name__ == '__main__':
    # draw_metrics_evaluation_comparison()
    # draw_line_comparison()
    # evaluate_significance()
    # draw_line_projects()
    # draw_empirical_diagram()


    # ff_total_migration, ff_total_retraining = draw_final_result()
    # print(ff_total_migration, ff_total_retraining)
    # print(scipy.stats.f_oneway([0, 1, 2], [2, 3, 4]))
    # data = str(scipy.stats.f_oneway([0, 1, 2,3,4,10], [2, 3, 4,2,3,3])).split(', ')[1]
    # pvalue = float(data[data.index('=') + 1: -1])
    # print(pvalue)
    # draw_violin_comparison()
    # draw_violin_metrics()

    # X1 = pd.Series([1, 2, 3, 4, 5, 6])
    # Y1 = pd.Series([0.3, 0.9, 2.7, 2, 3.5, 5])
    #
    # print(X1.corr(Y1, method="pearson"))
    # # X1.cov(Y1) / (X1.std() * Y1.std())
    # print(X1.corr(Y1,method='spearman'))


    # calculate_gt_comparison()
    # draw_portion()

    # draw_wrong_classification()
    # draw_dld_result()

    # draw_box_plot_distribution()
    # draw_new_violin_plot()
    draw_final_result()
    # draw_test()