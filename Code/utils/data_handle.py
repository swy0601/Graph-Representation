import statistics
from pathlib import Path

import re


def resoleTXT(txt):
    my_file = Path(txt)
    if not my_file.is_file():
        print(txt, " file is not exist, Please run Model of it firstly")
        return
    f = open(txt, "r")
    print(txt.split("\\")[-1].replace("_0.0003_8-2_20-fixed-5-128.txt", "").replace("demon3-", ""))
    out_data = f.read()
    f.close()
    lines = out_data.split("\n")
    metrics = [[], [], [], []]
    for line in lines:
        if line != "":
            result = line[-51:]
            result = result.replace("F1", "")
            result = re.sub(r'[a-zA-Z: ]', r'', result)
            result_list = result.split(",")
            metrics[0].append(float(result_list[0]))
            metrics[1].append(float(result_list[1]))
            metrics[2].append(float(result_list[2]))
            metrics[3].append(float(result_list[3]))

    cut = 200
    length = int(len(metrics[0]) / cut)
    patch = [[], [], [], []]
    for i in range(length):
        patch[0].append(max(metrics[0][cut * i:cut * (i + 1)]))
        patch[1].append(max(metrics[1][cut * i:cut * (i + 1)]))
        patch[2].append(max(metrics[2][cut * i:cut * (i + 1)]))
        patch[3].append(max(metrics[3][cut * i:cut * (i + 1)]))

    print("Each Accuracy of Five Patch:", patch[0])

    mean_results = []
    for i in patch:
        mean_results.append(statistics.mean(i))

    print(f'Average Results of Five Patch:'
          f' Accuracy: {mean_results[0]:.4f}, F-Measure:{mean_results[1]:.4f}, '
          f'AUC:{mean_results[2]:.4f}, MCC:{mean_results[3]:.4f}')


if __name__ == '__main__':
    resoleTXT('../result_DMon_control_data1.txt')
    resoleTXT('../result_DMon_control_data2.txt')
    resoleTXT('../result_DMon_control_data1-two-class.txt')
    resoleTXT('../result_DMon_control_data1-two-class.txt')
