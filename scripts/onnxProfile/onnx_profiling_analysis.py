# Script to analyse onnx model memory profiler summary

import os
import sys

import math
import numpy
import pandas

from matplotlib import pyplot as plt

from pathlib import Path

root = Path(__file__).parents[2].resolve()
workspace = Path(__file__).parent.resolve()

filepath = os.path.join(root, 'results', 'onnxProfile', 'logs', 'sdxlt_unet_summary.csv')
save_directory = os.path.join(root, 'results', 'onnxProfile', 'plots')

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

table = pandas.read_csv(filepath)

operators = table['Operator']
output_memory_bytes = table['Output Memory (in Bytes)'][:-1].to_numpy(dtype=numpy.int64)
total_memory = table['Memory (in MB)'][:-1].to_numpy(dtype=numpy.float64)

optimized_memory_usage = {}
optimized_operator_timeline = ()
optimized_operator_usage_timeline = ()
histogram_dict1 = {}
histogram_dict2 = {}

try:
    model_name = sys.argv[1]
    threshold = int(sys.argv[2])

except IndexError:
    raise Exception("[Usage] > python onnx_profiling_analysis.py [name of the model being analysed] [size of NPU on-chip memory in MB]")
    sys.exit()


def getDictKey(element, threshold):
    ceil_elem = math.ceil(element)

    if ceil_elem == 0:
        return '0', '0'
    
    if ceil_elem == 1:
        return '1', '(0,1]'
    
    elif element > float(threshold):
        return str(ceil_elem), '>' + str(threshold)

    else:
        return str(ceil_elem), '(' + str(ceil_elem - 1) + ',' + str(ceil_elem) + ']'


# optimize operators having consecutive same output memory size
for i, element in enumerate(total_memory):
    key = getDictKey(element, threshold)

    if optimized_memory_usage.get(key, None) is None:
        optimized_memory_usage[key] = [1, (element,)]

        if element > float(threshold):
            optimized_operator_timeline += (operators[i],)
            optimized_operator_usage_timeline += (element,)
    
    else:
        if output_memory_bytes[i-1] != output_memory_bytes[i]:
            optimized_memory_usage[key][0] += 1
            optimized_memory_usage[key][1] += (element,)
            
            if element > float(threshold):
                optimized_operator_timeline += (operators[i],)
                optimized_operator_usage_timeline += (element,)


# sort output_memory_usage by keys
def sortDict(dictionary):
    elements = list(dictionary.keys())
    elements.sort(key=lambda x: int(x[0]))
    
    sorted_dictionary = {element: dictionary[element] for element in elements}

    return sorted_dictionary


optimized_memory_usage = sortDict(optimized_memory_usage)

if threshold:
    threshold_key = '>' + str(threshold)

    for element in optimized_memory_usage:
        if element[1] != threshold_key:
            histogram_dict1[element] = optimized_memory_usage[element][0]
            histogram_dict2[element] = sum(optimized_memory_usage[element][1])
        
        else:
            if histogram_dict1.get(threshold_key, None) is None:
                histogram_dict1[threshold_key] = optimized_memory_usage[element][0]
                histogram_dict2[threshold_key] = sum(optimized_memory_usage[element][1])
            
            else:
                histogram_dict1[threshold_key] += optimized_memory_usage[element][0]
                histogram_dict2[threshold_key] += sum(optimized_memory_usage[element][1])
    
    histogram_dict1[(threshold_key, threshold_key)] = histogram_dict1.pop(threshold_key)
    histogram_dict2[(threshold_key, threshold_key)] = histogram_dict2.pop(threshold_key)

    max_operator_memory = max([int(key[0]) for key in optimized_memory_usage.keys()])

    print('Insights for {}:'.format(model_name))
    print('Maximum Memory Size of any Operator: {} MB'.format(max_operator_memory))
    print('Total Memory of Operators that have memory size > {} MB: {} MB'.format(threshold, round(histogram_dict2[(threshold_key, threshold_key)])))

if histogram_dict1 and histogram_dict2:
    num_keys = len(histogram_dict1)
    _, keys = zip(*histogram_dict1.keys())
    values = list(histogram_dict1.values())
    
    plot_keys = []
    plot_values = []

    weighed_values = list(histogram_dict2.values())
    
    plot_weighed_keys = []
    plot_weighed_values = []

    # do not plot value when value percent < 1%
    del_idx = []
    for i, value in enumerate(values):
        if (value * 100.0) / sum(values) < 1.0:
            del_idx.append(i)
    
    for j in range(num_keys):
        if j not in del_idx:
            plot_keys.append(keys[j])
            plot_values.append(values[j])
    
    del_idx = []
    for i, value in enumerate(weighed_values):
        if (value * 100.0) / sum(weighed_values) < 1.0:
            del_idx.append(i)
    
    for j in range(num_keys):
        if j not in del_idx:
            plot_weighed_keys.append(keys[j])
            plot_weighed_values.append(weighed_values[j])

    explode1 = [0] * len(plot_keys)
    explode1[-1] = 0.1

    explode2 = [0] * len(plot_weighed_keys)
    explode2[-1] = 0.1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 18))
    
    _, _, pcts1 = ax1.pie(
        plot_values,
        explode=explode1,
        labels=[key + ' MB' for key in plot_keys],
        autopct='%1d%%',
        wedgeprops={'edgecolor': 'black'},
        textprops={'fontsize': 12, 'fontweight': 'bold'},
        startangle=90
    )

    _, _, pcts2 = ax2.pie(
        plot_weighed_values,
        explode=explode2,
        labels=[key + ' MB' for key in plot_weighed_keys],
        autopct='%1d%%',
        wedgeprops={'edgecolor': 'black'},
        textprops={'fontsize': 12, 'fontweight': 'bold'},
        startangle=180
    )
    
    plt.setp(pcts1, size=14, color='white', fontweight='bold')
    plt.setp(pcts2, size=14, color='white', fontweight='bold')

    # set title
    fig.suptitle('{}\n\nShould Weights + Output of an Operator\nbe stored in Main Memory '.format(model_name) + 
                 'during single inference?\n\nIf memory size of the Operator > {} MB\n'.format(threshold) + 
                 '(on-chip memory) with no NPU or Last-level cache\n', fontweight='bold')
    
    ax1.set_title('Breakdown based on Count')
    ax2.set_title('Breakdown based on Weighed Count')

    plt.tight_layout()

    fig.savefig(os.path.join(save_directory, '_'.join(model_name.lower().split(' ')) + '_' + str(threshold) + 'mb_' + 'pie_plot.png'))

    plt.close(fig)

    fig, ax = plt.subplots(figsize=(36, 12))

    x = range(len(optimized_operator_timeline))

    ax.scatter(
        x,
        optimized_operator_usage_timeline
    )

    # set axes labels and title
    ax.set_xticks(x)
    ax.set_xticklabels(optimized_operator_timeline)
    plt.setp(ax.get_xticklabels(), rotation=90)
    
    ax.set_xlabel('Operator')
    ax.set_ylabel('Operator Memory Size (> {} MB) [in MB]'.format(threshold))

    fig.suptitle('{}\n\nShould Weights + Output of an Operator\nbe stored in Main Memory '.format(model_name) + 
                 'during single inference?\n\nIf memory size of the Operator > {} MB\n'.format(threshold) + 
                 '(on-chip memory) with no NPU or Last-level cache\n', fontweight='bold')

    ax.set_title('Memory Size of Operators (> {} MB)'.format(threshold))
    
    plt.tight_layout()

    fig.savefig(os.path.join(save_directory, '_'.join(model_name.lower().split(' ')) + '_' + str(threshold) + 'mb_' + 'operators_plot.png'))

    plt.close(fig)
