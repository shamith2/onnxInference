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

filepath = os.path.join(root, 'results', 'onnxProfile', 'logs', 'sdxlt_unet_summary_analysis.csv')
save_directory = os.path.join(root, 'results', 'onnxProfile', 'plots')

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

table = pandas.read_csv(filepath)

output_memory = table['Output Memory (in Bytes)'][:-1].to_numpy(dtype=numpy.int64)
total_memory = table['Memory (in MB)'][:-1].to_numpy(dtype=numpy.float64)

main_memory_total_memory_accesses = 0
memory_usage = {}
consecutive_output_memory_usage = {}
optimized_memory_usage = {}
histogram_dict = {}

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
    
    elif element > threshold * 1.0:
        return str(ceil_elem), '>' + str(threshold)

    else:
        return str(ceil_elem), '(' + str(ceil_elem - 1) + ',' + str(ceil_elem) + ']'


# optimize operators having consecutive same output memory size
for i, element in enumerate(total_memory):
    key = getDictKey(element, threshold)

    if memory_usage.get(key, None) is None:
        memory_usage[key] = 1
        consecutive_output_memory_usage[key] = 0
    
    else:
        memory_usage[key] += 1

        if output_memory[i-1] == output_memory[i]:
            consecutive_output_memory_usage[key] += 1


# sort output_memory_usage by keys
def sortDict(dictionary):
    elements = list(dictionary.keys())
    elements.sort(key=lambda x: int(x[0]))
    
    sorted_dictionary = {element: int(dictionary[element]) for element in elements}

    return sorted_dictionary

sorted_memory_usage = sortDict(memory_usage)
sorted_consecutive_output_memory_usage = sortDict(consecutive_output_memory_usage)

for element in sorted_memory_usage:
    optimized_memory_usage[element] = sorted_memory_usage[element] - sorted_consecutive_output_memory_usage[element]


if threshold:
    threshold_key = '>' + str(threshold)

    for element in optimized_memory_usage:
        if int(element[0]) <= threshold:
            histogram_dict[element] = optimized_memory_usage[element]
        
        else:
            if histogram_dict.get(threshold_key, None) is None:
                histogram_dict[threshold_key] = optimized_memory_usage[element]
            
            else:
                histogram_dict[threshold_key] += optimized_memory_usage[element]
            
            main_memory_total_memory_accesses += int(element[0]) * optimized_memory_usage[element]
    
    histogram_dict[(threshold_key, threshold_key)] = histogram_dict.pop(threshold_key)


max_operator_memory = max([int(key[0]) for key in optimized_memory_usage.keys()])

print('Insights for {}:'.format(model_name))
print('Maximum Memory Size of any Operator: {} MB'.format(max_operator_memory))
print('Total Memory of Operators that have memory size > {} MB: {} MB'.format(threshold, main_memory_total_memory_accesses))

if histogram_dict:
    fig, ax = plt.subplots(figsize=(12, 12))

    num_keys = len(histogram_dict)
    _, keys = zip(*histogram_dict.keys())
    values = list(histogram_dict.values())
    plot_keys = []
    plot_values = []

    # do not plot value when value percent < 1%
    del_idx = []
    for i, value in enumerate(values):
        if (value * 100.0) / sum(values) < 1.0:
            del_idx.append(i)
    
    for j in range(num_keys):
        if j not in del_idx:
            plot_keys.append(keys[j])
            plot_values.append(values[j])

    explode = [0] * len(plot_keys)
    explode[-1] = 0.1
    
    _, _, pcts = ax.pie(
        plot_values,
        explode=explode,
        labels=[key + ' MB' for key in plot_keys],
        autopct='%1d%%',
        wedgeprops={'edgecolor': 'black'},
        textprops={'fontsize': 12, 'fontweight': 'bold'},
        startangle=90
    )
    
    plt.setp(pcts, size=14, color='white', fontweight='bold')

    # set title
    fig.suptitle('{}\n'.format(model_name), fontweight='bold')

    ax.set_title('Should Weights + Output of an Operator\nbe stored in Main Memory ' + 
                 'during single inference?\n\nIf memory size of the Operator > {} MB\n'.format(threshold) + 
                 '(on-chip memory) with no NPU or SLC cache\n')

    plt.tight_layout()

    fig.savefig(os.path.join(save_directory, '_'.join(model_name.lower().split(' ')) + '_' + str(threshold) + 'mb_' + 'plot.png'))

    plt.show()
