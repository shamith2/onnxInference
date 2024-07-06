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

try:
    model_name = sys.argv[1]
    filename = sys.argv[2]
    threshold = int(sys.argv[3])

except IndexError:
    raise Exception("[Usage] > python onnx_profiling_analysis.py [name of the model being analysed] [analysis csv filename] [size of NPU on-chip memory in MB]")

model_dir = '_'.join(model_name.lower().split(' '))
filepath = os.path.join(root, 'results', 'onnxProfile', 'logs', model_dir, filename + '_summary.csv')
save_directory = os.path.join(root, 'results', 'onnxProfile', 'plots', model_dir)

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

    threshold_key = '>' + str(threshold)
    identifier_idx1 = plot_keys.index(threshold_key)
    identifier_idx2 = plot_weighed_keys.index(threshold_key)

    explode1 = [0] * len(plot_keys)
    explode1[identifier_idx1] = 0.1

    explode2 = [0] * len(plot_weighed_keys)
    explode2[identifier_idx2] = 0.1

    identifier1 = (plot_values[identifier_idx1] * 100.0) / sum(plot_values)
    identifier2 = (plot_weighed_values[identifier_idx2] * 100.0) / sum(plot_weighed_values)

    def pctPrint(pct, allvals, identifier, is_mb):
        absolute = int(numpy.round(pct / 100.0 * numpy.sum(allvals)))
        value = int(round(pct, 0))
        
        if math.isclose(pct, identifier, rel_tol=1e-6):
            return '{}%\n{} MB'.format(value, absolute) if is_mb else '{}%\n{}'.format(value, absolute)
        
        else:
            return '{}%'.format(value)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))
    
    _, _, pcts1 = ax1.pie(
        plot_values,
        explode=explode1,
        labels=[key + ' MB' for key in plot_keys],
        autopct=lambda pct: pctPrint(pct, plot_values, identifier1, is_mb=False),
        wedgeprops={'edgecolor': 'black'},
        textprops={'fontsize': 12, 'fontweight': 'bold'},
        startangle=90
    )

    _, _, pcts2 = ax2.pie(
        plot_weighed_values,
        explode=explode2,
        labels=[key + ' MB' for key in plot_weighed_keys],
        autopct=lambda pct: pctPrint(pct, plot_weighed_values, identifier2, is_mb=True),
        wedgeprops={'edgecolor': 'black'},
        textprops={'fontsize': 12, 'fontweight': 'bold'},
        startangle=180
    )
    
    plt.setp(pcts1, size=14, color='white', fontweight='bold')
    plt.setp(pcts2, size=14, color='white', fontweight='bold')

    # set title
    fig.suptitle('{}\n\nShould Weights + Output of an Operator\nbe stored in Main Memory or Last-level cache'.format(model_name) + 
                 'during single inference?\n\nIf memory size of the Operator > {} MB\n'.format(threshold) + 
                 '(on-chip memory) with no NPU cache\n\n', fontweight='bold')
    
    ax1.set_title('Breakdown based on Count')
    ax2.set_title('Breakdown based on Weighed Count')

    plt.tight_layout()

    fig.savefig(os.path.join(save_directory, model_dir + '_' + str(threshold) + 'mb_' + 'pie_plot.png'))

    plt.close(fig)

    unique_ops = list(set(optimized_operator_timeline))
    num_unique_ops = len(unique_ops)

    operator_memory_list = [int(key[0]) for key in optimized_memory_usage.keys()]

    max_operator_memory = max(operator_memory_list)

    # adapted from https://stackoverflow.com/questions/62802061/python-find-outliers-inside-a-list
    def find_outliers(data: list, m: int = 7):
        data = numpy.asarray(data)
        d = numpy.abs(data - numpy.median(data))
        median_d = numpy.median(d)
        std_dev = d / (median_d if median_d else 1.0)
        
        return data[numpy.nonzero(std_dev > m)]

    # find and remove outliers while plotting operators' memory values
    outlier_list = find_outliers(operator_memory_list)

    refined_operator_timeline = ()
    refined_operator_usage_timeline = ()

    for outlier in outlier_list:
        for i, elem in enumerate(optimized_operator_usage_timeline):
            if math.ceil(elem) < outlier:
                refined_operator_timeline += (optimized_operator_timeline[i],)
                refined_operator_usage_timeline += (optimized_operator_usage_timeline[i],)

    plot_ops = []
    refined_unique_ops = list(set(refined_operator_timeline))

    for i, op in enumerate(refined_operator_timeline):
        plot_ops.append(refined_unique_ops.index(op))

    fig, ax = plt.subplots(figsize=(18, 12))

    x = range(len(refined_operator_timeline))

    scatter = ax.scatter(
        x,
        refined_operator_usage_timeline,
        c=plot_ops
    )

    # set axes labels and title
    ax.set_xticks(x)
    ax.set_yticks(range(threshold, threshold + math.ceil(max(refined_operator_usage_timeline)), 5))

    handles, _ = scatter.legend_elements()

    legend = ax.legend(
        handles=handles,
        labels=refined_unique_ops,
        loc='best',
        title='Operators'
    )
    
    ax.add_artist(legend)
    
    ax.set_xlabel('Operator')
    ax.set_ylabel('Operator Memory Size [in MB]')

    plt.tick_params(bottom=True, labelbottom=False)

    fig.suptitle('{}\n\nShould Weights + Output of an Operator\nbe stored in Main Memory or Last-level cache'.format(model_name) + 
                 'during single inference?\n\nIf memory size of the Operator > {} MB\n'.format(threshold) + 
                 '(on-chip memory) with no NPU cache\n\nMaximum Operator Memory: {} MB\n\n'.format(max_operator_memory), fontweight='bold')

    ax.set_title('Memory Size of Operators (> {} MB)'.format(threshold))
    
    plt.tight_layout()

    fig.savefig(os.path.join(save_directory, model_dir + '_' + str(threshold) + 'mb_' + 'operators_plot.png'))

    plt.close(fig)
