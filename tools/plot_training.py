import sys
import re
import numpy as np
#import matplotlib
#matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


def get_block(file_ptr, start_exp, end_exps, inclusive=True):
    if isinstance(end_exps, str):
        end_exps = [end_exps]
    while True:
        line = file_ptr.readline()
        if not line:
            return None

        if re.match(start_exp, line):
            prev_line = file_ptr.tell()
            block = [line]
            while True:
                line = file_ptr.readline()
                if not line:
                    raise ValueError('Unexpected end of file.')
                block.append(line)
                if any([re.match(exp, line) for exp in end_exps]):
                    if inclusive:
                        return block
                    else:
                        file_ptr.seek(prev_line)
                        return block[:-1]
                prev_line = file_ptr.tell()


def find_train_outputs(logFile):
    START_PATTERN = '.+Creating training net from net file:'
    END_PATTERN = '.+Network initialization done'

    f = open(logFile, 'r')
    block = get_block(f, START_PATTERN, END_PATTERN)
    if block is None:
        raise ValueError('Unexpected end of file.')

    outputs = []
    PATTERN = ".+This network produces output (.+)\n"
    for line in block:
        m = re.match(PATTERN, line)
        if m:
            outputs.append(m.group(1))
    return outputs


def find_test_outputs(logFile):
    from collections import OrderedDict
    START_PATTERN = '.+Creating test net (\(\#\d+\)) specified by net file:'
    END_PATTERN = '.+Network initialization done'

    f = open(logFile, 'r')
    outputs = OrderedDict()
    while True:
        block = get_block(f, START_PATTERN, END_PATTERN)
        if block is None:
            break

        net_id = re.match(START_PATTERN, block[0]).group(1)
        outputs[net_id] = []
        PATTERN = ".+This network produces output (.+)\n"
        for line in block:
            m = re.match(PATTERN, line)
            if m:
                outputs[net_id].append(m.group(1))
    return outputs


def extract_time(line):
    day = float(line[3:5])
    hour = float(line[6:8])
    min = float(line[9:11])
    sec = float(line[12:14])
    msec = float(line[15:18])

    return msec/1000. + sec + min*60. + hour*3600. + day*3600.*24.


def extract_iter(line):
    PATTERN = ".+Iteration (\d+)"
    m = re.match(PATTERN, line)
    return int(m.group(1))


def extract_lr(line):
    PATTERN = ".+Iteration (\d+), lr = (\d+\.\d+|\d+e[+-]?\d+)"
    m = re.match(PATTERN, line)
    return float(m.group(2))


def extract_train_value(block, tag):
    PATTERN = ".+Train net output \#\d+: %s = (-?\d*\.?\d*[eE]?[+\-]?\d*|nan)" % tag
    for line in block:
        m = re.match(PATTERN, line)
        if m:
            return float(m.group(1))
    return None


def extract_test_value(block, tag):
    PATTERN = ".+Test net output \#\d+: %s = (\d+\.?\d*|\d+e[+-]?\d+|nan)" % tag
    for line in block:
        m = re.match(PATTERN, line)
        if m:
            return float(m.group(1))
    return None


def parse_train(logFile):
    START_PATTERN = ".+Iteration (\d+), loss = (\d+\.\d+|\d+e[+-]?\d+|nan)"
    END_PATTERN = ".+Iteration (\d+), lr = (\d+\.\d+|\d+e[+-]?\d+|nan)"

    outputs = find_train_outputs(logFile)
    training_points = dict([('time', []), ('lr', []), ('iter', [])] + [(out, []) for out in outputs])

    f = open(logFile, 'r')
    while True:
        block = get_block(f, START_PATTERN, END_PATTERN)
        if block is None:
            break

        training_points['iter'].append(extract_iter(block[0]))
        training_points['time'].append(extract_time(block[0]))
        training_points['lr'].append(extract_lr(block[-1]))
        for out in outputs:
            training_points[out].append(extract_train_value(block, out))
    return outputs, training_points


def parse_test(logFile):
    net_outputs = find_test_outputs(logFile)
    testing_points = {net: dict([('time', []), ('iter', [])] + [(out, []) for out in net_outputs[net]]) for net in net_outputs.keys()}
    f = open(logFile, 'r')
    while True:
        for net_id in net_outputs.keys():
            START_PATTERN = ".+Iteration (\d+), Testing net %s" % net_id.replace('(', '\(').replace(')', '\)').replace('#', '\#')
            END_PATTERNS = [".+Iteration (\d+), loss = ", ".+Iteration (\d+), Testing net "]
            block = get_block(f, START_PATTERN, END_PATTERNS, inclusive=False)
            if block is None:
                return net_outputs, testing_points

            for line in block:
                m = re.match(START_PATTERN, line)
                if m:
                    testing_points[net_id]['iter'].append(extract_iter(line))
                    testing_points[net_id]['time'].append(extract_time(line))
                    for out in net_outputs[net_id]:
                        testing_points[net_id][out].append(extract_test_value(block, out))


def plot_training(logFile):
    outputs_train, training_points = parse_train(logFile)
    outputs_test, test_points = parse_test(logFile)

    all_outputs = outputs_train
    for aux in outputs_test.itervalues():
        all_outputs += aux

    for stat in set(all_outputs):
        plt.figure()
        plt.plot(training_points['iter'], training_points[stat], label='Train')
        for net_id in outputs_test.keys():
            if stat in outputs_test[net_id]:
                plt.plot(test_points[net_id]['iter'], test_points[net_id][stat], label='Test %s' % net_id)
        plt.ylabel(stat)
        plt.legend(loc='lower right')
    plt.show()

if __name__ == '__main__':
    plot_training(sys.argv[1])



