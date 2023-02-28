from __future__ import print_function, absolute_import

import matplotlib
# matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np

import os
import re
import pprint

from scipy.signal import savgol_filter

import colorama
colorama.init(autoreset=True)

# debug use
pp = pprint.PrettyPrinter()

LINESTYLES = ['-', '--', '-.', ':']
MAX_COLORS = 6

#############################
# Parsing Utility Functions #
#############################


def parse_logs(file, smooth=False):
    loss_re = re.compile(
        'finished up to training Policy at step: ([0-9]+), loss: ([0-9]+\.[0-9]+)', re.IGNORECASE)
    mapping = []
    with open(file, 'r') as f:
        for line in f.readlines():
            info = line[23:].rstrip()
            m = loss_re.match(info)
            if m is None:
                continue

            step = m.group(1)
            loss = m.group(2)

            mapping.append((step, loss))

    try:
        x, y = zip(*mapping)
    except ValueError:
        print(colorama.Fore.RED + 'Could not find valid log for %s' % (file))
        return [], []

    if smooth:
        try:
            y = savgol_filter(y, 101, 1)
        except TypeError:
            print('Could not use filter for %s, not applying filter' % (file))

    return x, y


def parse_file(file, x_idx=(0, 0), y_idx=(1, 0), smooth=False):
    x_sec, x_sub_sec = x_idx
    y_sec, y_sub_sec = y_idx

    mapping = []
    with open(file, 'r') as f:
        for line in f.readlines():
            parsed = line.split('|')

            itr, meas = int(parsed[x_sec].split()[x_sub_sec]), float(
                parsed[y_sec].split()[y_sub_sec])
            mapping.append((itr, meas))

    try:
        x, y = zip(*mapping)
    except ValueError:
        print(colorama.Fore.RED + 'Could not find valid file for %s' % (file))
        return [], []

    if smooth:
        try:
            y = savgol_filter(y, 11, 1)
        except TypeError:
            print('Could not use filter for %s, not applying filter' % (file))

    return x, y

################
# Measurements #
################


def parse_times(file, type):
    pass


def parse_measurements(file, type, smooth=False):
    if type == 'd2':
        idx = 0
    elif ((type[:len('d3')] == 'd3') or
          (type[:len('d4')] == 'd4') or
          (type[:len('d0')] == 'd0') or
          (type[:len('simpler')] == 'simpler')):
        idx = 2
    else:
        raise ValueError, "Could not resolve the experiment type: %s" % (type)

    return parse_file(file, x_idx=(0, 0), y_idx=(1, idx), smooth=smooth)


def plot_measurements(d='models', smooth=False):
    not_ds_store = re.compile("(?!\.ds_store).*$", re.IGNORECASE)
    dirs = filter(lambda x: not_ds_store.match(x), os.listdir(d))
    for k in dirs:
        plot_keys = filter(lambda x: not_ds_store.match(x),
                           sorted(os.listdir(os.path.join(d, k))))

        hsv = plt.get_cmap('hsv')
        colors = hsv(np.linspace(0, 1.0, num=MAX_COLORS, endpoint=False))

        try:
            for i, v in enumerate(plot_keys):
                fdir = os.path.join(d, k, v, 'log_brief.txt')
                try:
                    x, y = parse_measurements(fdir, k, smooth=smooth)
                    plt.plot(x, y, label=v,
                             linestyle=LINESTYLES[i / MAX_COLORS], color=colors[i % MAX_COLORS])
                except IOError:
                    print("Performance: Could not find %s, skipping" % (fdir))
                    continue

            plt.title('%s Performance' % (k.capitalize()))
            plt.legend(loc='best')
            plt.xlabel('steps')
            plt.grid(True, linestyle='dashed')

            if k == 'd2':
                plt.ylabel('Health')
            else:
                plt.ylabel('Frags')

            if smooth:
                plot_dir = os.path.join('plot/smooth', '%s-smooth.png' % (k))
                plt.savefig(plot_dir, dpi=150)
            else:
                plot_dir = os.path.join('plot/raw', '%s.png' % (k))
                plt.savefig(plot_dir, dpi=150)

            plt.clf()
        except ValueError:
            print(colorama.Fore.RED + "Could not resolve the experiment type: %s" % (k))


def plot_loss(d='models', smooth=False):
    not_ds_store = re.compile("(?!\.ds_store).*$", re.IGNORECASE)
    dirs = filter(lambda x: not_ds_store.match(x), os.listdir(d))

    for k in dirs:
        plot_keys = filter(lambda x: not_ds_store.match(x),
                           sorted(os.listdir(os.path.join(d, k))))

        hsv = plt.get_cmap('hsv')
        colors = hsv(np.linspace(0, 1.0, num=MAX_COLORS, endpoint=False))

        for i, v in enumerate(plot_keys):
            fdir = os.path.join(d, k, v, 'info.log')
            try:
                x, y = parse_logs(fdir, smooth=smooth)
                plt.plot(x, y, label=v,
                         linestyle=LINESTYLES[i / MAX_COLORS], color=colors[i % MAX_COLORS])
            except IOError:
                print("Losses: Could not find %s, skipping" % (fdir))
                continue

        plt.title('%s Losses' % (k.capitalize()))
        plt.legend(loc='best')
        plt.xlabel('Steps')
        plt.ylabel('Loss')

        plt.grid(True, linestyle='dashed')

        if smooth:
            plot_dir = os.path.join('plot/smooth', '%s-loss-smooth.png' % (k))
            plt.savefig(plot_dir, dpi=150)
        else:
            plot_dir = os.path.join('plot/raw', '%s-loss.png' % (k))
            plt.savefig(plot_dir, dpi=150)

        plt.clf()


def main():
    if not os.path.exists('plot/smooth'):
        os.makedirs('plot/smooth')

    if not os.path.exists('plot/raw'):
        os.makedirs('plot/raw')

    plot_measurements()
    plot_measurements(smooth=True)

    plot_loss()
    plot_loss(smooth=True)


if __name__ == '__main__':
    main()
