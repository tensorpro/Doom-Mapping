import os
import subprocess as sp
import re

exp_filter = re.compile('^(.*)-(.*)$', re.IGNORECASE)


def rename_v2(d='models'):
    """
    Move a v1 model to v2
    """
    mapping = find_files()
    for k in mapping.iterkeys():
        exp_dir = os.path.join(d, k)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        for v in mapping[k]:
            exp_name = k + '-' + v

            curr_dir = os.path.join(d, exp_name)

            new_exp_dir = os.path.join(exp_dir, v)
            sp.call(['mv', curr_dir, new_exp_dir])
            old_log_dir = os.path.join(new_exp_dir, exp_name + '.log')
            old_model_dir = os.path.join(new_exp_dir, exp_name + '.pth')

            new_log_dir = os.path.join(new_exp_dir, 'info.log')
            new_model_dir = os.path.join(new_exp_dir, 'model.pth')
            sp.call(['mv', old_log_dir, new_log_dir])
            sp.call(['mv', old_model_dir, new_model_dir])


def match(dirs):
    """
    Match into (experiment, title)
    """
    def split_groups(x):
        m = exp_filter.match(x)
        if m:
            return m.group(1), m.group(2)
        else:
            raise ValueError, "Cannot parse experiment name"

    return map(split_groups, dirs)


def find_files(d='models'):
    dirs = os.listdir(d)
    m = match(dirs)
    plot_map = {}
    for k, v in m:
        if plot_map.get(k) is None:
            plot_map[k] = [v]
        else:
            plot_map[k].append(v)

    return plot_map


if __name__ == '__main__':
    rename_v2()
