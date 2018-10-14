#!/usr/bin/env python3

import argparse

from DynamicalSystemKit import model
from DynamicalSystemKit.utils import attr


def prm(S, prm_list=()):
    for p in prm_list:
        print(p, "=", attr.getattr_recursive(S, p.split(".")))


def evl(S, s=None):
    f_str, arg_str = s.split('(', 1)
    args = arg_str.rstrip(')').split(",")
    kwargs_list = [a for a in args if '=' in a]
    kwargs = {
        k: v for k, v in [args.pop(a).split('=', 1) for a in kwargs_list]}

    f = attr.getattr_recursive(S, f_str.split("."))
    print(s + " =", f(*args, **kwargs))


def exec_action(*, model_file, action, **kwargs):
    _model = load_module(model_file).Model
    f = globals()[action]
    f(_model, **kwargs)


def load_module(path):
    import sys
    import os
    import importlib
    module_path = path.rstrip("/")
    module_dir = os.path.dirname(module_path)
    module_name = os.path.basename(module_path)

    if module_name[-3:] == ".py":
        module_name = module_name[:-3]
    if module_dir:
        sys.path.append(module_dir)
    return importlib.import_module(module_name)


parser = argparse.ArgumentParser(description='Model info')
parser.add_argument('model_file', help='Model file')
subparsers = parser.add_subparsers(dest='action', help='Action help')
parser_prm = subparsers.add_parser('prm', help='Output a parameter')
parser_prm.add_argument('prm_list', nargs='*', help='Specify the parameters')
parser_evl = subparsers.add_parser('evl', help='Evaluate a method')
parser_evl.add_argument('s', help='The function call to evaluate')

cmd_args = parser.parse_args()
exec_action(**vars(cmd_args))
