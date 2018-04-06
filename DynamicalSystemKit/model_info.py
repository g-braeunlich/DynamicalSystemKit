#!/usr/bin/env python3

import argparse

from lib import elements
from sim import model
from sim.utils import attr


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


def exec_action(model_file=None, action=None, **kwargs):
    S = model.Model.from_file(model_file, elements)
    f = globals()[action]
    f(S, **kwargs)


parser = argparse.ArgumentParser(description='Model info')
parser.add_argument('model_file', help='Model file')
subparsers = parser.add_subparsers(dest='action', help='Action help')
parser_prm = subparsers.add_parser('prm', help='Output a parameter')
parser_prm.add_argument('prm_list', nargs='*', help='Specify the parameters')
parser_evl = subparsers.add_parser('evl', help='Evaluate a method')
parser_evl.add_argument('s', help='The function call to evaluate')

cmd_args = parser.parse_args()
exec_action(**vars(cmd_args))
