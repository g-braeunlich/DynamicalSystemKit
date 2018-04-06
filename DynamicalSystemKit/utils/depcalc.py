from collections import deque


def depcalc(dep_dict):
    order = []
    dep_dict = {node: set(deps) for node, deps in dep_dict.items()}
    while True:
        empty_keys = set(key for key, val in dep_dict.items() if not val)
        for key in empty_keys:
            del dep_dict[key]
        for deps in dep_dict.values():
            empty_keys -= deps
        order += list(empty_keys)
        if not dep_dict:
            break
        leaves = set.union(*dep_dict.values()) - dep_dict.keys()
        order += list(leaves)
        for deps in dep_dict.values():
            deps -= leaves
        if not leaves:
            break
    if dep_dict:
        while True:
            all_deps = set.union(*dep_dict.values())
            keys = set(dep_dict.keys()) & all_deps
            if keys == set(dep_dict.keys()):
                break
            dep_dict = {key: val for key,
                        val in dep_dict.items() if key in keys}
        raise ValueError("Cyclic dependency detected. Involved nodes: {}".format(
            all_deps))
    return order


def deporder(iterable, dependencies):
    leaves = deque(item for item in iterable if not dependencies.get(item))
    dependencies = {item: set(deps)
                    for item, deps in dependencies.items() if deps}
    remaining_items = set(iterable)
    while leaves:
        leaf = leaves.popleft()
        remaining_items.remove(leaf)
        for deps in dependencies.values():
            deps.discard(leaf)
        new_leaves = [item for item, deps in dependencies.items()
                      if not deps]
        leaves.extend(new_leaves)
        for l in new_leaves:
            del dependencies[l]
        yield leaf
    if remaining_items:
        raise ValueError("Circular dependency!")
