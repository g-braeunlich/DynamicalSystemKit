import json


_no_default = object()


def getattr_recursive_idx(obj, addr, default=_no_default):
    try:
        for part, idx in addr:
            obj = getattr(obj, part)
            if idx is not None:
                obj = obj[idx]
    except (AttributeError, IndexError) as err:
        if default is not _no_default:
            return default
        raise err
    return obj


class Node:

    def __init__(self, *, parent, alias, content, parent_node=None,
                 build_class=dict,
                 **kwargs):
        self.unpack_destinations = {(id(parent), alias): (parent, alias, ())}
        self.name = alias
        self.content = content
        self.post_references = []
        wrap_node(self, **kwargs)
        self.parent_nodes = set()
        self.parent_node = parent_node
        self.build_class = build_class
        if parent_node is not None:
            self.parent_nodes.add(parent_node)

    def unpack(self, **kwargs):
        substitute = self.build(**kwargs)
        for p, alias, addr in self.unpack_destinations.values():
            _substitute = getattr_recursive_idx(substitute, addr)
            p[alias] = _substitute
        for ref in self.post_references:
            ref.parent = substitute

    def build(self, **kwargs):
        return self.content

    def is_leaf(self):
        return all(not isinstance(child, Node) for child in self.content.values())


def foreign_quantity_node(addr, *, alias, content, foreign_quantity_wrapper,
                          parent_node, foreign_quantity_targets=None, **_kwargs):
    assert not content
    node = ForeignQuantityTarget(addr, alias, foreign_quantity_wrapper())
    parent_node.post_references.append(node)
    if foreign_quantity_targets is not None:
        foreign_quantity_targets.add(node)
    return node.foreign_quantity


class ForeignQuantityTarget:
    def __init__(self, target_addr, alias, obj):
        self.e, *remainder_addr = target_addr.split(".")
        self.target_addr = tuple(splitoff_index(_t) for _t in remainder_addr)
        self.alias = alias
        self.parent = None
        self.foreign_quantity = obj

    def unref(self, root):
        self.foreign_quantity.target = getattr_recursive_idx(
            root[self.e], self.target_addr)


class ElementNode(Node):

    def __init__(self, type_, *, namespace, content, **kwargs):
        type_args = content.pop("__type_args__", None)
        type_addr = tuple(splitoff_index(_t) for _t in type_.split("."))
        build_class = getattr_recursive_idx(namespace, type_addr)
        if type_args:
            build_class = build_class(**type_args)
        super().__init__(namespace=namespace, content=content,
                         build_class=build_class,
                         **kwargs)

    def build(self, **kwargs):
        kwargs.update(self.content)
        e = self.build_class(**kwargs)
        return e


def splitoff_index(s):
    idx = None
    if s[-1] == ']':
        [s, idx] = s[:-1].split('[')
    return s, idx and parse_slc(idx)


def parse_slc(s):
    parts = s.split(":")
    if len(parts) == 1:
        return int(s)
    return slice(*(int(p) if p else None for p in parts))


class ReferenceNode(Node):

    def __init__(self, target, *, content, **kwargs):
        super().__init__(content={}, build_class=None, **kwargs)
        assert not content
        target_addr = target.split(".")
        self.target_addr = tuple(splitoff_index(part) for part in target_addr)

    def partial_target(self, root, sub_addr=(), origin=None):
        if self is origin:
            str_addr = (base + ("[{}]".format(idx) if idx else "")
                        for base, idx in origin.target_addr)
            raise ValueError(
                "Circular reference: {}".format(".".join(str_addr)))
        addr = iter(self.target_addr + tuple(sub_addr))
        root_addr, idx = next(addr)
        target = root[root_addr]
        if idx is not None:
            target = target[idx]
        if isinstance(target, ReferenceNode):
            return target.partial_target(root, sub_addr=addr, origin=origin or self)
        for a_base, idx in addr:
            new_target = target.content.get(a_base)
            if idx is not None:
                new_target = new_target[idx]
            if not isinstance(new_target, Node):
                return target, ((a_base, idx),) + tuple(addr)
            target = new_target
            if isinstance(target, ReferenceNode):
                return target.partial_target(root, sub_addr=addr, origin=origin or self)
        return target, ()

    def unref(self, root):
        target, remainder_addr = self.partial_target(root)
        if target is self:
            raise ValueError(
                "Circular reference. Involved node: {}".format(".".join(self.target_addr)))
        [[key, (parent, alias, [])]] = self.unpack_destinations.items()
        if isinstance(target, Node):
            target.unpack_destinations[key] = (parent, alias, remainder_addr)
            target.parent_nodes |= self.parent_nodes
        if not isinstance(target, Node):
            parent[alias] = getattr_recursive_idx(target, remainder_addr)


def fetch_object(addr, *, namespace, content, **_kwargs):
    assert not content
    idx_addr = tuple(splitoff_index(_t) for _t in addr.split("."))
    return getattr_recursive_idx(namespace, idx_addr)


_keyword_mapping = {None: Node, "__type__": ElementNode, "__reference__": ReferenceNode,
                    "__foreign_quantity__": foreign_quantity_node, "__object__": fetch_object}

_keywords = frozenset(_keyword_mapping)


def extract_type(d):
    keywords = frozenset(d) & _keywords
    if not keywords:
        return (None, ())
    try:
        [keyword] = keywords
        return (keyword, (d.pop(keyword),))
    except ValueError:
        raise ValueError("Node with more than one keyword found: {}".format(
            ", ".join(keywords)))


def wrap_dict(d, nodes=None, **kwargs):
    for key, val in d.items():
        if isinstance(val, dict):
            keyword, args = extract_type(val)
            node = _keyword_mapping[keyword](
                *args, parent=d, alias=key, content=val, nodes=nodes,
                **kwargs)
            if nodes is not None and isinstance(node, Node):
                nodes.add(node)
            d[key] = node
    return d


def wrap_node(node, **kwargs):
    wrap_dict(node.content, parent_node=node, **kwargs)


def model_args_from_dict(root, namespace, inject_name=frozenset(),
                         inject_parent_names=frozenset(), **kwargs):
    nodes = set()
    foreign_quantity_targets = set()

    wrap_dict(root, namespace=namespace, nodes=nodes,
              foreign_quantity_targets=foreign_quantity_targets,
              **kwargs)
    refs = [node for node in nodes if isinstance(node, ReferenceNode)]
    while refs:
        node = refs.pop()
        node.unref(root)
        nodes.remove(node)
    leaves = [node for node in nodes if node.is_leaf()]
    while leaves:
        leaf = leaves.pop()
        name_kwargs = {}
        build_class = getattr(leaf, "build_class", None)
        if build_class and isinstance(build_class, type):
            if any(issubclass(build_class, c) for c in inject_name):
                name_kwargs["name"] = leaf.name
            if any(issubclass(build_class, c) for c in inject_parent_names):
                names = []
                parent = leaf.parent_node
                while parent:
                    names.append(parent.name)
                    parent = parent.parent_node
                name_kwargs["parent_names"] = names[::-1]
        leaf.unpack(**name_kwargs)
        nodes.remove(leaf)
        for node in leaf.parent_nodes:
            if node.is_leaf():
                leaves.append(node)

    if nodes:
        raise ValueError("Circular dependency in nodes.")

    while foreign_quantity_targets:
        foreign_quantity_targets.pop().unref(root)

    return root


def model_from_file(file_name, namespace, **kwargs):
    with open(file_name) as f:
        return model_from_stream(f, namespace, **kwargs)


def model_from_stream(stream, namespace, model_class=None, **kwargs):
    model_dict = json.load(stream)
    model_class = model_class or namespace.Model
    return model_class(**model_args_from_dict(model_dict, namespace, **kwargs))
