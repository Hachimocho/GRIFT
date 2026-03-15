import inspect


def _iter_classes(mod, package_prefix):
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        if getattr(obj, "__module__", "").startswith(package_prefix):
            yield name, obj


def test_nodes_contracts():
    import nodes as nodes_pkg
    classes = list(_iter_classes(nodes_pkg, "nodes."))
    assert classes, "No node classes discovered"
    for name, cls in classes:
        assert hasattr(cls, "tags"), f"{name} missing tags"
        assert hasattr(cls, "hyperparameters"), f"{name} missing hyperparameters"


def test_edges_contracts():
    import edges as edges_pkg
    classes = list(_iter_classes(edges_pkg, "edges."))
    assert classes, "No edge classes discovered"
    for name, cls in classes:
        assert hasattr(cls, "tags"), f"{name} missing tags"
        assert hasattr(cls, "hyperparameters"), f"{name} missing hyperparameters"


def test_traversals_contracts():
    import traversals as trav_pkg
    from traversals.Traversal import Traversal
    classes = [(n, c) for n, c in _iter_classes(trav_pkg, "traversals.") if issubclass(c, Traversal)]
    assert classes, "No traversal classes discovered"
    for name, cls in classes:
        # Ensure required instance methods exist (may still rely on base __len__)
        assert callable(getattr(cls, "traverse", None)), f"{name} missing traverse()"
        assert callable(getattr(cls, "get_pointers", None)), f"{name} missing get_pointers()"
        assert callable(getattr(cls, "reset_pointers", None)), f"{name} missing reset_pointers()"
