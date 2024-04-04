from graph.module import Module,Node

def test_model():
    # Create some nodes
    node1 = Node("node1", [], {"op": "Const", "value": 1})
    node2 = Node("node2", [], {"op": "Const", "value": 2})
    node3 = Node("node3", ["node1", "node2"], {"op": "Add"})
    node4 = Node("node4", ["node3"], {"op": "Mul", "value": 3})

    # Create a model
    model = Module([node3,node1, node2, node4])

    # Print the graph
    model.print_graph()