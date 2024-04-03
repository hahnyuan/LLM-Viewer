from graph.model import Model,Node
from analyzers.llm_analyzer import LLMAnalyzer
from hardwares.roofline_model import RooflineModel

def test_dummy_llm():
    # Create some nodes
    node1 = Node("node1", [], {"op": "Const", "value": 1})
    node2 = Node("node2", [], {"op": "Const", "value": 2})
    node3 = Node("node3", ["node1", "node2"], {"op": "Add"})
    node4 = Node("node4", ["node3"], {"op": "Mul", "value": 3})

    # Create a model
    model = Model([node3,node1, node2, node4])

    # Print the graph
    hardware_model=RooflineModel(1,1)
    analyzer=LLMAnalyzer(model,hardware_model)
    rst=analyzer.analyze(16,1)
    print(rst)