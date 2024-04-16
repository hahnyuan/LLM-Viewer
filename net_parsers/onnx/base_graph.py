import copy
import math
import numpy
import onnx
from collections import defaultdict

from .node import create_node
from .tensor import STATIC_TENSOR, DYNAMIC_TENSOR
from .tensor import get_attribute_data, Tensor, volume
from .log import Logger


# TODO(qiang.lu)optimize to : input_num, output_num
_SHAPE_TENSORS = {
    'Reshape': ('1of2',),
    'Resize': ('2of3', '3of4', '1of2'),
    'Upsample': ('2of3', '3of4', '1of2'),
    'Expand': ('1of2',),
    'Slice': ('1,2of3', '1,2,3of4', '1,2,3,4of5'),
    'ConstantOfShape': ('0of1',),
    'Tile': ('1of2',),
    'Range': ('0,1,2of3',),
    'OneHot': ('1of3',),
    'TopK': ('1of2',),
    'Pad': ('1of2', '1of3',),
    'NonMaxSuppression': ('2of5', '2of4'),
    'Split': ('1of2',),
    'Unsqueeze': ('1of2',),
    'Squeeze': ('1of2',),
    'ReduceSum': ('1of2',),
    'ReduceMean': ('1of2',)
}


def _contains_shape_tensor(n):
    "get node's input_tensors name"
    nodeset = _SHAPE_TENSORS.keys()
    shape_tensors = []
    if n.op_type in nodeset:
        tensor_descs = _SHAPE_TENSORS[n.op_type]
        for desc in tensor_descs:
            strs = desc.split('of')
            indice = strs[0]
            count = int(strs[1])
            if len(n.input) == count:
                indistr = indice.split(',')
                for istr in indistr:
                    shape_tensors.append(n.input[int(istr)])
    return shape_tensors


class OnnxParseConfig():
    def __init__(self, mcfg={}):
        self.cfg = mcfg
        self.constant_folding = mcfg.get('constant_folding',False)
        self.node_rename = mcfg.get('node_rename',False)  # rename by op_type


class BaseGraph():
    def __init__(self, g: onnx.GraphProto, mcfg: OnnxParseConfig):
        self.rawgraph = g
        self.cfg = mcfg

        self.nodemap = {}  # key: name, value: Node instance
        self.tensormap = {}  # key:name, value: tensor，包括：权重、输入、输出
        # 每个tensor是哪些节点（通常只有1个）的输出，用于构建节点的prevnodes
        self.producedby = defaultdict(list)  # key: output_name, value: List(node_name)
        # 每个tensor是哪些节点的输入，用于构建节点的nextnodes
        self.consumedby = defaultdict(list)  # key: tensor_name(without model output), value: List(node_name)
        self.initials = []  # initializer中的name
        self.dynamics = []
        self.input = []  # 模型的输入
        self.output = []  # 模型的输出
        self.valid_shape = False
        self.valid_profile = False
        self.logger = Logger(self.__class__.__name__, level='warning', fmt='full')

        if g is not None:
            self.__init_graph_from_onnxproto__(g, self.cfg.node_rename)
            self.__constant_search__(self.cfg.constant_folding)
            self.__update_nodes_tensors__(self.cfg.constant_folding)
            self.__find_shape_tensors__()

        self.graph_profile_info = {
            "OPs":0,
            "n_load_weight": 0,
            "n_load_act": 0,
            "n_store_act": 0,
        }

    def __init_graph_from_onnxproto__(self, g: onnx.GraphProto, do_node_rename):
        self.node_count = 0
        from .utils import timer

        tm = timer()
        tm.start()
        # -----构建self.nodemap、self.consumedby、self.producedby、节点的prevnodes以及input/output属性
        for node in g.node:
            newnode = create_node(node)
            if do_node_rename or len(newnode.name) == 0:
                newnode.name = newnode.op_type + '_' + str(self.node_count)
            self.node_count += 1
            
            newnode.input = node.input
            newnode.output = node.output
            self.__get_single_node_consumer_producer_info__(newnode.name, newnode)
            self.nodemap[newnode.name] = newnode
        self.__update_prev_nextnodes__()
        self.logger.info(f'Node Init Time Elapsed {tm.stop()}')

        tm.start()
        # 构建模型参数的tensor信息，放tensormap里，init initials first
        if len(g.initializer) == 0:
            self.logger.error(f'no initializer in the onnx')
            raise ValueError(f'no initializer in the onnx')
        for initial in g.initializer:
            tensor = Tensor(initial)
            self.tensormap[initial.name] = tensor
        for input in g.input:
            if input.name not in self.tensormap.keys():
                # print(f'not int tensormap, input.name: {input.name}')
                self.tensormap[input.name] = Tensor(input)

        for output in g.output:
            if output.name not in self.tensormap.keys():
                # print(f'not int tensormap, output.name: {output.name}')
                self.tensormap[output.name] = Tensor(output)

        # update tensormap(init dynamic tensor info)
        for valinfo in g.value_info:
            if valinfo.name not in self.tensormap.keys():
                # print(f'not int tensormap, valinfo.name: {valinfo.name}')
                tensor = Tensor(valinfo)
                self.tensormap[valinfo.name] = tensor
        self.logger.info(f'Tensor Init Time Elapsed {tm.stop()}')

        tm.start()

        # get prevnodes/nextnodes, also update tensormap(dynamic tensor info)
        for node_name in self.nodemap.keys():
            node = self.nodemap[node_name]
            for input_name in node.input:
                if input_name not in self.tensormap.keys():
                    self.tensormap[input_name] = Tensor(input_name)
            for output_name in node.output:
                if output_name not in self.tensormap.keys():
                    self.tensormap[output_name] = Tensor(output_name)
        self.logger.info(f'IO Tensor Init Time Elapsed {tm.stop()}')

    def __update_nodes_tensors__(self, constant_folding):
        from .utils import timer
        tm = timer()
        if constant_folding:
            rmlist = []
            for key in self.nodemap:
                if self.nodemap[key].constant:
                    rmlist.append(key)
            self.logger.info(f'GraphProto Nodes Count:{len(self.rawgraph.node)}')
            self.logger.info(f'Contant folding {rmlist[:3]}... {len(rmlist)} Nodes')
            for key in rmlist:
                self.nodemap.pop(key)

        if constant_folding:
            for name in self.nodemap.keys():
                for tname in self.nodemap[name].input:
                    if self.tensormap[tname].type == STATIC_TENSOR:
                        if tname not in self.initials:
                            self.initials.append(tname)
                    if self.tensormap[tname].type == DYNAMIC_TENSOR:
                        if tname not in self.dynamics:
                            self.dynamics.append(tname)
                for tname in self.nodemap[name].output:
                    if tname not in self.dynamics:
                        self.dynamics.append(tname)
        else:
            for initial in self.rawgraph.initializer:
                self.initials.append(initial.name)
            self.dynamics.extend(self.input)
            for name in self.nodemap.keys():
                for tname in self.nodemap[name].input:
                    if tname not in self.initials and tname not in self.dynamics:
                        self.dynamics.append(tname)
                for tname in self.nodemap[name].output:
                    if tname not in self.dynamics:
                        self.dynamics.append(tname)

        valid_tensors = []
        valid_tensors.extend(self.initials)
        valid_tensors.extend(self.dynamics)
        rm_list = []
        for tname in self.tensormap.keys():
            if tname not in valid_tensors:
                rm_list.append(tname)
        for tname in rm_list:
            self.tensormap.pop(tname)

        self.input = []
        self.output = []
        for name in self.nodemap.keys():
            node = self.nodemap[name]
            for tensor in node.input:
                if tensor not in self.producedby and tensor in self.dynamics:
                    if tensor not in self.input:
                        self.input.append(tensor)
            for tensor in node.output:
                if tensor not in self.consumedby or len(self.consumedby[tensor]) == 0:
                    if tensor not in self.output:
                        self.output.append(tensor)
        self.__update_consumer_producer__()
        self.__update_prev_nextnodes__()
        self.logger.info(f'Update Nodes Tensors  Time Elapsed {tm.stop()}')

    def __is_node_constant__(self, node):
        constant_node = True
        if node.op_type in ['DequantizeLinear']:
            return False
        for tname in node.input:
            if self.tensormap[tname].type == DYNAMIC_TENSOR:
                constant_node = False
                break
        return constant_node

    def __constant_search__(self, constant_folding):
        from .utils import timer
        tm = timer()
        for name in self.nodemap.keys():
            node = self.nodemap[name]
            if hasattr(node, 'constant'):
                continue
            constant_node = self.__is_node_constant__(node)
            node.constant = constant_node
            if constant_node:
                search_nodes = [name]
                while len(search_nodes) > 0:
                    this_node = self.nodemap[search_nodes[0]]
                    search_nodes.pop(0)
                    if constant_folding:
                        itensors = []
                        for input in this_node.input:
                            itensors.append(self.tensormap[input])
                            if self.tensormap[input].numpy is None and input != '':
                                self.logger.warning(f'Tensor {input} has shape only, {name} may has wrong value infer result')
                        otensors = []
                        for output in this_node.output:
                            otensors.append(self.tensormap[output])
                        this_node.value_infer(itensors, otensors)
                        if len(otensors) > 0:
                            for i, output in enumerate(this_node.output):
                                self.tensormap[output].type = STATIC_TENSOR
                    else:
                        for i, output in enumerate(this_node.output):
                            self.tensormap[output].type = STATIC_TENSOR
                    for output in this_node.output:
                        if output in self.consumedby:
                            for consumer in self.consumedby[output]:
                                cnode = self.nodemap[consumer]
                                if self.__is_node_constant__(cnode):
                                    cnode.constant = True
                                    search_nodes.append(consumer)
        self.logger.info(f'Constant Search Time Elapsed {tm.stop()}')

    def __update_prev_nextnodes__(self):
        for node_name in self.nodemap.keys():
            node = self.nodemap[node_name]
            self.nodemap[node_name].prevnodes = []
            self.nodemap[node_name].nextnodes = []
            for input_name in node.input:
                if input_name in self.producedby:
                    for producer in self.producedby[input_name]:
                        self.nodemap[node_name].prevnodes.append(self.nodemap[producer])
            for output_name in node.output:
                if output_name in self.consumedby:
                    for consumer in self.consumedby[output_name]:
                        self.nodemap[node_name].nextnodes.append(self.nodemap[consumer])

    def __update_consumer_producer__(self):
        self.producedby = defaultdict(list)
        self.consumedby = defaultdict(list)
        for name in self.nodemap:
            node = self.nodemap[name]
            self.__get_single_node_consumer_producer_info__(name, node)

    def __get_single_node_consumer_producer_info__(self, node_name, node):
        for input_name in node.input:
            if node_name not in self.consumedby[input_name]:
                self.consumedby[input_name].append(node_name)
        for output_name in node.output:
            if node_name not in self.producedby[output_name]:
                self.producedby[output_name].append(node_name)

    def update_tensor_relations(self):
        self.__update_consumer_producer__()
        self.__update_prev_nextnodes__()
        self.dynamics = list(self.producedby.keys())

    def __find_shape_tensors__(self):
        self.shape_tensors = []
        for n in self.nodemap.keys():
            shape_tensors = _contains_shape_tensor(self.nodemap[n])
            for st in shape_tensors:
                self.shape_tensors.append(st)
        self.shape_tensors = set(self.shape_tensors)
        # print(self.shape_tensors)
        for tensor in self.shape_tensors:
            if tensor not in self.initials and tensor in self.producedby.keys():
                searchnodes = self.producedby[tensor]
                while len(searchnodes) > 0:
                    nextnodes = []
                    for nname in searchnodes:
                        node = self.nodemap[nname]
                        node.shape_calc = True
                        if node.op_type == 'Shape':
                            continue
                        for input in node.input:
                            if input not in self.initials and input in self.producedby.keys():
                                producers = self.producedby[input]
                                nextnodes.extend([p for p in producers if self.nodemap[p].shape_calc == False])
                    searchnodes = nextnodes

    def get_initials_from_nodenames(self, nodenames):
        initializer = []
        enqueued = []
        for name in nodenames:
            for input in self.nodemap[name].input:
                if input in self.initials and input not in enqueued:
                    proto = self.tensormap[input].make_tensor_proto()
                    if proto is not None:
                        initializer.append(proto)
                    enqueued.append(input)
        return initializer

    def save_model(self, f: str, shape_only: bool = False, no_shape: bool = False, rawmodel: onnx.ModelProto = None):
        if len(self.nodemap.keys()) == 0:
            self.logger.warning(f'Empty graph {f} to save')
            return
        graph = self.make_graph_onnx(self.nodemap.keys(), 'graph', self.input, self.output,
                                     with_initializer=not shape_only, with_shape_info=not no_shape)
        if graph is not None and f is not None:
            attr = {}
            model = onnx.helper.make_model(graph, **attr)
            if rawmodel is not None:
                model.ir_version = rawmodel.ir_version
                model.opset_import.pop()
                for opset in rawmodel.opset_import:
                    model.opset_import.append(opset)
            onnx.save_model(model, f)

    def make_graph_onnx(self, nodenames, gname, inputnames, outputnames, with_initializer=True, with_shape_info=True):
        nodes = []
        for name in nodenames:
            nodes.append(self.nodemap[name].make_nodeproto())

        initializer = None
        if with_initializer:
            initializer = self.get_initials_from_nodenames(nodenames)

        inputs = []
        outputs = []
        for name in inputnames:
            if name in self.tensormap:
                proto = self.tensormap[name].make_value_proto(make_dummy=True)
                if proto is not None:
                    inputs.append(proto)
            else:
                inputs.append(onnx.helper.make_tensor_value_info(name, 1, None))
        for name in outputnames:
            if name in self.tensormap:
                proto = self.tensormap[name].make_value_proto(make_dummy=True)
                if proto is not None:
                    outputs.append(proto)
        value_infos = []
        if with_shape_info:
            for key in self.dynamics:
                if key in self.input or key in self.output:
                    continue
                tensor = self.tensormap[key]
                vinfo = tensor.make_value_proto()
                if vinfo is None:
                    continue
                value_infos.append(vinfo)
            if not with_initializer:
                for key in self.initials:
                    tensor = self.tensormap[key]
                    vinfo = tensor.make_value_proto()
                    if vinfo is None:
                        continue
                    value_infos.append(vinfo)
        graph = onnx.helper.make_graph(nodes=nodes, name=gname, inputs=inputs, outputs=outputs, initializer=initializer,
                                       value_info=value_infos)
        return graph

    def topsort_nodes(self, node_names, input_names):
        # update
        produced_by = {}
        for name in node_names:
            node = self.nodemap[name]
            for tname in node.output:
                produced_by[tname] = name

        consumed_by = {}
        dependencies = {}
        for name in node_names:
            node = self.nodemap[name]
            count = 0
            for tname in node.input:
                if tname in produced_by.keys():
                    count += 1
                    if tname in consumed_by.keys():
                        consumed_by[tname].append(name)
                    else:
                        consumed_by[tname] = [name]
            dependencies[name] = count

        ordered_nodes = []
        queue = []
        while True:
            for name in node_names:
                if dependencies[name] == 0:
                    queue.append(name)
                    ordered_nodes.append(name)
                    dependencies[name] -= 1
            if len(queue) == 0:
                break
            for name in queue:
                node = self.nodemap[name]
                for o in node.output:
                    if o in consumed_by.keys():
                        for con in consumed_by[o]:
                            dependencies[con] -= 1
            queue.clear()
        return ordered_nodes

    def graph_reorder_nodes(self):
        ordered_nodes = self.topsort_nodes(self.nodemap.keys(), self.input)
        new_map = {}
        for nname in ordered_nodes:
            new_map[nname] = self.nodemap[nname]
        self.nodemap = new_map
        self.update_tensor_relations()

    def update_input_by_map(self, inputs: {}):
        for key in inputs.keys():
            if key in self.tensormap.keys():
                self.tensormap[key].update_tensor(inputs[key])

    def check_inputs(self):
        for name in self.input:
            shape = self.tensormap[name].shape
            for val in shape:
                if isinstance(val, str):
                    return False, name
                if val < 0:
                    return False, name
        return True, None

    def shape_infer(self, inputs: {} = None):
        self.valid_shape = False
        if inputs is not None:
            self.update_input_by_map(inputs)
        in_valid, tname = self.check_inputs()
        if not in_valid:
            raise ValueError(
                f"The input tensor {tname}'s shape {self.tensormap[tname].shape2str()} is not valid, Please set it to a valid shape.")
        self.shapeinfer_optime_map = {}
        from .utils import timer
        tm = timer()
        for key in self.nodemap.keys():
            tm.start()
            node = self.nodemap[key]
            itensors = []
            for input in node.input:
                itensors.append(self.tensormap[input])
            otensors = []
            for output in node.output:
                otensors.append(self.tensormap[output])

            if node.shape_calc:
                node.value_infer(itensors, otensors)
            else:
                node.shape_infer(itensors, otensors)

            if node.op_type in self.shapeinfer_optime_map.keys():
                self.shapeinfer_optime_map[node.op_type] += tm.stop()
            else:
                self.shapeinfer_optime_map[node.op_type] = tm.stop()
        self.logger.info(self.shapeinfer_optime_map)
        self.valid_shape = True

    def value_infer(self, inputs: {}):
        self.update_input_by_map(inputs)
        for key in self.nodemap.keys():
            node = self.nodemap[key]
            itensors = []
            for input in node.input:
                itensors.append(self.tensormap[input])
            otensors = []
            for output in node.output:
                otensors.append(self.tensormap[output])
            node.value_infer(itensors, otensors)
        outputs = []
        for output in self.output:
            outputs.append(self.tensormap[output].numpy)
        return outputs
