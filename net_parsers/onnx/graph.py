import onnx
from collections import defaultdict
from prettytable import PrettyTable

from .base_graph import BaseGraph, OnnxParseConfig
from .utils import shapes2str


EXCLUDE_OPS=['Flatten', 'Relu', 'BatchNormalization', 'Shape', "Reshape", "Transpose"]

class Graph(BaseGraph):
    def __init__(self, g: onnx.GraphProto, mcfg: OnnxParseConfig):
        super().__init__(g, mcfg)
        self.graph_profile_info = {
            "OPs":0,
            "n_load_weight": 0,
            "n_load_act": 0,
            "n_store_act": 0,
        }

    def profile(self, exclude_ops=EXCLUDE_OPS):
        self.valid_profile = False
        if not self.valid_shape:
            self.logger.warning('Please perform a valid shape_infer() before profile().')
            return

        params_flag_map = defaultdict(int)  # judge shared_weight
        inputs_flag_map = defaultdict(int)  # judge shared_input
        for key in self.nodemap.keys():
            node = self.nodemap[key]
            for input_name in node.input:
                if input_name in self.initials:
                    params_flag_map[input_name] += 1  # weight
                    # params_flag_map[input_name] = len(self.consumedby[input_name])
                else:
                    inputs_flag_map[input_name] += 1  # activation
                    # inputs_flag_map[input_name] = len(self.consumedby[input_name])

        for name in self.nodemap.keys():
            node = self.nodemap[name]
            itensors = [self.tensormap[input_name] for input_name in node.input]
            otensors = [self.tensormap[output_name] for output_name in node.output]
            node.profile(itensors, otensors, params_flag_map, inputs_flag_map)

        # profile whole graph, consider shared_weight and shared_layer
        for name in self.nodemap.keys():
            node = self.nodemap[name]
            if node.op_type in exclude_ops:
                continue
            self.graph_profile_info["OPs"] += node.profile_info["OPs"]
            self.graph_profile_info["n_load_act"] += node.profile_info["n_load_act"]
            self.graph_profile_info["n_load_weight"] += node.profile_info["n_load_weight"]
            self.graph_profile_info["n_store_act"] += node.profile_info["n_store_act"]
        for name,node in self.nodemap.items():
            if node.op_type in exclude_ops:
                continue
            if name in params_flag_map and params_flag_map[name]>=2:
                self.graph_profile_info["n_load_weight"] -= node.profile_info['n_load_weight'] * (params_flag_map[name] - 1)
            if name in inputs_flag_map and inputs_flag_map[name]>=2:
                self.graph_profile_info["n_load_act"] -= node.profile_info['n_load_act'] * (inputs_flag_map[name] - 1)

        self.valid_profile = True

    def print_profile_info(self, save_path: str = None, exclude_ops=EXCLUDE_OPS):
        from utils import str_number_1024

        if not self.valid_profile:
            self.logger.warning('Please perform a valid profile() before print_profile_info().')
            return

        table_rows = []

        ops = int(round(self.graph_profile_info['OPs']))
        params = int(self.graph_profile_info['n_load_weight'])
        params += 1e-18
        ops += 1e-18
        for key in self.nodemap.keys():
            node = self.nodemap[key]
            if exclude_ops is not None and node.op_type in exclude_ops:
                continue
            row = [key, self.nodemap[key].op_type]
            row.append(str_number_1024(int(node.profile_info['OPs'])))
            row.append('{:.2%}'.format(node.profile_info['OPs'] / ops))
            row.append(str_number_1024(int(node.profile_info['n_load_weight'])))
            row.append('{:.2%}'.format(node.profile_info['n_load_weight'] / params))
            row.append(shapes2str(node.profile_info['input_shape']))
            row.append(shapes2str(node.profile_info['output_shape']))
            table_rows.append(row)
        row = ['Total', '_']
        row.append(str_number_1024(int(ops)))
        row.append('100%')
        row.append(str_number_1024(int(params)))
        row.append('100%')
        row.append('_')
        row.append('_')

        table_rows.append(row)
        headers = ['Name', 'Type', "OPs", "OPs/%", "Params", "Params/%", "InputShapes", "OutShapes"]

        table = PrettyTable()
        table.align = "l"
        table.field_names = headers
        table.add_rows(table_rows)
        if save_path is None:
            print(table)
        else:
            fp = open(f, 'w')
            fp.write(str(table))
            fp.close()

    def print_graph(self):
        for name,tensor in self.nodemap.items():
            print(f'name: {name}, op_type:{tensor.op_type}, input:{tensor.input}, output:{tensor.output}')