<template>
    <div class="main_graph" ref="graphContainer">
        <div id="graphContainer" @resize="handleResize"></div>

        <div class="float-search-window">
            <input type="text" v-model.lazy="searchText" placeholder="Search" />
            <div>
                <div v-for="(value) in searchResult" @click="SelectNode(value, true)">
                    {{ value }}
                </div>
            </div>
        </div>
        <div class="float-node-info-window">
            <div v-for="(value, key) in selectedNodeInfo" :key="key">
                {{ key }}: <br />{{ value }}<br /><br />
            </div>
        </div>
    </div>
</template>

<script setup>
import G6 from "@antv/g6"
import { onMounted, onBeforeUpdate, provide } from 'vue'
import { watch, inject, ref } from 'vue'
import { graph_config } from "./graphs/graph_config.js"
import axios from 'axios'

const model_id = inject('model_id')
const hardware = inject('hardware')
const graphUpdateTrigger = inject('graphUpdateTrigger')
const InferenceConfig = inject('InferenceConfig')
const ip_port = "127.0.0.1:5000"

var graph = null;
var graph_data;
var all_node_info = {}

const searchText = ref('')
var searchResult = []

var selectedNodeInfo = ref({"a":"b"})
var selected_node_id = ""

const changeGraphSizeWaitTimer = ref(false);
window.onresize = () => {
    if (!changeGraphSizeWaitTimer.value & graph != null) {
        // console.log("handleResize", window.innerWidth, window.innerHeight)
        var leftControlDiv = document.querySelector('.left_control');
        var width = leftControlDiv.offsetWidth;
        graph.changeSize(window.innerWidth - width, window.innerHeight)
        changeGraphSizeWaitTimer.value = true;
        setTimeout(function () {
            changeGraphSizeWaitTimer.value = false;
        }, 100);
    }
};

function graphUpdate(is_fit_view = false, is_init = false) {
    const url = 'http://' + ip_port + '/get_graph'
    console.log("graphUpdate", url)
    axios.post(url, { model_id: model_id.value, hardware: hardware.value, inference_config: InferenceConfig.value }).then(function (response) {
        console.log(response);
        graph_data = response.data
        all_node_info=graph_data.nodes
        if (is_init) {
            graph.changeData(graph_data)
        } else {
            // iterate each node
            graph_data.nodes.forEach(function (node) {
                // update the node
                graph.updateItem(node.id, {
                    description: node.description,
                });
            });

        }
        console.log(graph_data)
        // graph.render();
        // graph.refresh()
        selectedNodeInfo.value = {}
        selected_node_id = ""
        nowFocusNode = null
        if (is_fit_view) {
            // graph.render();
            setTimeout(() => {
                graph.fitView();
            }, 10);
        }

    })
        .catch(function (error) {
            console.log("error in graphUpdate");
            console.log(error);
        });

}

watch(() => graphUpdateTrigger.value, () => graphUpdate(false))

function handleSearch(newText, oldText) {
    console.log("handleSearch", newText)
    const nodes = graph.findAll('node', (node) => {
        const nodeId = node.get('id');
        // console.log("handleSearch", node)
        return nodeId.includes(newText)
    });
    console.log("handleSearch", nodes)
    searchResult.length = 0
    for (let i = 0; i < nodes.length; i++) {
        const node = nodes[i];
        const nodeId = node.get('id');
        searchResult.push(nodeId)
        if (i > 100) {
            break
        }

    }
}
watch(searchText, handleSearch)

var nowFocusNode = null
var nowFocusNodePrevColor = null
function SelectNode(nodeId, moveView = false) {
    if (moveView) {
        graph.focusItem(nodeId, true)
    }
    if (nowFocusNode) {
        // console.log("nowFocusNodePrevColor", nowFocusNodePrevColor)
        nowFocusNode.update({
            style: {
                fill: nowFocusNodePrevColor,
            },
        });
    }
    const node = graph.findById(nodeId)
    if (node) {
        // 高亮
        if (node.getModel().style.fill) {
            nowFocusNodePrevColor = node.getModel().style.fill
        } else {
            nowFocusNodePrevColor = "#ffffff"
        }
        node.update({
            style: {
                fill: "#dffdff",
            },
        });
        nowFocusNode = node
    }
    
    // for (node_info in all_node_info) {
    for (let node_info of Object.values(all_node_info)) {
        console.log(node_info,nodeId)
        if (node_info['id']==nodeId) {
            
            selectedNodeInfo.value = node_info["info"]
        }
    }
    // selectedNodeInfo.value = all_node_info[nodeId]
    // axios.post(url, { node_id: nodeId }).then(function (response) {
    //     console.log(response);
    //     // 将返回回来的json数据转换成node_info
    //     const nodeInfo = response.data
    //     console.log(url, nodeInfo)
    //     // 将nodeInfo传递给祖先组件
    //     node.nodeInfo = nodeInfo
    //     selectedNodeInfo.value = nodeInfo
    //     selectedNodeId.value = nodeId

    // })
    //     .catch(function (error) {
    //         console.log(error);
    //         alert("失败", url)
    //     });
}

onMounted(() => {
    graph = new G6.Graph(graph_config);  // 创建.
    graph.on('node:click', (event) => {
        const { item } = event;
        const node = item.getModel();
        clickNode(node);
    });
    graphUpdate(true, true);
    graph.render();
})

function clickNode(node) {
    console.log(node);
    const nodeId = node.id;
    SelectNode(nodeId)
}
</script>

<style scoped>
.main_graph {
    width: 75%;
    height: 100%;

    position: relative;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    float: right;
    /* width: 85%; */
    flex-grow: 1;
    background-color: #ffffff;
    border: 0px;
}

.float-search-window {
    position: absolute;
    top: 10px;
    right: 10px;
    height: auto;
    max-height: 50vh;
    background-color: #f1f1f1b7;
    padding: 3px;
    overflow-y: auto;
}

.float-node-info-window {
    position: absolute;
    top: 10px;
    left: 10px;
    background-color: #f1f1f1b7;
}
</style>