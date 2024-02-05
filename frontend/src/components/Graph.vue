<template>
    <div class="main_graph" ref="graphContainer">
        <div id="graphContainer" @resize="handleResize"></div>

        <div class="float-search-window">
            <!-- 浮窗内容 -->
            <input type="text" v-model.lazy="searchText" placeholder="Search" />
            <div>
                <!-- 搜索结果 -->
                <div v-for="(value) in searchResult" @click="SelectNode(value, true)">
                    {{ value }}
                </div>
            </div>
        </div>
        <div class="float-node-info-window">
            <!-- 浮窗内容 -->
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
import {data} from "./graphs/default_data.js"


var graph = null;
var graph_data;
var settingsData = inject('settingsData')
var selectedNodeInfo = inject('selectedNodeInfo')
var selectedNodeId = inject('selectedNodeId')
var graphUpdateTrigger = inject('graphUpdateTrigger')
const step = inject('step')
const searchText = ref('')
var searchResult = []
const selectedNodeListRef = inject('selectedNodeListRef');  // 用于统计哪些元素被选择了.

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

function graphUpdate(is_fit_view = false) {
    const ip_port = "127.0.0.1:5000"
    console.log("graphUpdate", settingsData)
    const url = 'http://' + ip_port + '/'
    axios.get(url).then(function (response) {
        // 这是异步的
        console.log(response);
        // 将返回回来的json数据转换成graph_data
        graph_data = response.data
        graph.changeData(graph_data)
        console.log(graph_data)
        // graph.render();
        // graph.refresh()
        selectedNodeInfo.value = {}
        selectedNodeId.value = ""
        nowFocusNode = null
        if (is_fit_view) {
            graph.render();
            setTimeout(() => {
                graph.fitView();
            }, 10);
        }

    })
        .catch(function (error) {
            console.log(error);
            alert("失败", url)
        });

}

graphUpdate(true);

watch(() => step.value, () => {
    graphUpdate(true);

})
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
    // 移动画面到选中节点
    if (moveView) {
        graph.focusItem(nodeId, true)
    }
    if (nowFocusNode) {
        // 取消高亮
        console.log("nowFocusNodePrevColor", nowFocusNodePrevColor)
        nowFocusNode.update({
            style: {
                stroke: nowFocusNodePrevColor,
            },
        });
    }
    const node = graph.findById(nodeId)
    if (node) {
        // 高亮
        if (node.getModel().style.stroke) {
            nowFocusNodePrevColor = node.getModel().style.stroke
        } else {
            nowFocusNodePrevColor = "#5B8FF9"
        }
        node.update({
            style: {
                stroke: 'red',
            },
        });
        nowFocusNode = node
    }
    // POST请求 http://{ip_port}/{phase}_node_info 获取nodeId的详细信息
    const ip_port = settingsData.value.ip_port
    const url = 'http://' + ip_port + '/' + step.value + '_node_info'
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
    graph.on('nodeselectchange', (e) => {
        for (let index = 0; index < e.selectedItems.nodes.length; index++) {
            selectedNodeListRef.value.push(e.selectedItems.nodes[index]._cfg.id);
        }
        console.log(selectedNodeListRef.value)
    });
    graph.render();
})

function clickNode(node) {
    console.log(node);
    const nodeId = node.id;
    SelectNode(nodeId)
}

function addSelNode(node) {
    console.log(node.id, "has been selected!")
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
    /* border: 1px solid #ccc; */
    /* box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2); */
    padding: 3px;
    overflow-y: auto;
}

.float-node-info-window {
    position: absolute;
    top: 10px;
    left: 10px;
    /* width: 200px;
    height: 200px; */
    background-color: #f1f1f1b7;
    /* border: 1px solid #ccc; */
    /* box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2); */
    /* padding: 10px; */
}
</style>