import G6 from "@antv/g6"

const ICON_MAP = {
  normal: 'https://gw.alipayobjects.com/mdn/rms_8fd2eb/afts/img/A*0HC-SawWYUoAAAAAAAAAAABkARQnAQ',
  b: 'https://gw.alipayobjects.com/mdn/rms_8fd2eb/afts/img/A*sxK0RJ1UhNkAAAAAAAAAAABkARQnAQ',
};

G6.registerNode(
  'card-node',
  {
    drawShape: function drawShape(cfg, group) {
      const color = cfg.is_linear ? '#F4664A' : '#30BF78';
      const r = 2;
      const shape = group.addShape('rect', {
        attrs: {
          x: 0,
          y: 0,
          width: 150,
          height: 60,
          stroke: color,
          radius: r,
        },
        // must be assigned in G6 3.3 and later versions. it can be any string you want, but should be unique in a custom item type
        name: 'main-box',
        draggable: true,
      });

      group.addShape('rect', {
        attrs: {
          x: 0,
          y: 0,
          width: 150,
          height: 21,
          fill: color,
          radius: [r, r, 0, 0],
        },
        // must be assigned in G6 3.3 and later versions. it can be any string you want, but should be unique in a custom item type
        name: 'title-box',
        draggable: true,
      });

      // left icon
      group.addShape('image', {
        attrs: {
          x: 4,
          y: 2,
          height: 16,
          width: 16,
          cursor: 'pointer',
          img: ICON_MAP[cfg.nodeType || 'app'],
        },
        // must be assigned in G6 3.3 and later versions. it can be any string you want, but should be unique in a custom item type
        name: 'node-icon',
      });

      // title text
      group.addShape('text', {
        attrs: {
          textBaseline: 'top',
          y: 5,
          x: 24,
          lineHeight: 20,
          text: cfg.title,
          fill: '#fff',
        },
        // must be assigned in G6 3.3 and later versions. it can be any string you want, but should be unique in a custom item type
        name: 'title',
      });

      // if (cfg.nodeLevel > 0) {
      //   group.addShape('marker', {
      //     attrs: {
      //       x: 184,
      //       y: 30,
      //       r: 6,
      //       cursor: 'pointer',
      //       symbol: cfg.collapse ? G6.Marker.expand : G6.Marker.collapse,
      //       stroke: '#666',
      //       lineWidth: 1,
      //     },
      //     // must be assigned in G6 3.3 and later versions. it can be any string you want, but should be unique in a custom item type
      //     name: 'collapse-icon',
      //   });
      // }

      // The content list
      cfg.panels.forEach((item, index) => {
        // name text
        group.addShape('text', {
          attrs: {
            textBaseline: 'top',
            y: 27,
            x: 24 + index * 60,
            lineHeight: 20,
            text: item.title,
            fill: 'rgba(0,0,0, 0.4)',
          },
          // must be assigned in G6 3.3 and later versions. it can be any string you want, but should be unique in a custom item type
          name: `index-title-${index}`,
        });

        // value text
        group.addShape('text', {
          attrs: {
            textBaseline: 'top',
            y: 45,
            x: 24 + index * 60,
            lineHeight: 20,
            text: item.value,
            fill: '#595959',
          },
          // must be assigned in G6 3.3 and later versions. it can be any string you want, but should be unique in a custom item type
          name: `index-value-${index}`,
        });
      });
      return shape;
    },
  },
  'single-node',
);


export const graph_config = {
    container: 'graphContainer', // String | HTMLElement，必须，在 Step 1 中创建的容器 id 或容器本身
    width: window.innerWidth, // Number，必须，图的宽度
    height: window.innerHeight, // Number，必须，图的高度
    defaultEdge: {
        // type: 'line',
        type: 'polyline',
        // type: 'quadratic',
        sourceAnchor: 1,
        // // 该边连入 target 点的第 0 个 anchorPoint，
        targetAnchor: 0,
        style: {

            endArrow: {
                path: G6.Arrow.triangle(5, 10), // 使用内置箭头路径函数，参数为箭头的 宽度、长度、偏移量（默认为 0，与 d 对应）
                fill: "#aaaaaa",
                opacity: 50,
            },
            stroke: "#000000",
        },
    },
    defaultNode: {
        // ... 其他属性
        // type: 'card-node',
        type: 'modelRect',
        // type: 'rect',
        // hight: 200,
        size: [200, 60], // 设置节点的默认宽度和高度
        anchorPoints: [
            [0.5, 0],
            [0.5, 1]
        ],
      //   anchorPoints: [
      //     [0, 0.5],
      //     [1, 0.5],
      // ],
        logoIcon: {
          show: false,
        },
        stateIcon: {
          show: false,
          img:
            'https://gw.alipayobjects.com/zos/basement_prod/c781088a-c635-452a-940c-0173663456d4.svg',
        },
        
        // style: {
        //     radius: 5,
        //     // fill: '#C6E5FF',
        //     // stroke: '#5B8FF9',
        // },
        labelCfg:{
          offset: 15,
          style: {
            fill: '#000000',
            fontSize: 20,
            stroke: '#E7E7E7',
          }
        },
        descriptionCfg: {
          style: {
            fill: '#656565',
            fontSize: 14,
          },
        },
        
    },
    // fitView: true,
    // plugins: [minimap], // 将 minimap 实例配置到图上
    modes: {
        // default: ['drag-canvas', 'zoom-canvas', 'drag-node', 'lasso-select'], // 允许拖拽画布、放缩画布、拖拽节点
        default: ['drag-canvas', 'zoom-canvas',  'lasso-select'], // 允许拖拽画布、放缩画布、拖拽节点
    },
    layout: {
      type: 'dagre',
      // rankdir: 'LR', // The center of the graph by default
      // align: 'UR',
      nodesep: 10,
      ranksep: 20,
      controlPoints: true,
    },
}