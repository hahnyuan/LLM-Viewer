export const data = {
    nodes: [
      {
        title: 'LayerNorm',
        is_linear: false,
        nodeType: 'normal',
        id: 'node1',
        nodeLevel: 2,
        panels: [
          { title: 'OPs', value: '59M' },
          { title: 'Access', value: '34MB' },
        ],
        // panels: ['OPs: 59M', 'Access: 34MB', 'Max Performance: 1T'],
        x: 100,
        y: 100,
      },
      {
        title: 'q_proj',
        is_linear: true,
        nodeType: 'normal',
        id: 'node2',
        nodeLevel: 0,
        panels: [
          { title: 'OPs', value: '59M' },
          { title: 'Access', value: '34MB' },
        ],
        // panels: ['OPs: 69G', 'Access: 67M', 'Max Performance: 166T'],

      },
      {
        title: 'k_proj',
        is_linear: true,
        nodeType: 'normal',
        id: 'node3',
        nodeLevel: 3,
        panels: [
          { title: 'OPs', value: '59M' },
          { title: 'Access', value: '34MB' },
        ],
        // panels: ['OPs: 69G', 'Access: 67M', 'Max Performance: 166T'],
        collapse: true,
      },
      {
        title: 'qk_matmul',
        is_linear: false,
        nodeType: 'normal',
        id: 'node4',
        nodeLevel: 3,
        panels: [
          { title: 'OPs', value: '59M' },
          { title: 'Access', value: '34MB' },
        ],
        // panels: ['OPs: 34G', 'Access: 302M', 'Max Performance: 87T'],
        collapse: true,
      },
    ],
    edges: [
          {
              source: 'node1', // String，必须，起始点 id
              target: 'node2', // String，必须，目标点 id
          },
          {
              source: 'node1', // String，必须，起始点 id
              target: 'node3', // String，必须，目标点 id
          },
          {
              source: 'node2', // String，必须，起始点 id
              target: 'node4', // String，必须，目标点 id
          },
          {
              source: 'node3', // String，必须，起始点 id
              target: 'node4', // String，必须，目标点 id
          },
      ],
  };