// Description: using package:dagre to calculate layout for the whole dag.
function DAGLayout(nodes, edges, transform) {
    const g = new dagre.graphlib.Graph({ compound:true });
    let nodeDic = {};
    console.log(nodes, edges)
    nodes.forEach((node) => {
        nodeDic[node.name] = node;
    });


    // Set an object for the graph label
    g.setGraph({
        rankdir: 'TB',
        nodesep: 50, //vertical
        edgesep: 15, //edge vertical margin
        ranksep: 60, //node horizontal margin
        // ranker: 'longest-path'
    });

    // Default to assigning a new object as a label for each new edge.
    g.setDefaultEdgeLabel(function () {
        return {};
    });

    for (let i = 0, len = nodes.length; i < len; ++i) {
        let node = nodes[i];
        node.x = 0;
        node.y = 0;
        //vertical layout: reverse h and w
        g.setNode(node.name, {
            label: node.name,
            width: node.status == "inner" ? node.w : node.w,
            height: node.h - 6,
        });
    }

    for (let j = 0; j < edges.length; j++) {
        let edge = edges[j];
        g.setEdge(edge.start, edge.end);
    }
    dagre.layout(g);

    for (let i = 0, len = nodes.length; i < len; ++i) {
        let node = nodes[i];
        let graphNode = g.node(node.name);
        [node.x, node.y] = transform(graphNode.x, graphNode.y);
    }

    let edgeDic = {};
    for (let k = 0; k < edges.length; k++) {
        let edge = edges[k];
        edgeDic[edge.start + "->" + edge.end] = edge;
        edge.startNode = nodeDic[edge.start];
        edge.endNode = nodeDic[edge.end];
        if (edge.startNode === undefined || edge.endNode === undefined) {
            console.log("Invalid edge:");
            console.log(edge);
        }
    }

    g.edges().forEach(function (e) {
        let edge = edgeDic[e.v + "->" + e.w];
        edge.points = []
        for (let i = 0; i < g.edge(e).points.length; i++) {
            let point = g.edge(e).points[i];
            //console.log(point, edge)
            [point.x, point.y] = transform(point.x, point.y);
            edge.points[i] = point;
        }
        if (edge.startNode.y > edge.endNode.y) {
            edge.points = edge.points.reverse()
        }
        /*
        if (!edge.points || edge.points.length == 0) return
        edge.points[0].x = edge.startNode.x
        edge.points[0].y = edge.startNode.y + edge.startNode.h / 2
        edge.points[edge.points.length - 1].x = edge.endNode.x
        edge.points[edge.points.length - 1].y = edge.endNode.y - edge.endNode.h / 2
        */
    });
    console.log("DAG done");
}

function calculateSegmentation(nodes, edges, width) {
    console.log("segmentation start");
    //calculate word

    let nodeDic = {};
    nodes.forEach(node => {
        nodeDic[node.name] = node;
        node.in = [];
        node.out = [];
    });
    edges.forEach(edge => {
        nodeDic[edge.start].out.push(nodeDic[edge.end]);
        nodeDic[edge.end].in.push(nodeDic[edge.start]);
    });

    let rankList = [];
    nodes.sort((a, b) => a.x - b.x);
    let rank = -1;
    let pointer = -1;
    nodes.forEach(node => {
        if (Math.abs(node.x - pointer) > 1e-5) {
            rank++;
            node.rank = rank;
            pointer = node.x;
        } else {
            node.rank = rank;
        }
        if (rankList[rank] === undefined) {
            rankList[rank] = [];
        }

        rankList[rank].push(node);
    });

    let c = new Array(rankList.length);
    let p = new Array(rankList.length);
    //c[0] = 0;
    c[-1] = 0;

    for (let j = 0; j < rankList.length; j++) {
        let min = Infinity;
        let index = -1;
        for (let i = 0; i <= j; i++) {
            let temp = c[i - 1] + calculateLC(rankList, i, j, rankList.length, width);
            if (temp < min) {
                min = temp;
                index = i;
            }
        }
        p[j] = index;
        c[j] = min;
    }


    let segList = [];
    let splitIndex = rankList.length - 1;
    while (p[splitIndex] !== undefined && splitIndex >= 0) {
        let seg = [];
        for (let i = p[splitIndex]; i <= splitIndex; i++) {
            seg = seg.concat(rankList[i]);
        }
        segList.push(seg);
        splitIndex = p[splitIndex] - 1;
    }
    segList.reverse();
    segList.forEach((seg, index) => {
        seg.forEach(node => node.seg = index);
    });
    console.log(segList);

    return segList;
}

function calculateLC(rankList, i, j, N, width) {
    let left = Infinity;
    let right = -Infinity;
    let lambda = 1; //1
    //can be accelarate
    for (let k = i; k <= j; k++) {
        rankList[k].forEach(node => {
            if (node.x - node.w / 2 < left) left = node.x - node.w / 2;
            if (node.x + node.w / 2 > right) right = node.x + node.w / 2;
        });
    }
    let length = right - left;
    let extra = (width - length) / width;

    if (j == N && extra > 0) {
        return 0;
    } else if (extra < 0) {
        return Infinity;
    } else {
        let outLinkNum = 0;
        let set = new Set();
        for (let k = i; k <= j; k++) {
            rankList[k].forEach(node => {
                set.add(node.name);
            });
        }
        for (let k = i; k <= j; k++) {
            rankList[k].forEach(node => {
                node.out.forEach(output => {
                    if (!set.has(output.name))
                        outLinkNum++;
                });
            });
        }
        return extra * extra * extra + lambda * outLinkNum;
    }
}

function modifyLayoutBySegmentation(nodes, edges, segList, visconfig) {
    let dic = {};
    for (let i = 0, len = nodes.length; i < len; ++i) {
        let node = nodes[i];
        node.out_edges = [];
        node.in_edges = [];
        dic[node.name] = node;
    }
    let edge_dic = {};
    for (let i = 0, len = edges.length; i < len; ++i) {
        let edge = edges[i];
        edge_dic[`${edge.start}->${edge.end}`] = edge;
    }
    edges.forEach(edge => {
        let node1 = dic[edge.start];
        node1.out_edges.push(`${edge.start}->${edge.end}`);
        let node2 = dic[edge.end];
        node2.in_edges.push(`${edge.start}->${edge.end}`);
    });

    let segPadding = visconfig.segPadding,
    leftPadding = visconfig.leftPadding,
    segTop = segPadding;
    let ranges = [];
    segList.forEach(seg => {
        let range = getSegHeight(seg);
        ranges.push(range);
        seg.forEach(node => {
            node.y = node.y - range.t + segTop;
            node.x = node.x - range.l + leftPadding;
            node.out_edges.forEach(edge_name => {
                edge_dic[edge_name].points.forEach(point => {
                    point.x = point.x - range.l + leftPadding;
                    point.y = point.y - range.t + segTop;
                });
            });
        });
        segTop += (segPadding + range.b - range.t);
    });

    //expand
    if (segList.length > 1) {
        for (let i = 0; i < segList.length; i++) {
            let seg = segList[i];
            let maxX = 0;
            let minX = Infinity;

            for (let j = 0; j < seg.length; j++) {
                maxX = Math.max(maxX, seg[j].x + seg[j].w / 2);
                minX = Math.min(minX, seg[j].x - seg[j].w / 2);
            }

            let scale = d3.scaleLinear()
                .domain([minX, maxX])
                .range([0 + visconfig.leftAndRightSpan,
                    visconfig.w - visconfig.leftAndRightSpan
                ]);

            seg.forEach(function (node) {
                node.x = scale(node.x);
                node.scale = scale;
                node.out_edges.forEach(edge_name => {
                    edge_dic[edge_name].points.forEach(point => {
                        point.x = scale(point.x);
                    });
                });
            })

        }
    }

    //modify svg
    if (segTop - segPadding > visconfig.h) {
        visconfig.svg.attr("height", segTop - segPadding + 10);
    }

    //edge
    edges.forEach(edge => {
        let node1 = dic[edge.start];
        let node2 = dic[edge.end];
        if (node1.seg != node2.seg) { //long edge
            edge.isLong = true;
        } else {
            edge.isLong = false;
        }
        edge.seg_edges = [];
        edge.seg_nodes = [];
    });

    let segNodes = [], segEdges = [];
    segList.forEach(seg => {
        let segStartNodes = {}, segEndNodes = {};

        seg.forEach(node => {
            for (let i = 0;i < node.in_edges.length;i++) {
                if (edge_dic[node.in_edges[i]].isLong) {
                    if (segStartNodes[node.index] === undefined) {
                        segStartNodes[node.index] = [];
                    }
                    segStartNodes[node.index].push(node.in_edges[i]);
                }
            }

            for (let i = 0;i < node.out_edges.length;i++) {
                if (edge_dic[node.out_edges[i]].isLong) {
                    if (segEndNodes[node.index] === undefined) {
                        segEndNodes[node.index] = [];
                    }
                    segEndNodes[node.index].push(node.out_edges[i]);
                }
            }
        });

        for (let node_index in segStartNodes) {
            let node = dic[node_index];
            let edge_num = segStartNodes[node_index].length;
            for (let i = 0;i < edge_num;i++) {
                segNodes.push({
                    'x': visconfig.leftAndRightSpan / 2,
                    'y': node.y + (i - (edge_num - 1) / 2) * visconfig.segNodeHeight,
                    'type': 'start'
                });
                let points = [{
                        'x': visconfig.leftAndRightSpan / 2 + 4,
                        'y': node.y + (i - (edge_num - 1) / 2) * visconfig.segNodeHeight
                    }, {
                        'x': visconfig.leftAndRightSpan / 2 + 8,
                        'y': node.y + (i - (edge_num - 1) / 2) * visconfig.segNodeHeight
                    }, {
                        'x': node.x - node.w / 2,
                        'y': node.y + (i - (edge_num - 1) / 2) * node.h / edge_num
                    }, {
                        'x': node.x,
                        'y': node.y
                    }];
                if (!node.is_var) {
                    points[2].y = node.y + (i - (edge_num - 1) / 2) * (node.h - visconfig.timeCostHeight) / edge_num;
                }
                segEdges.push({
                    'points': points
                });
                let edge = edge_dic[segStartNodes[node_index][i]];
                edge.seg_edges.push(segEdges.length - 1);
                edge.seg_nodes.push(segNodes.length - 1);
            }
        }

        for (let node_index in segEndNodes) {
            let node = dic[node_index];
            let edge_num = segEndNodes[node_index].length;
            for (let i = 0;i < edge_num;i++) {
                segNodes.push({
                    'x': visconfig.w - visconfig.leftAndRightSpan / 2,
                    'y': node.y + (i - (edge_num - 1) / 2) * visconfig.segNodeHeight,
                    'type': 'end'
                });
                segEdges.push({
                    'points': [{
                            'x': node.x,
                            'y': node.y
                        }, {
                            'x': node.x + node.w / 2,
                            'y': node.y
                        }, {
                            'x': visconfig.w - visconfig.leftAndRightSpan / 2 - 6,
                            'y': node.y + (i - (edge_num - 1) / 2) * visconfig.segNodeHeight
                        }, {
                            'x': visconfig.w - visconfig.leftAndRightSpan / 2,
                            'y': node.y + (i - (edge_num - 1) / 2) * visconfig.segNodeHeight
                        }]
                });
                let edge = edge_dic[segEndNodes[node_index][i]];
                edge.seg_edges.push(segEdges.length - 1);
                edge.seg_nodes.push(segNodes.length - 1);
            }
        }
    });
    return [segNodes, segEdges];
}

function getSegHeight(nodes) {
    let top = Infinity;
    let bottom = -Infinity;
    let left = Infinity;
    let right = -Infinity;
    nodes.forEach(node => {
        if (node.y - node.h / 2 < top) top = node.y - node.h / 2;
        if (node.y + node.h / 2 > bottom) bottom = node.y + node.h / 2;
        if (node.x - node.w / 2 < left) left = node.x - node.w / 2;
        if (node.x + node.w / 2 > right) right = node.x + node.w / 2;
    });
    return {
        "t": top,
        "b": bottom,
        "l": left,
        "r": right
    };
}

function value_equal(a, b) {
    let delta = Math.abs(a - b);
    return delta < 0.001;
}