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
