<template>
    <div class="networkLayoutWarpper">
        <vue-scroll :ops="scrollOptions" style="width:80%" @handle-scroll="updateScroll" ref="vs">
            <svg :id="id" class="one-network" :viewBox="`0 0 ${width} ${height}`" :width="width*scale" :height="height*scale">
                <g id="network-layout" :transform="`translate(${width/2},${heightMargin})`">
                    <g id="background-info"></g>
                    <g id="network-nodes"></g>
                    <g id="network-edges"></g>
                </g>
            </svg>
        </vue-scroll>
        <div style="width:20%; height:100%" @click.prevent="" @mouseenter.prevent="" ref="navContainer">
            <svg class="one-network-nav" :width="width*0.1" :height="height*0.1" @click.prevent="">
            </svg>
        </div>
    </div>
</template>

<script>
import {mapGetters} from 'vuex';
import dagre from 'dagre';
import * as d3 from 'd3';
import clone from 'just-clone';
import Util from './Util.vue';
import GlobalVar from './GlovalVar.vue';

/* eslint-disable */
export default {
    name: 'network-layout',
    mixins: [Util, GlobalVar],
    props: {
        id: {
            type: String,
            default: 'network-all',
        },
        scale: {
            type: Number,
            default: 1,
        },
    },
    computed: {
        ...mapGetters([
            'network',
        ]),
        svg: function() {
            return d3.select('#'+this.id);
        },
        svgMini: function() {
            return d3.select('svg.one-network-nav');
        },
        pageHeight: function() {
            return this.$refs.navContainer.offsetHeight;
        },
        scrollBar: function() {
            return this.$refs['vs'];
        },
        scrollPosition: function() {
            return this.$refs['vs'].getPosition().scrollTop;
        },
        backgroundG: function() {
            return this.svg.select('#background-info');
        },
        mainG: function() {
            return this.svg.select('#network-layout');
        },
        nodesG: function() {
            return this.svg.select('g#network-nodes');
        },
        edgesG: function() {
            return this.svg.select('g#network-edges');
        },
    },
    data: function() {
        return {
            // dagre layout options
            scrollOptions: {
                bar: {
                    background: '#c6bebe',
                },
            },
            dagreLayoutOptions: {
                rankdir: 'TB',
                ranker: 'network-simplex',
                ranksep: 30,
            },
            nodeRectAttrs: {
                // default node size
                'height': 40,
                'attrHeight': 20,
                'rx': 4,
                'fill': 'white',
                'stroke': 'black',
                'stroke-width': 2,
                'cursor': 'pointer',
            },
            nodeNameAttrs: {
                'text-anchor': 'middle',
                'font-family': 'Comic Sans MS',
                'font-weight': 'normal',
                'x': 100,
                'y': 20,
                'cursor': 'pointer',
                'font-size': '20px',
                'dy': 25,
            },
            nodeAttrAttrs: {
                'text-anchor': 'start',
                'font-family': 'Goudy Old Style',
                'font-weight': 'normal',
                'x': 5,
                'font-size': '12px',
                'dy': 15,
                'cursor': 'pointer',
            },
            nodeSepAttrs: {
                'stroke': 'black',
                'stroke-width': 1,
            },
            edgeAttrs: {
                'fill': 'none',
                'stroke': 'rgb(50,50,50)',
                'stroke-width': 1,
            },
            nodeBackgroundAttrs: {
                'height': 40,
                'widthMargin': 30,
                'rx': 4,
                'fill': 'white',
                'stroke': 'none',
                'opacity': 0,
            },
            // dagre nodes/edges/graph result
            localLayoutNetwork: {},
            nodes: {},
            edges: [],
            daggraph: {
                width: 0,
                height: 0,
            },
            heightMargin: 50, // margin-top, margin-bottom of main layout
            widthMargin: 200,
            width: 0, // width of dag graph
            height: 0,
            // layout class
            nodeClass: 'network-node',
            edgeClass: 'network-edge',
            nodeRectClass: 'network-rect',
            nodeBackgroundClass: 'network-node-background',
            nodeNameClass: 'network-node-name',
            nodeAttrsClass: 'network-node-attrs',
            nodeSepClass: 'network-node-sep',
            nodeParentGIDPrefix: 'network-node-parent-',
            nodeParentGClass: 'network-node-parent',
            nodesing: null,
            edgesing: null,
            nodeToolBtnsClass: 'network-node-tools',
        };
    },
    watch: {
        network: function(newnetwork, oldnetwork) {
            const newLayoutNetwork = clone(newnetwork);
            if (Object.keys(newLayoutNetwork).length===0) {
                return;
            }
            // init extent
            Object.values(newLayoutNetwork).forEach((d) => {
                d.expand = false;
            });
            // find root
            let root = Object.values(newLayoutNetwork)[0];
            while (root.parent !== undefined) {
                root = newLayoutNetwork[root.parent];
            }
            root.expand = true;
            this.localLayoutNetwork = newLayoutNetwork;
            this.drawAllLayout();
        },
        scale: function(newScale, oldScale) {
            this.updateNav();
        },
    },
    mounted: function() {
        const newLayoutNetwork = clone(this.network);
        if (Object.keys(newLayoutNetwork).length===0) {
            return;
        }
        // init extent
        Object.values(newLayoutNetwork).forEach((d) => {
            d.expand = false;
        });
        // find root
        let root = Object.values(newLayoutNetwork)[0];
        while (root.parent !== undefined) {
            root = newLayoutNetwork[root.parent];
        }
        root.expand = true;
        this.localLayoutNetwork = newLayoutNetwork;
        this.drawAllLayout();
    },
    methods: {
        /**
         * all things to do when you expand a network node, it will call following methods:
         * 1. getGraphFromNetwork() to calculate which nodes should be showed after expanding and the edges among them
         * 2. computeDAGLayout() to compute the dag layout
         * 3. emit reheight event to resize the svg, which works well with scroll
         * 4. draw() to render the network
         *
         * @param {string} nodeid - node to be expanded
         * @public
         */
        expandNode: function(nodeid) {
            let node = this.localLayoutNetwork[nodeid];
            // expand parent
            while (node !== undefined) {
                node.expand = true;
                node = this.localLayoutNetwork[node.parent];
            }
            this.drawAllLayout();
        },
        drawAllLayout: function() {
            [this.nodes, this.edges] = this.getGraphFromNetwork(this.localLayoutNetwork);
            this.computeDAGLayout(this.nodes, this.edges, this.daggraph, this.dagreLayoutOptions,
                this.nodeRectAttrs, this.nodeNameAttrs, this.nodeAttrAttrs);

            // if new width/height less than width/height, change width/height after draw to avoid the transition being occluded
            const newWidth = this.daggraph.width+this.widthMargin*2;
            const newHeight = this.daggraph.height + this.heightMargin*2;
            if (newWidth> this.width) {
                this.width = newWidth;
            }
            if (newHeight> this.height) {
                this.height = newHeight;
            }
            const that = this;
            this.draw([this.nodes, this.edges])
                .then(function() {
                    that.width = newWidth;
                    that.height = newHeight;
                });
        },
        /**
         * call expand(node.parent) to collapse a network node
         *
         * @param {string} nodeid - node to be collapsed
         * @public
         */
        collapseNode: function(nodeid) {
            const nodeParent = this.localLayoutNetwork[nodeid].parent;
            // collapse children
            let queues = [nodeid];
            while (queues.length > 0) {
                nodeid = queues.shift();
                const node = this.localLayoutNetwork[nodeid];
                node.expand = false;
                queues = queues.concat(node.children);
            }

            this.expandNode(nodeParent);
        },
        /**
         * init this.localLayoutNetwork based on rawNetwork, should be called when raw network data was changed
         * @param {Object} rawNetwork - raw network data
         * @public
         */
        initGraphNetwork: function(rawNetwork) {
            this.localLayoutNetwork = clone(rawNetwork);
            if (this.localLayoutNetwork==={}) {
                return;
            }
            // init extent
            Object.values(this.localLayoutNetwork).forEach((d) => {
                d.expand = false;
            });
        },
        /**
         * a pure function, get nodes and edges for showing from network
         * @param {Object} network - the whole network
         * @return {Array} an array of [nodes, edges]
         * @public
         */
        getGraphFromNetwork: function(network) {
            // find root
            let root = Object.values(network)[0];
            while (root.parent !== undefined) {
                root = network[root.parent];
            }
            // get nodes
            const nodes = {};
            const queue = [root.id];
            while (queue.length>0) {
                const nodeid = queue.shift();
                network[nodeid].children.forEach((childid) => {
                    if (network[childid].expand) {
                        queue.push(childid);
                    } else {
                        nodes[childid] = network[childid];
                    }
                });
            }

            // get edges
            const children2parent = {};
            const parent2chilren = {};
            // get two dicts: from nodes to leaves and from leaves to nodes
            // thus edges among nodes could be easily computed
            for (const node of Object.values(nodes)) {
                let queue = [node];
                parent2chilren[node.id] = [];
                while (queue.length > 0) {
                    const cur = queue.shift();
                    if (cur.children.length > 0) {
                        queue = queue.concat(cur.children.map((d) => network[d]));
                    } else {
                        parent2chilren[node.id].push(cur);
                        children2parent[cur.id] = node;
                    }
                }
            }

            // get edges from last two dicts
            const edges = {};
            for (const [nodeID, children] of Object.entries(parent2chilren)) {
                children.forEach((child) => {
                    for (const nextId of child.outputs) {
                        const source = network[nodeID];
                        const target = children2parent[nextId];
                        if (target === undefined || nodeID === target.id) {
                            continue;
                        }
                        edges[source.id + ',' + target.id] = {
                            source: source,
                            target: target,
                        };
                    }
                });
            }
            return [nodes, Object.values(edges)];
        },
        /**
         * compute dag layout, the result will be added as attributes of nodes and edges
         * @param {Object} nodes - nodes of graph
         * @param {Object[]} edges - edges of graph
         * @param {Object} graph - graph width/height
         * @param {Object} layoutOptions - options for dagre.layout()
         * @param {Object} nodeOptions - options for nodes
         * @param {Object} nodeNameOptions - options for nodes name
         * @param {Object} nodeAttrOptions - options for node attrs
         * @public
         */
        computeDAGLayout: function(nodes, edges, graph, layoutOptions, nodeOptions, nodeNameOptions, nodeAttrOptions) {
            // init nodes width
            for (const node of Object.values(nodes)) {
                let maxWidth = this.getTextWidth(node.type,
                    `${nodeNameOptions['font-weight']} ${nodeNameOptions['font-size']} ${nodeNameOptions['font-family']}`);
                for (const [attrname, attr] of Object.entries(node.attrs)) {
                    maxWidth = Math.max(maxWidth, this.getTextWidth(attrname+': '+attr,
                        `${nodeAttrOptions['font-weight']} ${nodeAttrOptions['font-size']} ${nodeAttrOptions['font-family']}`));
                }

                node.width = maxWidth + 15*2;
                node.height = nodeOptions.height + Object.entries(node.attrs).length * nodeOptions.attrHeight;
            }

            // layout
            const g = new dagre.graphlib.Graph();
            g.setGraph(layoutOptions);
            g.setDefaultEdgeLabel(function() {
                return {};
            });
            for (const node of Object.values(nodes)) {
                g.setNode(node.id, node);
            }
            for (const edge of edges) {
                g.setEdge(edge.source.id, edge.target.id);
            }
            dagre.layout(g);

            // move graph x-center to 0
            let nodeCenterX = 0;
            let nodeCenterCnts = 0;
            let nodeCenterY = Number.MAX_VALUE;
            g.nodes().forEach(function(d) {
                const node = g.node(d);
                if (nodeCenterY>node.y-node.height/2) {
                    nodeCenterY = node.y-node.height/2;
                    nodeCenterX = node.x;
                    nodeCenterCnts = 1;
                } else if (nodeCenterY === node.y-node.height/2) {
                    nodeCenterX += node.x;
                    nodeCenterCnts++;
                }
            });
            nodeCenterX = nodeCenterX / nodeCenterCnts;
            g.nodes().forEach(function(d) {
                g.node(d).x -= nodeCenterX;
                g.node(d).y -= nodeCenterY;
            });

            // set edge layout result
            const edgeObj = {};
            for (const edge of edges) {
                edgeObj[edge.source.id + ',' + edge.target.id] = edge;
            }
            g.edges().forEach(function(e) {
                g.edge(e).points.forEach(function(e) {
                    e.x -= nodeCenterX;
                    e.y -= nodeCenterY;
                });

                edgeObj[e.v + ',' + e.w].points = g.edge(e).points;
            });

            graph.width = g.graph().width;
            graph.height = g.graph().height;
        },
        /**
         * main render function, which will call: create(), update(), remove()
         *
         * @param {Object[]} graph - [nodes, edges]
         * @public
         */
        draw: async function(graph) {
            const [nodes, edges] = graph;
            this.nodesing = this.nodesG.selectAll('g.' + this.nodeClass)
                .data(Object.values(nodes), (d) => d.id);
            this.edgesing = this.edgesG.selectAll('path.' + this.edgeClass)
                .data(edges, (d) => d.source.id + ',' + d.target.id);

            await this.remove(graph);
            await this.update(graph);
            await this.create(graph);
            this.updateNav();
        },
        /**
         * a tool function for beautiful edge routing
         *
         * @param {Object[]} points - array of edge points
         *
         * @return {string} pathData - a string for svg.path.d
         * @public
         */
        one_edge: function(points) {
        // const movePoint = (p, x, y, s) => {
        //     return { x: p.x * s + x, y: p.y * s + y }
        // };
        // points = points.map(p => movePoint(p, transX, transY, scale))


            const len = points.length;
            if (len === 0) {
                return '';
            }
            const start = `M ${points[0].x} ${points[0].y}`;
            let vias = [];

            const getInter = (p1, p2, n) => {
                return `${p1.x * n + p2.x * (1 - n)} ${p1.y * n + p2.y * (1 - n)}`;
            };

            const getCurve = (points) => {
                const vias = [];
                const len = points.length;
                const ratio = 0.5;
                for (let i = 0; i < len - 2; i++) {
                    let p1;
                    let p5;
                    if (i === 0) {
                        p1 = `${points[i].x} ${points[i].y}`;
                    } else {
                        p1 = getInter(points[i], points[i + 1], ratio);
                    }
                    const p2 = getInter(points[i], points[i + 1], 1 - ratio);
                    const p3 = `${points[i + 1].x} ${points[i + 1].y}`;
                    const p4 = getInter(points[i + 1], points[i + 2], ratio);
                    if (i === len - 3) {
                        p5 = `${points[i + 2].x} ${points[i + 2].y}`;
                    } else {
                        p5 = getInter(points[i + 1], points[i + 2], 1 - ratio);
                    }
                    const cPath = `M ${p1} L${p2} Q${p3} ${p4} L${p5}`;
                    vias.push(cPath);
                }
                return vias;
            };
            vias = getCurve(points);
            const pathData = `${start}  ${vias.join(' ')}`;
            return pathData;
        },
        create: async function() {
            const that = this;
            return new Promise((resolve, reject) => {
                const rectAttrs = that.nodeRectAttrs;
                const nodesing = that.nodesing.enter()
                    .append('g')
                    .attr('class', that.nodeClass)
                    .attr('transform', (d) => 'translate(' + d.x + ',' + d.y + ')')
                    .attr('opacity', 0)
                    .on('mouseenter', function(e, d) {
                        // eslint-disable-next-line no-invalid-this
                        const ele = d3.select(this);
                        that.drawToolBtns(d, ele, that.nodeToolBtnsClass, {
                            collapseNode: that.collapseNode,
                            expandNode: that.expandNode,
                        }, that.localLayoutNetwork);

                        const parentID = that.nodes[d.id].parent.split('/').join('-');
                        that.drawNodeParent(d.id, that.localLayoutNetwork, that.nodes, that.backgroundG,
                            that.nodeParentGIDPrefix+parentID, that.nodeParentGClass);
                        that.updateNav();
                    })
                    .on('mouseleave', function(e, d) {
                        // eslint-disable-next-line no-invalid-this
                        const ele = d3.select(this);
                        const parentID = that.nodes[d.id].parent.split('/').join('-');
                        that.removeToolBtns(ele, that.nodeToolBtnsClass);
                        that.removeNodeParent(that.backgroundG, that.nodeParentGIDPrefix+parentID);
                        that.updateNav();
                    });

                nodesing.transition()
                    .duration(that.createDuration)
                    .attr('opacity', 1)
                    .on('end', resolve);

                nodesing.append('rect')
                    .attr('class', that.nodeBackgroundClass)
                    .attr('width', (d) => d.width+that.nodeBackgroundAttrs['widthMargin'])
                    .attr('height', (d) => d.height)
                    .attr('x', (d) => -d.width/2)
                    .attr('y', (d) => -d.height/2)
                    .attr('rx', that.nodeBackgroundAttrs.rx)
                    .attr('opacity', that.nodeBackgroundAttrs.opacity)
                    .attr('fill', that.nodeBackgroundAttrs.fill)
                    .attr('stroke', that.nodeBackgroundAttrs.stroke);

                nodesing.append('rect')
                    .attr('class', that.nodeRectClass)
                    .attr('width', (d) => d.width)
                    .attr('height', (d) => d.height)
                    .attr('x', (d) => -d.width/2)
                    .attr('y', (d) => -d.height/2)
                    .attr('rx', rectAttrs.rx)
                    .attr('fill', rectAttrs.fill)
                    .attr('stroke', rectAttrs.stroke)
                    .attr('stroke-width', rectAttrs['stroke-width'])
                    .attr('cursor', rectAttrs.cursor)
                    .on('click', function(e, d) {
                        that.$store.commit('setFeatureMapNodeID', d.id);
                    });

                nodesing.append('text')
                    .attr('class', that.nodeNameClass)
                    .text((d) => d.type)
                    .attr('x', (d) => 0)
                    .attr('y', (d) => -d.height/2)
                    .attr('text-anchor', that.nodeNameAttrs['text-anchor'])
                    .attr('font-size', that.nodeNameAttrs['font-size'])
                    .attr('cursor', that.nodeNameAttrs.cursor)
                    .attr('dy', that.nodeNameAttrs.dy)
                    .attr('font-family', that.nodeNameAttrs['font-family'])
                    .on('click', function(e, d) {
                        that.$store.commit('setFeatureMapNodeID', d.id);
                    });

                nodesing.append('line')
                    .attr('class', that.nodeSepClass)
                    .attr('x1', (d) => -d.width/2)
                    .attr('y1', (d) => -d.height/2+that.nodeRectAttrs['height'])
                    .attr('x2', (d) => d.width/2)
                    .attr('y2', (d) => -d.height/2+that.nodeRectAttrs['height'])
                    .attr('stroke', that.nodeSepAttrs['stroke'])
                    .attr('stroke-width', that.nodeSepAttrs['stroke-width']);

                nodesing.each(function(d) {
                    let i = 0;
                    // eslint-disable-next-line no-invalid-this
                    const ele = d3.select(this);
                    for (const [attrname, attrvalue] of Object.entries(d.attrs)) {
                        ele.append('text')
                            .attr('class', that.nodeAttrsClass)
                            .attr('id', attrname+','+i)
                            .text(`${attrname}: ${attrvalue}`)
                            .attr('x', that.nodeAttrAttrs['x']-d.width/2)
                            .attr('y', -d.height/2 + that.nodeRectAttrs['height']+20*i)
                            .attr('text-anchor', that.nodeAttrAttrs['text-anchor'])
                            .attr('font-size', that.nodeAttrAttrs['font-size'])
                            .attr('dy', that.nodeAttrAttrs.dy)
                            .attr('cursor', that.nodeAttrAttrs.cursor)
                            .attr('font-family', that.nodeAttrAttrs['font-family'])
                            .on('click', function() {
                                that.$store.commit('setFeatureMapNodeID', d.id);
                            });
                        i++;
                    }
                });

                const edgeAttrs = that.edgeAttrs;
                const edgeing = that.edgesing.enter()
                    .append('path')
                    .attr('class', that.edgeClass)
                    .attr('opacity', 0)
                    .attr('fill', edgeAttrs.fill)
                    .attr('stroke', edgeAttrs.stroke)
                    .attr('stroke-width', edgeAttrs['stroke-width'])
                    .attr('transform', (d) => 'translate(' + 0 + ',' + 0 + ')')
                    .attr('d', (d) => that.one_edge(d.points));

                edgeing.transition()
                    .duration(that.createDuration)
                    .attr('opacity', 1)
                    .on('end', resolve);

                // if no enter elements, resolve immediately
                if ((that.nodesing.enter().size() === 0) && (that.edgesing.enter().size())) {
                    resolve();
                }
            });
        },
        update: async function() {
            const that = this;
            return new Promise((resolve, reject) => {
                const nodesing = that.nodesing;
                nodesing.transition()
                    .duration(that.updateDuration)
                    .attr('transform', (d) => 'translate(' + d.x + ',' + d.y + ')')
                    .on('end', resolve);

                nodesing.selectAll('.'+that.nodeBackgroundClass)
                    .attr('width', (d) => d.width+that.nodeBackgroundAttrs['widthMargin'])
                    .attr('height', (d) => d.height)
                    .attr('x', (d) => -d.width/2)
                    .attr('y', (d) => -d.height/2);

                nodesing.selectAll('.'+that.nodeRectClass)
                    .transition()
                    .duration(that.updateDuration)
                    .attr('width', (d) => d.width)
                    .attr('height', (d) => d.height)
                    .attr('x', (d) => -d.width/2)
                    .attr('y', (d) => -d.height/2)
                    .on('end', resolve);

                nodesing.selectAll('.' + that.nodeNameClass)
                    .transition()
                    .duration(that.updateDuration)
                    .attr('x', (d) => 0)
                    .attr('y', (d) => -d.height/2)
                    .attr('dy', that.nodeNameAttrs.dy)
                    .on('end', resolve);

                nodesing.selectAll('.' + that.nodeSepClass)
                    .attr('x1', (d) => -d.width/2)
                    .attr('y1', (d) => -d.height/2+that.nodeRectAttrs['height'])
                    .attr('x2', (d) => d.width/2)
                    .attr('y2', (d) => -d.height/2+that.nodeRectAttrs['height']);

                nodesing.each(function(d) {
                    // eslint-disable-next-line no-invalid-this
                    const ele = d3.select(this);
                    ele.selectAll('.'+that.nodeAttrsClass)
                        .attr('x', that.nodeAttrAttrs['x']-d.width/2)
                        .attr('y', function(d) {
                            // eslint-disable-next-line no-invalid-this
                            const i = parseInt(d3.select(this).attr('id').split(',')[1]);
                            return -d.height/2 + that.nodeRectAttrs['height']+20*i;
                        });
                });

                const edgeing = that.edgesing;
                edgeing.transition()
                    .duration(that.updateDuration)
                    .attr('transform', (d) => 'translate(' + 0 + ',' + 0 + ')')
                    .attr('d', (d) => that.one_edge(d.points))
                    .on('end', resolve);

                // if no update elements, resolve immediately
                if ((that.nodesing.size() === 0) && (that.edgesing.size() === 0)) {
                    resolve();
                }
            });
        },
        remove: async function() {
            const that = this;
            return new Promise((resolve, reject) => {
                that.nodesing.exit()
                    .transition()
                    .duration(that.removeDuration)
                    .attr('opacity', 0)
                    .remove()
                    .on('end', resolve);

                that.edgesing.exit()
                    .transition()
                    .duration(that.removeDuration)
                    .attr('opacity', 0)
                    .remove()
                    .on('end', resolve);

                // remove hover parent
                that.mainG.selectAll('.'+that.nodeParentGClass).remove();

                // if no exit elements, resolve immediately
                if ((that.nodesing.exit().size() === 0) && (that.edgesing.exit().size() === 0)) {
                    resolve();
                }
            });
        },
        updateNav: function() {
            const that=this;
            const content=this.svg.html();
            window.content=content;
            window.t=this;
            window.d3=d3;
            this.svgMini.html(content);
            this.svgMini.append('rect')
                .attr('id', 'networkNavBar')
                .attr('fill', '#c6bebe')
                .attr('opacity', 0.3)
                .call(d3.drag()
                    .on('start', function() {
                        d3.select(this).attr('opacity', 0.5);
                    })
                    .on('drag', function(e) {
                        console.log(e.dy, that.scale, 10);
                        that.scrollBar.scrollBy({
                            'dx': e.dx * 1.44 * that.scale,
                            'dy': e.dy * 1.44 * that.scale,
                        });
                    })
                    .on('end', function() {
                        d3.select(this).attr('opacity', 0.3);
                    }),
                );
            this.updateScroll();
        },
        updateScroll: function() {
            const pos=this.scrollBar.getPosition().scrollTop/this.scale;
            // console.log('pos', pos,this.scale,this.pageHeight);
            const barHeight = this.pageHeight/this.scale;

            this.svgMini.select('#networkNavBar')
                .attr('x', 0)
                .attr('y', pos)
                .attr('width', 2000)
                .attr('height', barHeight);

            if (this.height*0.1 > this.pageHeight) {
                this.svgMini.attr('viewBox',
                    [0, pos/(this.height-this.pageHeight/this.scale) * (this.height-this.pageHeight*10), this.width, this.pageHeight*10])
                    .attr('height', this.pageHeight);
            } else {
                this.svgMini.attr('viewBox', [0, 0, this.width, this.height]);
            }
        },
        /**
         * draw tool buttons when hover on a network node
         *
         * @param {Object} d - data of network node
         * @param {Object} ele - d3 element( d3.select(this) )
         * @param {string} gClass - class of buttons' group
         * @param {Object} callbacks - click functions of buttons
         * @param {Object} nodes - all nodes
         * @public
         */
        drawToolBtns: function(d, ele, gClass, callbacks, nodes) {
            const iconWidth = 15;
            const iconMargin = 5;
            const elementWidth = d.width;
            let top = 5;
            const g = ele.append('g').attr('class', gClass)
                .attr('transform', `translate(0,${-d.height/2})`);

            if (d.children.length > 0) {
                // draw zoom-in
                g.append('image')
                    .attr('xlink:href', '/static/images/zoomin.svg')
                    .attr('x', elementWidth/2+iconMargin)
                    .attr('y', top)
                    .attr('width', iconWidth)
                    .attr('height', iconWidth)
                    .attr('cursor', 'pointer')
                    .on('click', function() {
                        callbacks.expandNode(d.id);
                    });

                top += iconWidth+iconMargin;
            }

            if (d.parent !== undefined && nodes[d.parent].parent !== undefined) {
                // draw zoom-out
                g.append('image')
                    .attr('xlink:href', '/static/images/zoomout.svg')
                    .attr('x', elementWidth/2+iconMargin)
                    .attr('y', top)
                    .attr('width', iconWidth)
                    .attr('height', iconWidth)
                    .attr('cursor', 'pointer')
                    .on('click', function() {
                        if (d.parent != undefined) {
                            callbacks.collapseNode(d.parent);
                        }
                    });
                top += iconWidth+iconMargin;
            }
        },
        /**
         * remove tool buttons when mouse leave a network node
         *
         * @param {Object} ele - d3 element( d3.select(this) )
         * @param {string} gClass - class of buttons' group
         * @public
         */
        removeToolBtns: function(ele, gClass) {
            ele.select('.'+gClass).remove();
        },
        /**
         * show node parent when hover on a node
         * @param {string} nodeid - nodeid
         * @param {Object} allnodes - whole network
         * @param {Object} shownodes - nodes
         * @param {Object} container - container group
         * @param {string} gID - id of group
         * @param {string} gClass - class of group
         */
        drawNodeParent: function(nodeid, allnodes, shownodes, container, gID, gClass) {
            const nodeParentID = allnodes[nodeid].parent;
            const nodeParent = allnodes[nodeParentID];
            if (nodeParent.parent === undefined) {
                // don't show root nodes
                return;
            }

            // first, find all siblings and children
            let queue = [].concat(nodeParent.children);
            const allChildren = [];
            while (queue.length>0) {
                const nodeID = queue.shift();
                const node = allnodes[nodeID];
                if (shownodes[nodeID] !== undefined) {
                    allChildren.push(shownodes[nodeID]);
                }
                queue = queue.concat(node.children);
            }

            // compute the rect
            const rect = {
                minx: Number.MAX_SAFE_INTEGER,
                miny: Number.MAX_SAFE_INTEGER,
                maxx: Number.MIN_SAFE_INTEGER,
                maxy: Number.MIN_SAFE_INTEGER,
            };
            for (const child of allChildren) {
                rect.minx = Math.min(rect.minx, child.x-child.width/2);
                rect.miny = Math.min(rect.miny, child.y-child.height/2);
                rect.maxx = Math.max(rect.maxx, child.x+child.width/2);
                rect.maxy = Math.max(rect.maxy, child.y+child.height/2);
            }
            const rectWidthMargin = 25;
            const rectHeightMargin = 10;
            rect.minx -= rectWidthMargin;
            rect.miny -= rectHeightMargin;
            rect.maxx += rectWidthMargin;
            rect.maxy += rectHeightMargin;

            // draw
            const g = container.append('g')
                .attr('id', gID)
                .attr('class', gClass)
                .attr('transform', `translate(${rect.minx},${rect.miny})`);

            g.append('rect')
                .attr('x', 0)
                .attr('y', 0)
                .attr('width', rect.maxx-rect.minx)
                .attr('height', rect.maxy-rect.miny)
                .attr('fill', '#F5F5F5')
                .attr('stroke', 'black')
                .attr('stroke-width', 1)
                .attr('stroke-dasharray', '5,5')
                .attr('rx', 5)
                .attr('ry', 5);

            g.append('text')
                .text(nodeParent.type)
                .attr('x', rect.maxx-rect.minx+5)
                .attr('y', shownodes[nodeid].y-rect.miny)
                .attr('text-anchor', 'start')
                .attr('font-size', '20px')
                .attr('dy', '10px')
                .attr('font-family', 'Comic Sans MS');
        },
        /**
         * remove node parent
         * @param {Object} container - container group
         * @param {string} gID - id of group
         */
        removeNodeParent: function(container, gID) {
            container.select('#'+gID).remove();
        },
    },
};
</script>
<style>
div.networkLayoutWarpper{
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: row;
}
</style>
