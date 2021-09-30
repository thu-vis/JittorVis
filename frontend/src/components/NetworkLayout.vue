<template>
<g id="network-layout">
    <g id="network-nodes"></g>
    <g id="network-edges"></g>
</g>
</template>

<script>
import { mapGetters, mapState } from 'vuex'
import dagre from 'dagre'
import * as d3 from 'd3'

export default {
    name: 'network-layout',
    computed: {
        ...mapGetters([
            'network'
        ]),
        ...mapState([
            'focusID'
        ]),
        nodesG: function () {
            return d3.select('g#network-nodes')
        },
        edgesG: function () {
            return d3.select('g#network-edges')
        }
    },
    data: function () {
        return {
            // transition duration
            createDuration: 1000,
            updateDuration: 1000,
            removeDuration: 500,
            // dagre layout options
            dagreLayoutOptions: {
                rankdir: 'TB',
                ranker: 'network-simplex'
            },
            nodeRectAttrs: {
                // default node size
                width: 200,
                height: 40,
                rx: 4,
                fill: 'none',
                stroke: 'black',
                'stroke-width': 2
            },
            nodeNameAttrs: {
                'text-anchor': 'middle',
                x: 100,
                y: 20,
                'font-size': 20,
                dy: 5
            },
            edgeAttrs: {
                fill: 'none',
                stroke: 'rgb(50,50,50)',
                'stroke-width': 1
            },
            // dagre nodes
            nodes: {},
            edges: [],
            // layout class
            nodeClass: 'network-node',
            edgeClass: 'network-edge',
            nodeNameClass: 'network-node-name',
            nodesing: null,
            edgesing: null
        }
    },
    watch: {
        network: function (newNetwork, oldNetwork) {
            [this.nodes, this.edges] = this.getGraphFromNetwork(newNetwork, this.focusID)
            this.computeDAGLayout(this.nodes, this.edges, this.dagreOptions, this.nodeRectAttrs)
            this.draw([this.nodes, this.edges])
        },
        focusID: function (newFocusID, oldFocusID) {
            [this.nodes, this.edges] = this.getGraphFromNetwork(this.network, newFocusID)
            this.computeDAGLayout(this.nodes, this.edges, this.dagreOptions, this.nodeRectAttrs)
            this.draw([this.nodes, this.edges])
        }
    },
    methods: {
        /**
         * a pure function, get nodes and edges for showing from network and focusID
         * @param {Object} network - the whole network
         * @param {string} focusID - node to be focus, whose children will be visualized
         * @return {Array} an array of [nodes, edges]
         */
        getGraphFromNetwork: function (network, focusID) {
            // get nodes
            let focusNode = network[focusID]
            if (focusNode === undefined) {
                throw new Error('wrong focus node id!')
            }
            let nodes = {}
            for (let childID of focusNode.children) {
                nodes[childID] = {
                    ...network[childID]
                }
            }

            // get edges
            let children2parent = {}
            let parent2chilren = {}
            // get two dicts: from nodes to leaves and from leaves to nodes
            // thus edges among nodes could be easily computed
            for (let node of Object.values(nodes)) {
                let queue = [node]
                parent2chilren[node.id] = []
                while (queue.length > 0) {
                    let cur = queue.shift()
                    if (cur.children.length > 0) {
                        queue = queue.concat(cur.children.map(d => network[d]))
                    } else {
                        parent2chilren[node.id].push(cur)
                        children2parent[cur.id] = node
                    }
                }
            }

            // get edges from last two dicts
            let edges = {}
            for (let [nodeID, children] of Object.entries(parent2chilren)) {
                children.forEach(child => {
                    for (let nextId of child.outputs) {
                        let source = network[nodeID]
                        let target = children2parent[nextId]
                        if (target === undefined || nodeID === target.id) {
                            continue
                        }
                        edges[source.id + ',' + target.id] = {
                            source: source,
                            target: target
                        }
                    }
                })
            }
            return [nodes, Object.values(edges)]
        },
        /**
         * compute dag layout, the result will be added as attributes of nodes and edges
         * @param {Object} nodes - nodes of graph
         * @param {Object[]} edges - edges of graph
         * @param {Object} layoutOptions - options for dagre.layout()
         * @param {Object} nodeOptions - options for nodes
         */
        computeDAGLayout: function (nodes, edges, layoutOptions, nodeOptions) {
            // init nodes width
            for (let node of Object.values(nodes)) {
                node.width = nodeOptions.width
                node.height = nodeOptions.height
            }

            // layout
            let g = new dagre.graphlib.Graph()
            g.setGraph({})
            g.setDefaultEdgeLabel(function () { return {} })
            for (let node of Object.values(nodes)) {
                g.setNode(node.id, node)
            }
            for (let edge of edges) {
                g.setEdge(edge.source.id, edge.target.id)
            }
            dagre.layout(g, layoutOptions)
            let edgeObj = {}
            for (let edge of edges) {
                edgeObj[edge.source.id + ',' + edge.target.id] = edge
            }
            g.edges().forEach(function (e) {
                edgeObj[e.v + ',' + e.w].points = g.edge(e).points
            })
        },
        /**
         * main layout function
         */
        draw: async function (graph) {
            let [nodes, edges] = graph
            this.nodesing = this.nodesG.selectAll('g.' + this.nodeClass).data(Object.values(nodes), d => d.id)
            this.edgesing = this.edgesG.selectAll('path.' + this.edgeClass).data(edges, d => d.source.id + ',' + d.target.id)

            await this.remove(graph)
            await this.update(graph)
            await this.create(graph)
            await this.transform(graph)
            console.log('draw done ')
        },
        create: async function () {
            let that = this
            return new Promise((resolve, reject) => {
                let rectAttrs = that.nodeRectAttrs
                let nodesing = that.nodesing.enter()
                    .append('g')
                    .attr('class', that.nodeClass)
                    .attr('transform', d => 'translate(' + d.x + ',' + d.y + ')')
                    .attr('opacity', 0)
                    .on('click', function (e, d) {
                        that.$store.commit('setFocusID', d.id)
                    })
                nodesing.transition()
                    .duration(that.createDuration)
                    .attr('opacity', 1)
                    .on('end', resolve)

                nodesing.append('rect')
                    .attr('width', rectAttrs.width)
                    .attr('height', rectAttrs.height)
                    .attr('rx', rectAttrs.rx)
                    .attr('fill', rectAttrs.fill)
                    .attr('stroke', rectAttrs.stroke)
                    .attr('stroke-width', rectAttrs['stroke-width'])

                nodesing.append('text')
                    .attr('class', that.nodeNameClass)
                    .text(d => d.attrs.type)
                    .attr('x', that.nodeNameAttrs.x)
                    .attr('y', that.nodeNameAttrs.y)
                    .attr('text-anchor', that.nodeNameAttrs['text-anchor'])
                    .attr('font-size', that.nodeNameAttrs['font-size'])
                    .attr('dy', that.nodeNameAttrs.dy)

                let edgeAttrs = that.edgeAttrs
                let link = d3.linkVertical()
                    .source(d => d.points[0])
                    .target(d => d.points[2])
                    .x(d => d.x)
                    .y(d => d.y)
                let edgeing = that.edgesing.enter()
                    .append('path')
                    .attr('class', that.edgeClass)
                    .attr('opacity', 0)
                    .attr('fill', edgeAttrs.fill)
                    .attr('stroke', edgeAttrs.stroke)
                    .attr('stroke-width', edgeAttrs['stroke-width'])
                    .attr('transform', 'translate(' + rectAttrs.width / 2 + ',' + rectAttrs.height / 2 + ')')
                    .attr('d', link)

                edgeing.transition()
                    .duration(that.createDuration)
                    .attr('opacity', 1)
                    .on('end', resolve)

                // if no enter elements, resolve immediately
                if ((that.nodesing.enter().size() === 0) && (that.edgesing.enter().size())) {
                    resolve()
                }
            })
        },
        update: async function () {
            let that = this
            return new Promise((resolve, reject) => {
                let rectAttrs = that.nodeRectAttrs
                let nodesing = that.nodesing
                nodesing.transition()
                    .duration(that.updateDuration)
                    .attr('transform', d => 'translate(' + d.x + ',' + d.y + ')')
                    .on('end', resolve)

                nodesing.selectAll('rect')
                    .transition()
                    .duration(that.updateDuration)
                    .attr('width', rectAttrs.width)
                    .attr('height', rectAttrs.height)
                    .on('end', resolve)

                nodesing.selectAll('.' + that.nodeNameClass)
                    .transition()
                    .duration(that.updateDuration)
                    .attr('x', that.nodeNameAttrs.x)
                    .attr('y', that.nodeNameAttrs.y)
                    .attr('dy', that.nodeNameAttrs.dy)
                    .on('end', resolve)

                let link = d3.linkVertical()
                    .source(d => d.points[0])
                    .target(d => d.points[2])
                    .x(d => d.x)
                    .y(d => d.y)
                let edgeing = that.edgesing
                edgeing.transition()
                    .duration(that.updateDuration)
                    .attr('transform', 'translate(' + rectAttrs.width / 2 + ',' + rectAttrs.height / 2 + ')')
                    .attr('d', link)
                    .on('end', resolve)

                // if no update elements, resolve immediately
                if ((that.nodesing.size() === 0) && (that.edgesing.size() === 0)) {
                    resolve()
                }
            })
        },
        remove: async function () {
            let that = this
            return new Promise((resolve, reject) => {
                that.nodesing.exit()
                    .transition()
                    .duration(that.removeDuration)
                    .remove()
                    .on('end', resolve)

                that.edgesing.exit()
                    .transition()
                    .duration(that.removeDuration)
                    .remove()
                    .on('end', resolve)

                // if no exit elements, resolve immediately
                if ((that.nodesing.exit().size() === 0) && (that.edgesing.exit().size() === 0)) {
                    resolve()
                }
            })
        },
        transform: async function () {

        }

    }
}
</script>
