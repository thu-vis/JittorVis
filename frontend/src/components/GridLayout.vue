<template>
    <div id="grid-layout">
        <div id="grid-icons">
            <img id="grid-zoomin-icon" class="grid-icon" src="/static/images/zoomin.svg" @click="initlasso">
            <img id="grid-home-icon" class="grid-icon" src="/static/images/home.png" @click="zoomin()">
        </div>
        <svg id="grid-drawer" ref="gridsvg">
            <g id="grid-main-g" transform="translate(0,0)">
                <g id="grid-g"></g>
                <g id="highlight-g"></g>
                <g id="lasso-g"></g>
            </g>
        </svg>
        <waiting-icon v-if="rendering"></waiting-icon>
    </div>
</template>

<script>
import {mapGetters} from 'vuex';
import axios from 'axios';
import * as d3 from 'd3';
window.d3 = d3;
require('../js/d3-lasso.js');
import Util from './Util.vue';
import GlobalVar from './GlovalVar.vue';
import WaitingIcon from './WaitingIcon.vue';
import cloneDeep from 'lodash.clonedeep';
import PriorityQueue from 'priorityqueue';
import {createPopper} from '@popperjs/core';

export default {
    name: 'GridLayout',
    components: {WaitingIcon},
    mixins: [Util, GlobalVar],
    computed: {
        ...mapGetters([
            'confusionMatrix',
            'labelHierarchy',
            'labelnames',
            'URL_GET_GRID',
            'hierarchyColors',
            'colors',
        ]),
        svg: function() {
            return d3.select('#grid-drawer');
        },
        mainG: function() {
            return this.svg.select('#grid-main-g');
        },
        girdG: function() {
            return this.mainG.select('#grid-g');
        },
        lassoG: function() {
            return this.mainG.select('#lasso-g');
        },
        svgWidth: function() {
            return this.gridCellAttrs['size'] * this.gridInfo['width'];
        },
        nodesDict: function() {
            const nodesDict = {};
            for (const node of this.nodes) {
                nodesDict[node.index] = node;
            }
            return nodesDict;
        },
        lasso: function() {
            return d3.lasso;
        },
        highlightG: function() {
            return this.mainG.select('#highlight-g');
        },
    },
    watch: {
        // all info was loaded
        labelnames: function(newColors, oldColors) {
            if (!this.rendering && this.nodes.length>0 ) {
                this.render();
            }
        },
    },
    data: function() {
        return {
            nodes: [],
            showImageNodesMax: 40,
            showImageNodes: [],
            depth: 0,
            gridInfo: {},
            rendering: false,

            //
            gridCellsInG: undefined,
            lassoNodesInG: undefined,

            //
            gridCellAttrs: {
                'gClass': 'grid-cell-in-g',
                'size': 30,
                'stroke-width': 0,
                'stroke': 'gray',
                'rectOpacity': 1,
                'centerR': 3,
                'centerClass': 'lasso-node',
                'centerClassNotSelect': 'lasso-not-possible',
                'centerClassSelect': 'lasso-possible',
                'imageMargin': 2,
            },

            tooltipClass: 'cell-tooltip',
        };
    },
    methods: {
        zoomin: function(nodes) {
            this.rendering = true;
            if (nodes===undefined) {
                // zoom home
                nodes = this.nodes;
                this.depth = 0;
            }
            if (nodes.length>0 && typeof(nodes[0])!=='number') {
                nodes = nodes.map((d) => d.index);
            }
            const that = this;
            const tsnes = nodes.map((d) => this.nodesDict[d].tsne);
            axios.post(this.URL_GET_GRID, {
                nodes: nodes,
                depth: this.depth,
                constraints: tsnes,
            }).then(function(response) {
                that.nodes = response.data.nodes;
                that.depth = response.data.depth;
                that.gridInfo = response.data.grid;
                that.render();
            });
        },
        render: async function() {
            // sort nodes and find most unconfident nodes
            this.nodes.sort(function(a, b) {
                return a.confidence-b.confidence;
            });
            for (let i=0; i<Math.min(this.showImageNodesMax, this.nodes.length); i++) {
                this.nodes[i].showImage = true;
            }
            // set color
            this.setLabelColor(this.labelHierarchy, this.colors, this.nodes, this.labelnames);

            this.gridCellsInG = this.girdG.selectAll('.'+this.gridCellAttrs['gClass']).data(this.nodes, (d)=>d.index);
            this.lassoNodesInG = this.lassoG.selectAll('.'+this.gridCellAttrs['centerClass']).data(this.nodes, (d)=>d.index);

            await this.remove();
            this.transform();
            await this.update();
            await this.create();

            this.gridCellsInG = this.girdG.selectAll('.'+this.gridCellAttrs['gClass']);
            this.lassoNodesInG = this.lassoG.selectAll('.'+this.gridCellAttrs['centerClass']);
            this.rendering = false;
        },
        create: async function() {
            const that = this;
            return new Promise((resolve, reject) => {
                const gridCellsInG = that.gridCellsInG.enter()
                    .append('g')
                    .attr('class', that.gridCellAttrs['gClass'])
                    .attr('opacity', 0)
                    .attr('transform', (d) => `translate(${(d.grid%that.gridInfo.width)*that.gridCellAttrs['size']},
                        ${Math.floor(d.grid/that.gridInfo.width)*that.gridCellAttrs['size']})`)
                    .on('mouseenter', function(e, d) {
                        // eslint-disable-next-line no-invalid-this
                        const node = d3.select(this).node();
                        that.$emit('hoveredNode', [that.labelnames[d.label], that.labelnames[d.pred]]);
                        that.createTooltip(d)
                            .then(function(tooltip) {
                                createPopper(node, tooltip, {
                                    modifiers: [
                                        {
                                            name: 'offset',
                                            options: {
                                                offset: [0, 8],
                                            },
                                        },
                                    ],
                                });
                            });
                    })
                    .on('mouseleave', function() {
                        that.$emit('hoveredNode', [null, null]);
                        that.removeTooltip();
                    });

                gridCellsInG.transition()
                    .duration(that.createDuration)
                    .attr('opacity', 1)
                    .on('end', resolve);

                gridCellsInG.append('rect')
                    .attr('x', 0)
                    .attr('y', 0)
                    .attr('width', that.gridCellAttrs['size'])
                    .attr('height', that.gridCellAttrs['size'])
                    .attr('stroke', that.gridCellAttrs['stroke'])
                    .attr('stroke-width', that.gridCellAttrs['stroke-width'])
                    .attr('fill', (d)=>that.hierarchyColors[that.labelnames[d.pred]].fill)
                    .attr('opacity', (d)=>that.hierarchyColors[that.labelnames[d.pred]].opacity);

                gridCellsInG.filter(function(d) {
                    return d.showImage;
                }).append('image')
                    .attr('x', that.gridCellAttrs['imageMargin'])
                    .attr('y', that.gridCellAttrs['imageMargin'])
                    .attr('width', that.gridCellAttrs['size']-2*that.gridCellAttrs['imageMargin'])
                    .attr('height', that.gridCellAttrs['size']-2*that.gridCellAttrs['imageMargin'])
                    .attr('href', '')
                    .each(function(node) {
                        const getImageGradientURL = that.$store.getters.URL_GET_IMAGE_GRADIENT;
                        // eslint-disable-next-line no-invalid-this
                        const img = d3.select(this);
                        axios.get(getImageGradientURL(node.index, 'origin'))
                            .then(function(response) {
                                img.attr('href', that.toImage(response.data));
                            });
                    });

                that.lassoNodesInG.enter().append('circle')
                    .attr('class', that.gridCellAttrs['centerClass'])
                    .attr('r', that.gridCellAttrs['centerR'])
                    .attr('cx', (d)=>that.gridCellAttrs['size']/2+(d.grid%that.gridInfo.width)*that.gridCellAttrs['size'])
                    .attr('cy', (d)=>that.gridCellAttrs['size']/2+Math.floor(d.grid/that.gridInfo.width)*that.gridCellAttrs['size']);


                if ((that.gridCellsInG.enter().size() === 0) && (that.lassoNodesInG.enter().size() === 0)) {
                    resolve();
                }
            });
        },
        update: async function() {
            const that = this;
            return new Promise((resolve, reject) => {
                that.gridCellsInG.transition()
                    .duration(that.updateDuration)
                    .attr('transform', (d) => `translate(${(d.grid%that.gridInfo.width)*that.gridCellAttrs['size']},
                        ${Math.floor(d.grid/that.gridInfo.width)*that.gridCellAttrs['size']})`)
                    .on('end', resolve);

                that.gridCellsInG.selectAll('rect')
                    .transition()
                    .duration(that.updateDuration)
                    .attr('fill', (d)=>that.hierarchyColors[that.labelnames[d.pred]].fill)
                    .attr('opacity', (d)=>that.hierarchyColors[that.labelnames[d.pred]].opacity)
                    .on('end', resolve);

                that.lassoNodesInG
                    .attr('cx', (d)=>that.gridCellAttrs['size']/2+(d.grid%that.gridInfo.width)*that.gridCellAttrs['size'])
                    .attr('cy', (d)=>that.gridCellAttrs['size']/2+Math.floor(d.grid/that.gridInfo.width)*that.gridCellAttrs['size']);

                if ((that.gridCellsInG.size() === 0) && (that.lassoNodesInG.size() === 0)) {
                    resolve();
                }
            });
        },
        remove: async function() {
            const that = this;
            return new Promise((resolve, reject) => {
                that.gridCellsInG.exit()
                    .transition()
                    .duration(that.removeDuration)
                    .attr('opacity', 0)
                    .remove()
                    .on('end', resolve);

                that.lassoNodesInG.exit()
                    .transition()
                    .duration(that.removeDuration)
                    .attr('opacity', 0)
                    .remove()
                    .on('end', resolve);

                if ((that.gridCellsInG.exit().size() === 0) && (that.lassoNodesInG.exit().size() === 0)) {
                    resolve();
                }
            });
        },
        transform: async function() {
            const that = this;
            return new Promise((resolve, reject) => {
                // compute transform
                const svgRealWidth = that.$refs.gridsvg.clientWidth;
                const svgRealHeight = that.$refs.gridsvg.clientHeight;
                const realSize = Math.min(svgRealWidth, svgRealHeight);
                let shiftx = 0;
                let shifty = 0;
                let scale = 1;
                if (that.svgWidth > realSize) {
                    scale = realSize/that.svgWidth;
                } else {
                    scale = 1;
                }
                shiftx = (svgRealWidth-scale*that.svgWidth)/2;
                shifty = (svgRealHeight-scale*that.svgWidth)/2;
                that.mainG.transition()
                    .duration(that.transformDuration)
                    .attr('transform', `translate(${shiftx} ${shifty}) scale(${scale})`)
                    .on('end', resolve);
            });
        },
        initlasso: function() {
            // Lasso functions
            const that = this;
            const lassoStart = function() {
                lasso.items()
                    .classed('lasso-not-possible', true)
                    .classed('lasso-possible', false);
            };

            const lassoDraw = function() {
                // Style the possible dots
                lasso.possibleItems()
                    .classed('lasso-not-possible', false)
                    .classed('lasso-possible', true);

                // Style the not possible dot
                lasso.notPossibleItems()
                    .classed('lasso-not-possible', true)
                    .classed('lasso-possible', false);
            };

            const lassoEnd = function() {
            // Reset the color of all dots
                lasso.items()
                    .classed('lasso-not-possible', false)
                    .classed('lasso-possible', false);
                const selectednodes = lasso.selectedItems().data();
                if (selectednodes.length>0) {
                    that.zoomin(selectednodes);
                }
                that.stoplasso();
            };

            const lasso = d3.lasso()
                .closePathSelect(true)
                .closePathDistance(100)
                .items(this.lassoNodesInG)
                .targetArea(this.svg)
                .on('start', lassoStart)
                .on('draw', lassoDraw)
                .on('end', lassoEnd);

            this.svg.call(lasso);
        },
        stoplasso: function() {
            this.svg.select('.lasso').remove();
            this.svg.on('.drag', null);
        },
        highlightCells: function(cells) {
            const cellDict = {};
            const that = this;
            for (const cell of cells) cellDict[cell] = true;
            this.gridCellsInG.filter((d) => cellDict[d.index]!==undefined)
                .each(function(d) {
                    that.highlightG.append('rect')
                        .attr('x', (d.grid%that.gridInfo.width)*that.gridCellAttrs['size'])
                        .attr('y', Math.floor(d.grid/that.gridInfo.width)*that.gridCellAttrs['size'])
                        .attr('width', that.gridCellAttrs['size'])
                        .attr('height', that.gridCellAttrs['size'])
                        .attr('stroke', that.gridCellAttrs['stroke'])
                        .attr('stroke-width', 4)
                        .attr('fill', 'none');
                });
        },
        unhighlightCells: function(cells) {
            this.highlightG
                .selectAll('rect')
                .remove();
        },
        setLabelColor: function(labelHierarchy, colors, nodes, labelnames) {
            const hierarchy = cloneDeep(labelHierarchy);
            const root = {
                name: '',
                children: hierarchy,
            };
            // count samples in each class
            const counts = {};
            for (const node of nodes) {
                if (counts[labelnames[node.pred]] === undefined) {
                    counts[labelnames[node.pred]] = 0;
                }
                counts[labelnames[node.pred]]++;
            }
            const dfsCount = function(root, counts) {
                if (typeof(root)==='string') {
                    if (counts[root]===undefined) {
                        counts[root] = 0;
                    }
                    return {
                        name: root,
                        count: counts[root],
                        children: [],
                        realChildren: [],
                        emptyChildren: [],
                    };
                } else {
                    let count = 0;
                    const realChildren = [];
                    const emptyChildren = [];
                    for (let i=0; i<root.children.length; i++) {
                        root.children[i] = dfsCount(root.children[i], counts);
                        count += root.children[i].count;
                        if (root.children[i].count !== 0) {
                            realChildren.push(root.children[i]);
                        } else {
                            emptyChildren.push(root.children[i]);
                        }
                    }
                    // filter out empty nodes
                    root.realChildren = realChildren;
                    root.emptyChildren = emptyChildren;
                    counts[root.name] = count;
                    root.count = count;
                    return root;
                }
            };
            dfsCount(root, counts);
            // set hierarchy color
            const pq = new PriorityQueue({
                'comparator': (a, b)=>{
                    return a.count>b.count?1:(a.count<b.count?-1:0);
                },
            });
            pq.push(root);
            const classThreshold = 10;
            const countThreshold = 0.5;
            const showNodes = {};
            for (const topnode of root.children) {
                showNodes[topnode.name] = colors[topnode.name];
            }
            while (true) {
                const top = pq.top();
                if ((pq.length-1+top.realChildren.length<=classThreshold) || (top.count/root.count>=countThreshold)) {
                    pq.pop();
                    showNodes[top.name] = colors[top.name];
                    for (const child of top.realChildren) {
                        pq.push(child);
                    }
                } else {
                    for (const node of pq.toArray()) {
                        showNodes[node.name] = colors[node.name];
                    }
                    break;
                }
            }
            const hierarchyColors = {};
            const dfsSetColor = function(root, showNodes, hierarchyColors, baseColor) {
                if (showNodes[root.name] !== undefined) {
                    baseColor = showNodes[root.name];
                }
                hierarchyColors[root.name] = baseColor;
                for (const child of root.children) {
                    dfsSetColor(child, showNodes, hierarchyColors, baseColor);
                }
            };
            dfsSetColor(root, showNodes, hierarchyColors);
            this.$store.commit('setHierarchyColors', hierarchyColors);
            this.$store.commit('setShownClass', Object.keys(showNodes));
        },
        createTooltip: function(node) {
            const that = this;
            const getImageGradientURL = this.$store.getters.URL_GET_IMAGE_GRADIENT;
            const tooltip = d3.select('#grid-layout').append('div').attr('class', that.tooltipClass).style('display', 'none');
            return axios.get(getImageGradientURL(node.index, 'origin'))
                .then(function(response) {
                    tooltip.style('display', 'flex');
                    tooltip.html(`<div class="grid-tooltip-info">ID: ${node.index}</div>
                        <div>${that.labelnames[node.label]} -> ${that.labelnames[node.pred]}</div>
                        <div>confidence: ${Math.round(node.confidence*100000)/100000}</div>
                    <img class="gird-tooltip-image" src="${that.toImage(response.data)}"/>
                    <div id="grid-tooltip-arrow" data-popper-arrow></div>`);
                    return tooltip.node();
                });
        },
        removeTooltip: function() {
            d3.selectAll('.'+this.tooltipClass).remove();
        },
    },
    mounted: function() {
        const that = this;
        axios.post(that.URL_GET_GRID, {
            nodes: [],
            depth: 0,
        }).then(function(response) {
            that.nodes = response.data.nodes;
            that.depth = response.data.depth;
            that.gridInfo = response.data.grid;
            if (!that.rendering && that.labelnames.length>0) {
                that.rendering = true;
                that.render();
            }
        });
    },
};
</script>

<style>
#grid-layout {
    width: -moz-calc(100% - 20px);
    width: -webkit-calc(100% - 20px);
    width: -o-calc(100% - 20px);
    width: calc(100% - 20px);
    height: -moz-calc(100% - 20px);
    height: -webkit-calc(100% - 20px);
    height: -o-calc(100% - 20px);
    height: calc(100% - 20px);
    margin: 10px 10px 10px 10px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    position: relative;
}

#grid-drawer {
    width: 100%;
    height: 100%;
    flex-shrink: 100;
}

#grid-icons {
    width: 100%;
    height: 50px;
    display: flex;
    justify-content: flex-start;
    align-items: center;
    margin: 0 0 0 50px;
    flex-shrink: 0;
}

.grid-icon {
    width: 20px;
    height: 20px;
    margin: 0 5px 0 5px;
    cursor: pointer;
}

.lasso-not-possible, .lasso-node {
    fill: none
}

.lasso-possible {
    fill: rgb(200,200,200);
}

.lasso path {
    stroke: rgb(80,80,80);
    stroke-width:2px;
}

.lasso .drawn {
    fill-opacity:.05 ;
}

.lasso .loop_close {
    fill:none;
    stroke-dasharray: 4,4;
}

.lasso .origin {
    fill:#3399FF;
    fill-opacity:.5;
}

.cell-tooltip {
  display: flex;
  flex-direction: column;
  align-items: center;
  background: #ffffff;
  color: gray;
  font-weight: bold;
  padding: 5px 10px;
  font-size: 13px;
  border-radius: 4px;
}

.gird-tooltip-image {
    width: 100px;
    height: 100px;
    margin: 10px 0 0 0;
}

#grid-tooltip-arrow,
#grid-tooltip-arrow::before {
  position: absolute;
  width: 8px;
  height: 8px;
  background: inherit;
}

#grid-tooltip-arrow {
  visibility: hidden;
}

#grid-tooltip-arrow::before {
  visibility: visible;
  content: '';
  transform: rotate(45deg);
}

.cell-tooltip[data-popper-placement^='top'] > #grid-tooltip-arrow {
  bottom: -4px;
}

.cell-tooltip[data-popper-placement^='bottom'] > #grid-tooltip-arrow {
  top: -4px;
}

.cell-tooltip[data-popper-placement^='left'] > #grid-tooltip-arrow {
  right: -4px;
}

.cell-tooltip[data-popper-placement^='right'] > #grid-tooltip-arrow {
  left: -4px;
}
</style>
