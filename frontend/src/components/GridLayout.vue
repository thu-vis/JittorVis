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
        hierarchyColors: function(newColors, oldColors) {
            if (!this.rendering && this.nodes.length>0 ) {
                this.render();
            }
        },
    },
    data: function() {
        return {
            nodes: [],
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
            },
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
                        ${Math.floor(d.grid/that.gridInfo.width)*that.gridCellAttrs['size']})`);

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
                    .attr('fill', (d)=>that.hierarchyColors[that.labelnames[d.label]].fill)
                    .attr('opacity', (d)=>that.hierarchyColors[that.labelnames[d.label]].opacity);

                that.lassoNodesInG.enter().append('circle')
                    .attr('class', that.gridCellAttrs['centerClass'])
                    .attr('r', that.gridCellAttrs['centerR'])
                    .attr('cx', (d)=>that.gridCellAttrs['size']/2+(d.grid%that.gridInfo.width)*that.gridCellAttrs['size'])
                    .attr('cy', (d)=>that.gridCellAttrs['size']/2+Math.floor(d.grid/that.gridInfo.width)*that.gridCellAttrs['size']);


                if ((that.gridCellsInG.enter().size() === 0)) {
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
                    .attr('fill', (d)=>that.hierarchyColors[that.labelnames[d.label]].fill)
                    .attr('opacity', (d)=>that.hierarchyColors[that.labelnames[d.label]].opacity)
                    .on('end', resolve);

                if ((that.gridCellsInG.size() === 0)) {
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

                if ((that.gridCellsInG.exit().size() === 0)) {
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
    },
    mounted: function() {
        const that = this;
        window.gridlayout = this;
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
</style>
