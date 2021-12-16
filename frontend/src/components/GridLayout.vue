<template>
    <div id="grid-layout" ref="gridsvg">
        <svg id="grid-drawer">
            <g id="grid-main-g" transform="translate(0,0)">
                <g id="grid-g"></g>
                <g id="lasso-g"></g>
            </g>
        </svg>
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

export default {
    name: 'GridLayout',
    mixins: [Util, GlobalVar],
    computed: {
        ...mapGetters([
            'confusionMatrix',
            'labelHierarchy',
            'labelnames',
            'colors',
            'URL_GET_GRID',
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
    },
    watch: {
        // all info was loaded
        colors: function(newColors, oldColors) {
            const that = this;
            axios.post(that.URL_GET_GRID, {
                nodes: [],
                depth: 0,
            }).then(function(response) {
                that.nodes = response.data.nodes;
                that.depth = response.data.depth;
                that.gridInfo = response.data.grid;
                that.render();
            });
        },
    },
    data: function() {
        return {
            nodes: [],
            depth: 0,
            gridInfo: {},

            //
            gridCellsInG: undefined,
            lassoNodesInG: undefined,

            //
            gridCellAttrs: {
                'gClass': 'grid-cell-in-g',
                'size': 30,
                'stroke-width': '1px',
                'stroke': 'gray',
                'centerR': 3,
                'centerClass': 'lasso-node',
                'centerClassNotSelect': 'lasso-not-possible',
                'centerClassSelect': 'lasso-possible',
            },
        };
    },
    methods: {
        zoomin: function(nodes) {
            if (nodes===undefined) {
                nodes = [];
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
            await this.update();
            await this.transform();
            await this.create();

            this.gridCellsInG = this.girdG.selectAll('.'+this.gridCellAttrs['gClass']);
            this.lassoNodesInG = this.lassoG.selectAll('.'+this.gridCellAttrs['centerClass']);
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
                    .attr('fill', (d)=>that.colors[that.labelnames[d.label]]);

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
                    .attr('fill', (d)=>that.colors[that.labelnames[d.label]])
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

                that.zoomin(lasso.selectedItems().data());
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
}

#grid-drawer {
    width: 100%;
    height: 100%;
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
