<template>
    <div id="data-content">
        <div id="left">
            <div id="left-confusion-matrix-container">
                <confusion-matrix id="confusion-matrix" @clickCell="clickConfusionCell" :showColor="true"></confusion-matrix>
            </div>
        </div>
        <div id="right">
            <gird-layout ref="grider"></gird-layout>
        </div>
    </div>
</template>

<script>
import ConfusionMatrix from './ConfusionMatrix.vue';
import GirdLayout from './GridLayout.vue';
import axios from 'axios';

export default {
    components: {ConfusionMatrix, GirdLayout},
    name: 'DataView',
    methods: {
        clickConfusionCell: function(d) {
            const store = this.$store;
            const that = this;
            axios.post(store.getters.URL_GET_IMAGES_IN_MATRIX_CELL, {
                labels: d.rowNode.leafs,
                preds: d.colNode.leafs,
            }).then(function(response) {
                const images = response.data;
                if (images.length>0) {
                    axios.post(store.getters.URL_FIND_GRID_PARENT, {
                        children: images,
                        parents: that.$refs.grider.nodes.map((d) => d.index),
                    }).then(function(response) {
                        const parentCells = response.data;
                        console.log('parent cells', parentCells);
                        that.$refs.grider.unhighlightCells();
                        that.$refs.grider.highlightCells(parentCells);
                    });
                }
            });
        },
    },
};
</script>

<style scoped>
#data-content {
    width: 100%;
    height: 100%;
    overflow: hidden;
    display: flex;
}

#left {
    width: 50%;
    height: 100%;
}

#right {
    width: 50%;
    height: 100%;
}

#left-confusion-matrix-container {
  width: 100%;
  height: 100%;
  display: flex;
  margin: 10px 10px 10px 10px;
}
</style>
