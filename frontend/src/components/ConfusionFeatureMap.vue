<<<<<<< HEAD
<template>
    <div id="confusionfeaturemaps">
        <confusion-node-map v-for="nodeid in featureMapNodeIDs" :node-id="nodeid" :key="nodeid"
=======
/* eslint-disable guard-for-in */
<template>
    <div id="confusionfeaturemaps">
        <confusion-node-map v-for="cellid in confusionCellIDs" :cell-id="cellid" :key="idToString(cellid)"
>>>>>>> 2d7bde681034fcd91b47df6d0c8c70a5a4bba485
            v-on:delete-id="deleteId"></confusion-node-map>
    </div>
</template>

<script>
import {mapGetters} from 'vuex';
import ConfusionNodeMap from './ConfusionNodeMap.vue';

export default {
    components: {ConfusionNodeMap},
    name: 'confusionfeaturemap',
    computed: {
        ...mapGetters([
<<<<<<< HEAD
            'featureMapNodeID',
        ]),
    },
    watch: {
        featureMapNodeIDs: function(newNodeIDs, oldNodeIDs) {
            console.log('watched');
        },
        featureMapNodeID: function(newNodeID, oldNodeID) {
            console.log('setid', newNodeID, oldNodeID);
            this.featureMapNodeIDs.push(newNodeID);
=======
            'confusionCellID',
        ]),
    },
    watch: {
        confusionCellID: function(newID, oldID) {
            // console.log('set cell', newID, oldID);
            this.confusionCellIDs.push(newID);
        },
        confusionCellIDs: function(newIDs, oldIDs) {
            // console.log('cell watched', newIDs, oldIDs);
>>>>>>> 2d7bde681034fcd91b47df6d0c8c70a5a4bba485
        },
    },
    data: function() {
        return {
            leafNode: null,
            leafNodeShape: [],
<<<<<<< HEAD
            featureImages: [],
            featureMapNodeIDs: [],
=======
            confusionCellIDs: [],
>>>>>>> 2d7bde681034fcd91b47df6d0c8c70a5a4bba485
        };
    },
    methods: {
        deleteId(id) {
<<<<<<< HEAD
            console.log('delete', id);
            const index = this.featureMapNodeIDs.indexOf(id);
            console.log(index);
            if (index !== -1) {
                this.featureMapNodeIDs.splice(index, 1);
            }
        },
=======
            window.ids = this.confusionCellIDs;
            window.id = id;
            // eslint-disable-next-line guard-for-in
            for (const i in this.confusionCellIDs) {
                const cellid = this.confusionCellIDs[i];
                if (cellid['labels'] == id['labels'] && cellid['preds'] == id['preds']) {
                    this.confusionCellIDs.splice(i, 1);
                    break;
                }
            }
        },
        idToString(id) {
            return id['labels'].join('_') + '-' + id['preds'].join('_');
        },
>>>>>>> 2d7bde681034fcd91b47df6d0c8c70a5a4bba485
    },
};
</script>

<style scoped>
#confusionfeaturemaps {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
}
</style>
