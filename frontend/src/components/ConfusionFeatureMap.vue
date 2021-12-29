/* eslint-disable guard-for-in */
<template>
    <div id="confusionfeaturemaps">
        <confusion-node-map v-for="cellid in confusionCellIDs" :cell-id="cellid" :key="idToString(cellid)"
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
        },
    },
    data: function() {
        return {
            leafNode: null,
            leafNodeShape: [],
            confusionCellIDs: [],
        };
    },
    methods: {
        deleteId(id) {
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
