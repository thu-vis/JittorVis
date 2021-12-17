<template>
    <div id="confusionfeaturemaps">
        <confusion-node-map v-for="(cellid, index) in confusionCellIDs" :cell-id="cellid" :key="index"
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
            console.log('set cell', newID, oldID);
            this.confusionCellIDs.push(newID);
        },
        confusionCellIDs: function(newIDs, oldIDs) {
            console.log('cell watched', newIDs, oldIDs);
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
