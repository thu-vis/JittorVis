<template>
    <div id="featuremaps">
        <node-map v-for="nodeid in featureMapNodeIDs" :node-id="nodeid" :key="nodeid"
            v-on:delete-id="deleteId"></node-map>
    </div>
</template>

<script>
import {mapGetters} from 'vuex';
import NodeMap from './NodeMap.vue';

export default {
    components: {NodeMap},
    name: 'featuremap',
    computed: {
        ...mapGetters([
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
        },
    },
    data: function() {
        return {
            leafNode: null,
            leafNodeShape: [],
            featureImages: [],
            featureMapNodeIDs: [],
        };
    },
    methods: {
        deleteId(id) {
            console.log('delete', id);
            const index = this.featureMapNodeIDs.indexOf(id);
            console.log(index);
            if (index !== -1) {
                this.featureMapNodeIDs.splice(index, 1);
            }
        },
    },
};
</script>

<style scoped>
#featuremaps {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
}
</style>
