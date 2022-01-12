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
        featureMapNodeID: function(newNodeID, oldNodeID) {
            if (newNodeID===-1) return;
            console.log('setid', newNodeID, oldNodeID);
            this.featureMapNodeIDs.push(newNodeID);
        },
    },
    data: function() {
        return {
            featureMapNodeIDs: [],
        };
    },
    methods: {
        deleteId(id) {
            console.log('delete', id);
            this.$store.commit('setFeatureMapNodeID', -1);
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
    height: 100%;
}
</style>
