<template>
    <div id="featuremaps">
        <node-map v-for="nodeid in featureMapNodeIDs" :node-id="nodeid" :key="nodeid"
            v-on:delete-id="deleteId"></node-map>
    </div>
</template>

<script>
import {mapGetters} from 'vuex';
// import axios from 'axios';
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
        // featureMapNodeID: function(newNodeID, oldNodeID) {
        //     const that = this;
        //     if (newNodeID != undefined) {
        //         const store = this.$store;
        //         axios.post(store.getters.URL_GET_FEATURE_INFO, {
        //             'branch': newNodeID,
        //         }).then(function(response) {
        //             that.leafNode = response.data.leafID;
        //             if (that.leafNode === -1) {
        //                 that.featureImages = [];
        //                 return;
        //             }
        //             that.leafNodeShape =response.data.shape;
        //             const getFeature = store.getters.URL_GET_FEATURE;
        //             if (that.leafNodeShape.length===1) {
        //                 // linear layer
        //                 that.featureImages = [getFeature(that.leafNode, -1)];
        //             } else {
        //                 // other layer
        //                 that.featureImages = [];
        //                 for (let i=0; i<that.leafNodeShape[0]; i++) {
        //                     that.featureImages.push(getFeature(that.leafNode, i));
        //                 }
        //             }
        //         });
        //     }
        // },
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

.featuremap {
    width: 80px;
    height: 80px;
    margin: 3px 3px 3px 3px;
}
</style>
