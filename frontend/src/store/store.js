import Vue from 'vue';
import Vuex from 'vuex';

Vue.use(Vuex);

export default new Vuex.Store({
    state: {
        APIBASE: '//166.111.80.25:5005',
        allData: {
            network: {},
            statistic: {
                'loss': [[0, 150], [1, 230], [2, 224], [3, 218], [4, 135], [5, 147], [6, 260]],
                'accuracy': [[0, 0.2], [1, 0.3], [2, 0.5], [3, 0.6], [4, 0.92], [5, 0.93], [6, 0.935]],
                'recall': [[0, 250], [1, 230], [2, 224], [3, 218], [4, 135], [5, 147], [6, 260]],
            },
        },
        layoutNetwork: {}, // very similar to allData.network, with some attributes for layout added
        focusID: '_model/', // default focus node is root node
        featureMapNodeID: null, // which node to show feature map
    },
    mutations: {
        setAllData(state, allData) {
            state.allData.network = allData.network;
        },
        setFocusID(state, focusID) {
            if ((state.allData.network[focusID] === undefined) || (state.allData.network[focusID].children.length===0)) {
                return;
            }
            state.focusID = focusID;
        },
        setLayoutNetwork(state, layoutNetwork) {
            state.layoutNetwork = layoutNetwork;
        },
        setFeatureMapNodeID(state, featureMapNodeID) {
            state.featureMapNodeID = featureMapNodeID;
        },
    },
    getters: {
        network: (state) => state.allData.network,
        statistic: (state) => state.allData.statistic,
        featureMapNodeID: (state) => state.featureMapNodeID,
        layoutNetwork: (state) => state.layoutNetwork,
        URL_GET_ALL_DATA: (state) => state.APIBASE + '/api/allData',
        URL_GET_FEATURE_INFO: (state) => state.APIBASE + '/api/featureInfo',
        URL_GET_FEATURE: (state) => {
            return (leafID, index) => state.APIBASE + `/api/feature?leafID=${leafID}&index=${index}`;
        },
    },
});
