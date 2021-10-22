import Vue from 'vue';
import Vuex from 'vuex';
import clone from 'just-clone';

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
        layoutInfo: {
            layoutNetwork: {}, // very similar to allData.network, with some attributes for layout added
            focusID: '_model/', // default focus node is root node
            t: -1, // a timestamp
        },
        featureMapNodeID: null, // which node to show feature map
    },
    mutations: {
        setAllData(state, allData) {
            state.allData.network = allData.network;
            const newLayoutNetwork = clone(state.allData.network);
            if (newLayoutNetwork==={}) {
                return;
            }
            // init extent
            Object.values(newLayoutNetwork).forEach((d) => {
                d.expand = false;
            });
            // find root
            let root = Object.values(newLayoutNetwork)[0];
            while (root.parent !== undefined) {
                root = newLayoutNetwork[root.parent];
            }
            root.expand = true;
            state.layoutInfo = {
                layoutNetwork: newLayoutNetwork,
                focusID: root.id,
                t: Date.now(),
            };
        },
        setLayoutInfo(state, layoutInfo) {
            state.layoutInfo = layoutInfo;
        },
        setFeatureMapNodeID(state, featureMapNodeID) {
            state.featureMapNodeID = featureMapNodeID;
        },
    },
    getters: {
        network: (state) => state.allData.network,
        statistic: (state) => state.allData.statistic,
        featureMapNodeID: (state) => state.featureMapNodeID,
        layoutInfo: (state) => state.layoutInfo,
        URL_GET_ALL_DATA: (state) => state.APIBASE + '/api/allData',
        URL_GET_FEATURE_INFO: (state) => state.APIBASE + '/api/featureInfo',
        URL_GET_FEATURE: (state) => {
            return (leafID, index) => state.APIBASE + `/api/feature?leafID=${leafID}&index=${index}`;
        },
    },
});
