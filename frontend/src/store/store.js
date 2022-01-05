import Vue from 'vue';
import Vuex from 'vuex';
Vue.use(Vuex);

export default new Vuex.Store({
    state: {
        APIBASE: BACKEND_BASE_URL,
        network: {},
        statistic: {
            'loss': [],
            'accuracy': [],
            'recall': [],
        },
        confusionMatrix: [],
        labelHierarchy: [],
        labelnames: [],
        layoutInfo: {
            layoutNetwork: {}, // very similar to allData.network, with some attributes for layout added
            focusID: '_model/', // default focus node is root node
            t: -1, // a timestamp
        },
        colors: {},
        hierarchyColors: {},
        featureMapNodeID: null, // which node to show feature map
        confusionCellID: null, // which cell clicked ({labels, preds})
    },
    mutations: {
        setAllData(state, allData) {
            state.network = allData.network;
            state.statistic = allData.statistic;
        },
        setNetwork(state, network) {
            state.network = network;
        },
        setLayoutInfo(state, layoutInfo) {
            state.layoutInfo = layoutInfo;
        },
        setFeatureMapNodeID(state, featureMapNodeID) {
            state.featureMapNodeID = featureMapNodeID;
        },
        setConfusionMatrix(state, confusionMatrix) {
            state.confusionMatrix = confusionMatrix.matrix;
            state.labelHierarchy = confusionMatrix.hierarchy;
            state.labelnames = confusionMatrix.names;
        },
        setColors(state, colors) {
            state.colors = colors;
        },
        setConfusionCellID(state, confusionCellID) {
            state.confusionCellID = confusionCellID;
        },
        setConfusionCellID(state, confusionCellID) {
            state.confusionCellID = confusionCellID;
        },
        setHierarchyColors(state, hierarchyColors) {
            state.hierarchyColors = hierarchyColors;
        },
    },
    getters: {
        network: (state) => state.network,
        statistic: (state) => state.statistic,
        featureMapNodeID: (state) => state.featureMapNodeID,
        confusionCellID: (state) => state.confusionCellID,
        layoutInfo: (state) => state.layoutInfo,
        confusionMatrix: (state) => state.confusionMatrix,
        labelHierarchy: (state) => state.labelHierarchy,
        labelnames: (state) => state.labelnames,
        colors: (state) => state.colors,
        hierarchyColors: (state) => state.hierarchyColors,

        URL_GET_ALL_DATA: (state) => state.APIBASE + '/api/allData',
        URL_GET_CONFUSION_MATRIX: (state) => state.APIBASE + '/api/confusionMatrix',
        URL_GET_FEATURE_INFO: (state) => state.APIBASE + '/api/featureInfo',
        URL_GET_FEATURE: (state) => {
            return (leafID, index) => state.APIBASE + `/api/feature?leafID=${leafID}&index=${index}`;
        },
        URL_GET_IMAGES_IN_MATRIX_CELL: (state) => state.APIBASE+'/api/confusionMatrixCell',
        URL_GET_IMAGE_GRADIENT: (state) => {
            return (imageID, method) => state.APIBASE + `/api/imageGradient?imageID=${imageID}&method=${method}`;
        },
        URL_GET_GRID: (state) => state.APIBASE + '/api/grid',
        URL_FIND_GRID_PARENT: (state) => state.APIBASE + '/api/findParent',
        URL_RUN_IMAGE_ON_MODEL: (state) => state.APIBASE + '/api/networkOnImage',
    },
});
