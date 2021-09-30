import Vue from 'vue'
import Vuex from 'vuex'

Vue.use(Vuex)

export default new Vuex.Store({
    state: {
        APIBASE: '//127.0.0.1:5005',
        allData: {
            network: {},
            statistic: {
                loss: [[0, 150], [1, 230], [2, 224], [3, 218], [4, 135], [5, 147], [6, 260]],
                acc: [[0, 150], [1, 230], [2, 224], [3, 218], [4, 135], [5, 147], [6, 260]]
            }
        },
        focusID: '_model/' // default focus node is root node
    },
    mutations: {
        setAllData (state, allData) {
            state.allData.network = allData.network
        },
        setFocusID (state, focusID) {
            state.focusID = focusID
        }
    },
    getters: {
        network: state => state.allData.network,
        statistic: state => state.allData.statistic,
        URL_GET_ALL_DATA: state => state.APIBASE + '/api/allData'
    }
})
