import Vue from 'vue'
import Vuex from 'vuex'

Vue.use(Vuex)

export default new Vuex.Store({
    state: {
        APIBASE: '//127.0.0.1:5005',
        allData: {
            network: {},
            statistic: {}
        }
    },
    mutations: {
        setAllData (state, allData) {
            state.allData = allData
        }
    },
    getters: {
        network: state => state.allData.network,
        statistic: state => state.allData.statistic,
        URL_GET_ALL_DATA: state => state.APIBASE + '/api/allData'
    }
})
