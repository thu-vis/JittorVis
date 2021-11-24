<template>
  <div id="app">
    <div id="navigation">
      <div id="navi-title">JittorVis</div>
    </div>
    <div id="content">
      <div id="left">
        <div id="confusion-matrix-container">
          <span>—— Confusion Matrix ——</span>
          <div class="dummy-container">
            <confusion-matrix id="confusion-matrix"></confusion-matrix>
          </div>
        </div>
        <div id="statistic-container">
          <statistic v-for="item in Object.keys(statistic)" :key="item" :dataName="item" :statisticData="statistic[item]"></statistic>
        </div>
      </div>
      <div id="middle">
        <div id="tree-container">
          <div id="network-container"><network></network></div>
        </div>
      </div>
      <div id="featuremap-container">
            <span>—— Features ——</span>
            <vue-scroll :ops="scrollOptions">
              <feature-map></feature-map>
            </vue-scroll>
        </div>
    </div>
  </div>
</template>

<script>
import Network from './components/Network.vue';
import Statistic from './components/Statistic.vue';
import FeatureMap from './components/FeatureMap.vue';
import ConfusionMatrix from './components/ConfusionMatrix.vue';
import {mapGetters} from 'vuex';
import axios from 'axios';

// main vue component
export default {
    components: {Network, Statistic, FeatureMap, ConfusionMatrix},
    name: 'App',
    mounted: function() {
        const store = this.$store;
        axios.get(store.getters.URL_GET_ALL_DATA)
            .then(function(response) {
                store.commit('setAllData', response.data);
                console.log('network data', store.getters.network);
            });
        axios.post(store.getters.URL_GET_CONFUSION_MATRIX)
            .then(function(response) {
                store.commit('setConfusionMatrix', response.data);
                console.log('confusion matrix data', store.getters.confusionMatrix);
            });
    },
    computed: {
        ...mapGetters([
            'statistic',
        ]),
    },
    data: function() {
        return {
            scrollOptions: {
                bar: {
                    background: '#c6bebe',
                },
            },
        };
    },
};
</script>

<style>
html, body, #app {
  margin: 0;
  width: 100%;
  height: 100%;
}

#app {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  display: flex;
  flex-direction: column;
}

#navigation {
  width: 100%;
  height: 50px;
  display: flex;
  align-items: center;
  background: rgb(54, 54, 54);
}

#navi-title {
  color: rgb(255, 255, 255);
  font-weight: 900;
  font-size: 40px;
  margin: 0 0 0 20px;
}

#statistic-container {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
}

#tree-container {
  width: 100%;
  height: 100%;
  display: flex;
  overflow: hidden;
  margin: 5px 5px 0 0;
  border-top: 1px solid lightgray;
}

#featuremap-container {
  width: 40%;
  height: 100%;
  border-left: 1px solid lightgray;
}

#content {
  width: 100%;
  height: 100%;
  overflow: hidden;
  display: flex;
}

#left {
  width: 20%;
  height: 100%;
  display: flex;
  flex-direction: column;
  border-right: 1px solid lightgray;
  padding: 0 10px 0 0;
}

#middle {
  width: 40%;
  height: 100%;
}

#network-container {
  width: 100%;
  height: 100%;
}

#featuremap-container {
  display: flex;
  flex-direction: column;
  align-items: center;
}

#featuremap-container > span, #confusion-matrix-container > span {
  font-family: Lucida Sans Typewriter;
  font-weight: 400;
  margin: 5px 0 5px 0;
}

#confusion-matrix-container {
  position: relative;
  width: 100%;
  padding-top: 100%;
}

#confusion-matrix-container > span {
  position:  absolute;
  text-align: center;
  top: 0;
  left: 0;
  bottom: 0;
  right: 0;
}

.dummy-container {
  position:  absolute;
  top: 0;
  left: 0;
  bottom: 0;
  right: 0;
  width: 100%;
  height: 100%;
  display: flex;
  justify-content: flex-start;
  align-items: flex-start;
}

</style>
