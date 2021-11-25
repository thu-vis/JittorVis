<template>
  <div id="app">
    <div id="navigation">
      <div id="navi-title">JittorVis</div>
    </div>
    <model-view></model-view>
  </div>
</template>

<script>
import ModelView from './components/ModelView.vue';
import axios from 'axios';

// main vue component
export default {
    components: {ModelView},
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
</style>
