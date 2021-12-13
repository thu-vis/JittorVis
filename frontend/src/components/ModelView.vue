<template>
    <div id="model-content">
      <div id="left">
        <div id="confusion-matrix-container">
          <span>—— Confusion Matrix ——</span>
            <confusion-matrix id="confusion-matrix" @clickCell="clickConfusionCell"></confusion-matrix>
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
</template>

<script>
import Network from './Network.vue';
import Statistic from './Statistic.vue';
import FeatureMap from './FeatureMap.vue';
import ConfusionMatrix from './ConfusionMatrix.vue';
import {mapGetters} from 'vuex';
import axios from 'axios';


export default {
    components: {Network, Statistic, FeatureMap, ConfusionMatrix},
    name: 'ModelView',
    data: function() {
        return {
            scrollOptions: {
                bar: {
                    background: '#c6bebe',
                },
            },
        };
    },
    computed: {
        ...mapGetters([
            'statistic',
        ]),
    },
    methods: {
        clickConfusionCell: function(d) {
            const store = this.$store;
            axios.post(store.getters.URL_GET_IMAGES_IN_MATRIX_CELL, {
                labels: d.rowNode.leafs,
                preds: d.colNode.leafs,
            }).then(function(response) {
                const images = response.data;
                console.log(`confusion matrix cell ${d.key}`, images);
                if (images.length>0) {
                    const getImageGradientURL = store.getters.URL_GET_IMAGE_GRADIENT;
                    axios.get(getImageGradientURL(images[0]))
                        .then(function(response) {
                            console.log('get gradient', response.data);
                        });
                }
            });
        },
    },
};
</script>

<style>
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
}

#featuremap-container {
  width: 40%;
  height: 100%;
  border-left: 1px solid lightgray;
}

#model-content {
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

#featuremap-container > span, #confusion-matrix-container > span, #confusion-matrix-container > span {
  font-family: Lucida Sans Typewriter;
  font-weight: 400;
  margin: 5px 0 5px 0;
}

#confusion-matrix-container {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
}

</style>
