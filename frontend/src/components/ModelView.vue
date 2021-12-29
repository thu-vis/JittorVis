<template>
    <div id="model-content">
      <div id="left">
        <div id="confusion-matrix-container">
          <span>—— Confusion Matrix ——</span>
            <confusion-matrix id="confusion-matrix" @clickCell="clickConfusionCell"></confusion-matrix>
            <div id="image-selector">
              <span>Images in Selected Cells</span>
              <el-select v-model="selectedImage" placeholder="Images" size="mini" @change="runNetworkOnImage">
                <el-option
                  v-for="image in images"
                  :key="image"
                  :label="'image-'+image"
                  :value="image">
                </el-option>
              </el-select>
            </div>
        </div>
        <div id="statistic-container">
          <statistic v-for="item in Object.keys(statistic)" :key="item" :dataName="item" :statisticData="statistic[item]"></statistic>
        </div>
      </div>
      <div id="middle">
        <div id="tree-container">
          <div id="network-container"><network ref="network"></network></div>
        </div>
      </div>
      <div id="right">
        <div id="featuremap-container">
            <span>—— Features ——</span>
            <vue-scroll :ops="scrollOptions">
              <feature-map></feature-map>
            </vue-scroll>
        </div>
        <div id="confusion-featuremap-container">
            <span>——  Confusion Features ——</span>
            <vue-scroll :ops="scrollOptions">
              <confusion-feature-map></confusion-feature-map>
            </vue-scroll>
        </div>
      </div>
    </div>
</template>

<script>
import Network from './Network.vue';
import Statistic from './Statistic.vue';
import FeatureMap from './FeatureMap.vue';
import ConfusionFeatureMap from './ConfusionFeatureMap.vue';
import ConfusionMatrix from './ConfusionMatrix.vue';
import {mapGetters} from 'vuex';
// import axios from 'axios';


export default {
    components: {Network, Statistic, FeatureMap, ConfusionFeatureMap, ConfusionMatrix},
    name: 'ModelView',
    data: function() {
        return {
            scrollOptions: {
                bar: {
                    background: '#c6bebe',
                },
            },
            images: [],
            selectedImage: '',
        };
    },
    computed: {
        ...mapGetters([
            'statistic',
            'URL_RUN_IMAGE_ON_MODEL',
        ]),
    },
    methods: {
        clickConfusionCell: function(d) {
            const that = this;
            that.$store.commit('setConfusionCellID', {labels: d.rowNode.leafs, preds: d.colNode.leafs});
            // console.log(that.$store.state.confusionCellID);
            // axios.post(store.getters.URL_GET_IMAGES_IN_MATRIX_CELL, {
            //     labels: d.rowNode.leafs,
            //     preds: d.colNode.leafs,
            // }).then(function(response) {
            //     const images = response.data;
            //     console.log(`confusion matrix cell ${d.key}`, images);
            //     if (images.length>0) {
            //         const getImageGradientURL = store.getters.URL_GET_IMAGE_GRADIENT;
            //         axios.get(getImageGradientURL(images[0]))
            //             .then(function(response) {
            //                 console.log('get gradient', response.data);
            //             });
            //     }
            // });
        },
        runNetworkOnImage: function(id) {
            if (id==='') return;
            const store = this.$store;
            this.$refs.network.rendering = true;
            axios.post(this.URL_RUN_IMAGE_ON_MODEL, {
                imageID: id,
            }).then(function(response) {
                store.commit('setNetwork', response.data);
                console.log('new network data', store.getters.network);
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

#right {
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
  height: 50%;
  width: 100%;
}

#confusion-featuremap-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  height: 50%;
  width: 100%;
}


#confusion-featuremap-container > span, #featuremap-container > span, #confusion-matrix-container > span {
  border-left: 1px solid lightgray;
}

#confusion-featuremap-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  height: 50%;
  width: 100%;
  border-left: 1px solid lightgray;
}

#confusion-featuremap-container > span, #featuremap-container > span, #confusion-matrix-container > span {
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

#image-selector {
  display: flex;
  align-items: center;
  justify-content: flex-start;
  width: 100%;
}

#image-selector > span {
  font-size: 8px;
  margin: 0 10px 0 10px;
}
</style>
