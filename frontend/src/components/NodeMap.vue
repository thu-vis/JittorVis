<template>
    <div class="featurenode">
        <div class="featurenode-header" style="margin-bottom:3px; margin-top:3px">
            <span style="font-size:10px; cursor:pointer; font-family:Comic Sans MS;
                margin-left:3px;">{{ nodeId }}</span>
            <el-slider v-show="curVis=='origin'" v-model="actThreshold" :min="0" :max="maxActivation" input-size="mini"
                :step="0.001" style="width: 150px; margin: 0 20px 0 20px;"></el-slider>
            <div style="width: 50px; flex-grow: 100;"></div>
            <div style="display:inline-flex; width:100px; margin-right:10px">
                <el-select class="visselect" :popper-append-to-body="false" v-on:change="changeKey" v-model="curVis" placeholder="origin">
                    <el-option
                        v-for="item in visOptions"
                        :key="item"
                        :label="item"
                        :value="item">
                    </el-option>
                </el-select>
            </div>
            <span style="font-size:10px; cursor:pointer; margin-right:10px; margin-top:3px;" v-on:click="$emit('delete-id', nodeId)">
                <i class="el-icon-close"></i>
            </span>
        </div>
        <vue-scroll :ops="scrollOptions">
            <div id="featuremaps">
                <img class="featuremap" v-for="(image, index) in filteredFeatureImage" :key="index" :src="image" />
            </div>
        </vue-scroll>
        <waiting-icon v-if="rendering"></waiting-icon>
    </div>
</template>

<script>

import axios from 'axios';
import {Select, Option, Slider} from 'element-ui';
import Vue from 'vue';
import {mapGetters} from 'vuex';
import WaitingIcon from './WaitingIcon.vue';

Vue.use(Select);
Vue.use(Option);
Vue.use(Slider);

export default {
    name: 'nodemap',
    components: {WaitingIcon},
    props: {
        nodeId: String,
    },
    computed: {
        ...mapGetters([
            'layoutInfo',
        ]),
        filteredFeatureImage: function() {
            const that = this;
            return this.featureImages.filter((image, index) => {
                return that.featureMaxActivations[index]>=that.actThreshold;
            });
        },
    },
    data: function() {
        return {
            curVis: 'origin',
            visOptions: ['origin', 'discrepancy map'],
            leafNode: null,
            leafNodeShape: [],
            featureImages: [],
            featureMaxActivations: [],
            featureMinActivations: [],
            maxActivation: 1,
            scrollOptions: {
                bar: {
                    background: '#c6bebe',
                },
            },
            featureMapSize: 50,
            actThreshold: 0,
            rendering: true,
        };
    },
    methods: {
        changeKey: function() {
            this.getFeatureImages();
        },
        getFeatureImage: function(featureURL) {
            const that = this;
            axios.get(featureURL)
                .then(function(response) {
                    const featureImage = that.featureMapToImage(response.data);
                    that.featureImages[featureURL] = featureImage;
                });
        },
        featureMapToImage: function(feature, maxv, minv) {
            window.data = feature;
            const canvas = this.featureMapToImage.canvas || (this.featureMapToImage.canvas = document.createElement('canvas'));
            const context = canvas.getContext('2d');
            const height = feature.length;
            const width = feature[0].length;
            canvas.width = width;
            canvas.height = height;
            const image = context.createImageData(width, height);
            const data = image.data;
            window.image = data;
            for (let i=0; i<data.length; i+=4) {
                const v = feature[Math.floor(i/4/width)][(i/4)%width];
                data[i] = v>0?(255-Math.floor(v/maxv*255)):255;
                data[i+1] = v>0?(255-Math.floor(v/maxv*255)):(255-Math.floor(v/minv*255));
                data[i+2] = v>0?255:(255-Math.floor(v/minv*255));
                data[i+3] = 255;
            }
            context.putImageData(image, 0, 0);
            return canvas.toDataURL();
        },
        toImage: function(feature) {
            // return array;
            window.feature = feature;
            const canvas = this.toImage.canvas || (this.toImage.canvas = document.createElement('canvas'));
            const context = canvas.getContext('2d');
            const height = feature.length;
            const width = feature[0].length;
            const depth = feature[0][0].length;
            canvas.width = width;
            canvas.height = height;
            const image = context.createImageData(width, height);
            const data = image.data;
            if (depth==1) {
                for (let i=0; i<data.length; i+=4) {
                    const v = feature[Math.floor(i/4/width)][(i/4)%width];
                    data[i] = v[0];
                    data[i+1] = v[0];
                    data[i+2] = v[0];
                    data[i+3] = 255;
                }
            } else {
                for (let i=0; i<data.length; i+=4) {
                    const v = feature[Math.floor(i/4/width)][(i/4)%width];
                    data[i] = v[0];
                    data[i+1] = v[1];
                    data[i+2] = v[2];
                    data[i+3] = 255;
                }
            }
            context.putImageData(image, 0, 0);
            return canvas.toDataURL();
        },
        getFeatureImages: function() {
            if (this.nodeId != undefined) {
                const that = this;
                const store = this.$store;
                that.rendering = true;
                console.log('imageID', store.getters.selectedImageID);
                axios.post(store.getters.URL_GET_FEATURE_INFO, {
                    'branch': this.nodeId,
                    'method': this.curVis,
                    'imageID': store.getters.selectedImageID,
                }).then(function(response) {
                    that.rendering = false;
                    that.leafNode = response.data.leafID;
                    if (that.leafNode === -1) {
                        that.featureImages = [];
                        return;
                    }
                    that.leafNodeShape = response.data.shape;
                    that.featureMaxActivations = response.data.maxActivations;
                    that.featureMinActivations = response.data.minActivations;
                    if (that.leafNodeShape.length===1) {
                        that.featureImages = [];
                    } else {
                        // other layer
                        const featureMatrixs = response.data.features;
                        // get max/min value to compute d3-scale
                        let maxv = 0.00001;
                        let minv = -0.0001;
                        that.featureMaxActivations.forEach((d) => {
                            maxv = Math.max(maxv, d);
                        });
                        that.maxActivation = maxv;
                        that.featureMinActivations.forEach((d) => {
                            minv = Math.min(minv, d);
                        });
                        console.log(`Feature map max: ${maxv}, min: ${minv}`);
                        console.log('Feature matrix', featureMatrixs);
                        window.featureMatrixs = featureMatrixs;
                        if (that.curVis=='' || that.curVis=='origin') {
                            that.featureImages = featureMatrixs.map((d) => {
                                return that.featureMapToImage(d, maxv, minv);
                            });
                        } else {
                            that.featureImages = featureMatrixs.map((d) => {
                                return that.toImage(d);
                            });
                        }
                    }
                });
            }
        },
    },
    mounted: function() {
        this.getFeatureImages();
    },
};
</script>

<style scoped>
.featurenode {
    width: 100%;
    border-top: 1px solid #aaaaaa;
    border-radius: 2px;
    padding: 3px;
    margin: 5px 10px 5px 2px;
    position: relative;
}

.featurenode-header {
    display: flex;
    justify-content: start;
    align-items: center;
}

#featuremaps {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    align-items: flex-start;
}
.featuremap {
    width: 19%;
    margin: 3px 3px 3px 3px;
}
.el-select >>> .el-input__inner {
    font-size: 5px;
    line-height: 15px;
    height: 15px;
    width: 100px;
    padding-left: 2px;
    padding-right: 15px;
    border: 1px solid #aaaaaa;
    border-radius: 2px;
}
.el-select >>> .el-select-dropdown__item {
    font-size: 5px;
    padding: 0 3px;
    line-height: 20px;
    height: 20px;
}
.el-select >>> .el-popper[x-placement^=bottom] {
    margin-top: 5px;
}
.el-select >>> .el-input__icon {
    line-height: 15px;
    width: 10px;
    font-size: 5px;
}
.el-select >>> .el-input__suffix {
    right: 3px;
}
</style>
