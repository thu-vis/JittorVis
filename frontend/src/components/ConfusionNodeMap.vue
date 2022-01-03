<template>
    <div class="featurenode">
        <div class="featurenode-header" style="margin-bottom:3px; margin-top:3px">
            <span style="font-size:10px; cursor:pointer; font-family:Comic Sans MS;
                margin-left:3px;">{{cellId['class_label']}}->{{cellId['class_pred']}}</span>
                <!-- cellId['labels'] + cellId['preds'] -->
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
            <span style="font-size:10px; cursor:pointer; margin-right:10px; margin-top:3px;" v-on:click="$emit('delete-id', cellId)">
                <i class="el-icon-close"></i>
            </span>
        </div>
        <vue-scroll :ops="scrollOptions">
            <div id="featuremaps">
                <img class="featuremap" v-for="(image, index) in featureImages" :key="index" :src="image" />
            </div>
        </vue-scroll>
    </div>
</template>

<script>

import axios from 'axios';
import {Select, Option, Slider} from 'element-ui';
import Vue from 'vue';
import {mapGetters} from 'vuex';
import Util from './Util.vue';

Vue.use(Select);
Vue.use(Option);
Vue.use(Slider);

export default {
    name: 'confusionnodemap',
    mixins: [Util],
    props: {
        nodeId: String,
        cellId: Object,
    },
    computed: {
        ...mapGetters([
            'layoutInfo',
        ]),
    },
    data: function() {
        return {
            curVis: 'origin',
            visOptions: ['origin', 'vanilla_bp', 'guided_bp', 'grad_cam', 'layer_cam', 'grad_times_image', 'gbp_grad_times_image'],
            featureImages: [],
            images: [],
            scrollOptions: {
                bar: {
                    background: '#c6bebe',
                },
            },
        };
    },
    methods: {
        changeKey: function() {
            this.getFeatureImages();
        },
        clearFeatureImages: function() {
            this.featureImages = [];
        },
        getFeatureImages: async function() {
            this.clearFeatureImages();
            const that = this;
            const images = that.images;
            // const store = that.$store;
            for (let i=0; i<images.length; i++) {
                await that.getSingleFeatureImage(i);
            }
            // 异步
            // if (images.length>0) {
            //     const getImageGradientURL = store.getters.URL_GET_IMAGE_GRADIENT;
            //     for (let i=0; i<images.length; i++) {
            //         console.log(i);
            //         axios.get(getImageGradientURL(images[i], that.curVis))
            //             .then(function(response) {
            //                 console.log('get gradient', response.data);
            //                 that.featureImages.push(that.toImage(response.data));
            //             });
            //     }
            // }
        },
        getSingleFeatureImage: async function(i) {
            const that = this;
            const images = that.images;
            const store = that.$store;
            const getImageGradientURL = store.getters.URL_GET_IMAGE_GRADIENT;
            // console.log(i);
            await axios.get(getImageGradientURL(images[i], that.curVis))
                .then(function(response) {
                    // console.log('get gradient', response.data);
                    that.featureImages.push(that.toImage(response.data));
                    // that.featureImages.push(response.data);
                });
        },
    },
    mounted: function() {
        if (this.cellId != undefined) {
            const that = this;
            const store = that.$store;
            // console.log('cell_id', that.cellId);
            axios.post(store.getters.URL_GET_IMAGES_IN_MATRIX_CELL, {
                'labels': that.cellId['labels'],
                'preds': that.cellId['preds'],
            }).then(function(response) {
                that.images = response.data;
                that.getFeatureImages();
            });
        }
    },
};
</script>

<style scoped>
.featurenode {
    width: 100%;
    max-height: 480px;
    /* overflow: hidden; */
    border-top: 1px solid #aaaaaa;
    border-radius: 2px;
    padding: 3px;
    margin: 5px 10px 5px 2px;
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
    max-height: 480px;
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
