<template>
    <div class="featurenode">
        <div style="margin-bottom:3px; margin-top:3px">
            <span style="font-size:10px; cursor:pointer; font-family:Comic Sans MS;
                margin-left:3px;">{{ nodeId }}</span>
            <span style="font-size:10px; cursor:pointer; display:inline-flex; float:right;
                margin-right:10px; margin-top:3px;" v-on:click="$emit('delete-id', nodeId)">
                <i class="el-icon-close"></i>
            </span>
            <div style="display:inline-flex; width:50px; float:right; margin-right:10px">
                <el-select class="visselect" :popper-append-to-body="false" v-on:change="changeKey" v-model="curVis" placeholder="origin">
                    <el-option
                        v-for="item in visOptions"
                        :key="item"
                        :label="item"
                        :value="item">
                    </el-option>
                </el-select>
            </div>
        </div>
        <vue-scroll :ops="scrollOptions">
            <div class="featuremaps">
                <img class="featuremap" v-for="url in featureImages" :key="url" :src="url" />
            </div>
        </vue-scroll>
    </div>
</template>

<script>

import axios from 'axios';
import {Select, Option} from 'element-ui';
import Vue from 'vue';
import {mapGetters} from 'vuex';

Vue.use(Select);
Vue.use(Option);

export default {
    name: 'nodemap',
    props: {
        nodeId: String,
    },
    computed: {
        ...mapGetters([
            'layoutInfo',
        ]),
    },
    data: function() {
        return {
            curVis: '',
            visOptions: ['original', 'smooth grad'],
            leafNode: null,
            leafNodeShape: [],
            featureImages: [],
            scrollOptions: {
                bar: {
                    background: '#c6bebe',
                },
            },
        };
    },
    methods: {
        changeKey: function() {
        },
    },
    mounted: function() {
        if (this.nodeId != undefined) {
            const that = this;
            const store = this.$store;
            axios.post(store.getters.URL_GET_FEATURE_INFO, {
                'branch': this.nodeId,
            }).then(function(response) {
                window.response = response;
                that.leafNode = response.data.leafID;
                if (that.leafNode === -1) {
                    that.featureImages = [];
                    return;
                }
                that.leafNodeShape =response.data.shape;
                const getFeature = store.getters.URL_GET_FEATURE;
                if (that.leafNodeShape.length===1) {
                    // linear layer
                    that.featureImages = [getFeature(that.leafNode, -1)];
                } else {
                    // other layer
                    that.featureImages = [];
                    for (let i=0; i<that.leafNodeShape[0]; i++) {
                        that.featureImages.push(getFeature(that.leafNode, i));
                    }
                }
            });
        }
    },
};
</script>

<style scoped>
.featurenode {
    width: 100%;
    max-height: 240px;
    overflow: hidden;
    border-top: 1px solid #aaaaaa;
    border-radius: 2px;
    padding: 3px;
    margin: 5px 10px 5px 2px;
}
#featuremaps {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    max-height: 240px;
}
.featuremap {
    width: 46%;
    margin: 3px 3px 3px 3px;
}
.el-select >>> .el-input__inner {
    font-size: 5px;
    line-height: 15px;
    height: 15px;
    width: 50px;
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
