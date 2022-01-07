<template>
  <div id="app">
    <el-menu
      :default-active="activeRoute"
      :router="true"
      mode="horizontal"
      background-color="#545c64"
      text-color="#fff"
      active-text-color="#ffd04b">
      <li id="navi-title">JittorVis</li>
      <el-menu-item index="/modelview">Model</el-menu-item>
      <el-menu-item index="/dataview">Data</el-menu-item>
    </el-menu>
    <svg width="0" height="0">
        <defs id="texture">
            <pattern v-for="(texture, i) in textures" v-html="texture" :key="i">
            </pattern>
        </defs>
    </svg>
    <router-view></router-view>
  </div>
</template>

<script>
/* eslint-disable max-len */
import ModelView from './components/ModelView.vue';
import DataView from './components/DataView.vue';
import Vue from 'vue';
import VueRouter from 'vue-router';
import {Menu, MenuItem} from 'element-ui';
import axios from 'axios';

Vue.use(Menu);
Vue.use(MenuItem);
Vue.use(VueRouter);

const router = new VueRouter({
    routes: [
        {path: '/modelview', component: ModelView},
        {path: '/dataview', component: DataView},
    ],
});

// main vue component
export default {
    name: 'App',
    data: function() {
        return {
            activeRoute: '/modelview',
            textures: [],
        };
    },
    mounted: function() {
        const store = this.$store;
        const that = this;
        axios.get(store.getters.URL_GET_ALL_DATA)
            .then(function(response) {
                store.commit('setAllData', response.data);
                console.log('network data', store.getters.network);
            });
        axios.post(store.getters.URL_GET_CONFUSION_MATRIX)
            .then(function(response) {
                store.commit('setConfusionMatrix', response.data);
                console.log('confusion matrix data', response.data);
                const colors = that.initColor(store.getters.labelHierarchy);
                store.commit('setColors', colors);
                console.log('colors', store.getters.colors);
                // init hierarchy colors
                const hierarchyColors = {...colors};
                store.commit('setHierarchyColors', hierarchyColors);
            });
        if (this.$route.path === '/') {
            this.$router.push('/modelview');
            this.activeRoute = '/modelview';
        } else {
            this.activeRoute = this.$route.path;
        }
    },
    methods: {
        initColor(hierarchy) {
            const that = this;
            const basecolors = ['#8dd3c7', '#ffffb3', '#fb8072', '#80b1d3',
                '#fdb462', '#b3de69', '#fccde5', '#bc80bd', '#ccebc5', '#ffed6f'];
            const colors = {}; // fill and opacity

            const generateTexture = function(color, index, name) {
                name = name.replaceAll(' ', '-');
                const textures = [
                    `<pattern id="texture-${name}" x="0" y="0" width="100" height="100" patternUnits="userSpaceOnUse" patternTransform="translate(106.51 -16.39)">
                    <rect fill="none" width="100" height="100"></rect>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2.5px;fill:none;" x1="53.03" y1="-53.03" x2="-53.03" y2="53.03"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2.5px;fill:none;" x1="58.03" y1="-48.03" x2="-48.03" y2="58.03"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2.5px;fill:none;" x1="63.03" y1="-43.03" x2="-43.03" y2="63.03"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2.5px;fill:none;" x1="68.03" y1="-38.03" x2="-38.03" y2="68.03"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2.5px;fill:none;" x1="73.03" y1="-33.03" x2="-33.03" y2="73.03"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2.5px;fill:none;" x1="78.03" y1="-28.03" x2="-28.03" y2="78.03"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2.5px;fill:none;" x1="83.03" y1="-23.03" x2="-23.03" y2="83.03"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2.5px;fill:none;" x1="88.03" y1="-18.03" x2="-18.03" y2="88.03"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2.5px;fill:none;" x1="93.03" y1="-13.03" x2="-13.03" y2="93.03"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2.5px;fill:none;" x1="98.03" y1="-8.03" x2="-8.03" y2="98.03"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2.5px;fill:none;" x1="103.03" y1="-3.03" x2="-3.03" y2="103.03"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2.5px;fill:none;" x1="108.03" y1="1.97" x2="1.97" y2="108.03"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2.5px;fill:none;" x1="113.03" y1="6.97" x2="6.97" y2="113.03"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2.5px;fill:none;" x1="118.03" y1="11.97" x2="11.97" y2="118.03"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2.5px;fill:none;" x1="123.03" y1="16.97" x2="16.97" y2="123.03"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2.5px;fill:none;" x1="128.03" y1="21.97" x2="21.97" y2="128.03"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2.5px;fill:none;" x1="133.03" y1="26.97" x2="26.97" y2="133.03"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2.5px;fill:none;" x1="138.03" y1="31.97" x2="31.97" y2="138.03"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2.5px;fill:none;" x1="143.03" y1="36.97" x2="36.97" y2="143.03"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2.5px;fill:none;" x1="148.03" y1="41.97" x2="41.97" y2="148.03"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2.5px;fill:none;" x1="153.03" y1="46.97" x2="46.97" y2="153.03"></line>
                    </pattern>`,

                    `<pattern id="texture-${name}" x="0" y="0" width="100" height="100" patternUnits="userSpaceOnUse">
                    <rect fill="none" width="100" height="100"></rect>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="153.03" y1="53.03" x2="46.97" y2="-53.03"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="147.48" y1="58.59" x2="41.41" y2="-47.48"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="141.92" y1="64.14" x2="35.86" y2="-41.92"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="136.37" y1="69.7" x2="30.3" y2="-36.37"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="130.81" y1="75.26" x2="24.74" y2="-30.81"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="125.26" y1="80.81" x2="19.19" y2="-25.26"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="119.7" y1="86.37" x2="13.63" y2="-19.7"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="114.14" y1="91.92" x2="8.08" y2="-14.14"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="108.59" y1="97.48" x2="2.52" y2="-8.59"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="103.03" y1="103.03" x2="-3.03" y2="-3.03"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="97.48" y1="108.59" x2="-8.59" y2="2.52"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="91.92" y1="114.14" x2="-14.14" y2="8.08"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="86.37" y1="119.7" x2="-19.7" y2="13.63"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="80.81" y1="125.26" x2="-25.26" y2="19.19"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="75.26" y1="130.81" x2="-30.81" y2="24.74"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="69.7" y1="136.37" x2="-36.37" y2="30.3"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="64.14" y1="141.92" x2="-41.92" y2="35.86"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="58.59" y1="147.48" x2="-47.48" y2="41.41"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="53.03" y1="153.03" x2="-53.03" y2="46.97"></line>
                    </pattern>`,

                    `<pattern id="texture-${name}" x="0" y="0" width="100" height="100" patternUnits="userSpaceOnUse" patternTransform="translate(36.8 32.21)">
                    <rect fill="none" width="100" height="100"></rect>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;fill:none" y1="0" x2="100" y2="0"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;fill:none" y1="7.14" x2="100" y2="7.14"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;fill:none" y1="14.29" x2="100" y2="14.29"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;fill:none" y1="21.43" x2="100" y2="21.43"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;fill:none" y1="28.57" x2="100" y2="28.57"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;fill:none" y1="35.71" x2="100" y2="35.71"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;fill:none" y1="42.86" x2="100" y2="42.86"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;fill:none" y1="50" x2="100" y2="50"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;fill:none" y1="57.14" x2="100" y2="57.14"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;fill:none" y1="64.29" x2="100" y2="64.29"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;fill:none" y1="71.43" x2="100" y2="71.43"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;fill:none" y1="78.57" x2="100" y2="78.57"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;fill:none" y1="85.71" x2="100" y2="85.71"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;fill:none" y1="92.86" x2="100" y2="92.86"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;fill:none" y1="100" x2="100" y2="100"></line>
                    </pattern>`,

                    `<pattern id="texture-${name}" x="0" y="0" width="100" height="100" patternUnits="userSpaceOnUse" patternTransform="translate(103.95 40.44)">
                    <rect fill="none" width="100" height="100"></rect>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" y1="125" y2="-25"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="5.56" y1="125" x2="5.56" y2="-25"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="11.11" y1="125" x2="11.11" y2="-25"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="16.67" y1="125" x2="16.67" y2="-25"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="22.22" y1="125" x2="22.22" y2="-25"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="27.78" y1="125" x2="27.78" y2="-25"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="33.33" y1="125" x2="33.33" y2="-25"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="38.89" y1="125" x2="38.89" y2="-25"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="44.44" y1="125" x2="44.44" y2="-25"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="50" y1="125" x2="50" y2="-25"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="55.56" y1="125" x2="55.56" y2="-25"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="61.11" y1="125" x2="61.11" y2="-25"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="66.67" y1="125" x2="66.67" y2="-25"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="72.22" y1="125" x2="72.22" y2="-25"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="77.78" y1="125" x2="77.78" y2="-25"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="83.33" y1="125" x2="83.33" y2="-25"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="88.89" y1="125" x2="88.89" y2="-25"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="94.44" y1="125" x2="94.44" y2="-25"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="100" y1="125" x2="100" y2="-25"></line>
                    </pattern>`,

                    `<pattern id="texture-${name}" x="0" y="0" width="100" height="100" patternUnits="userSpaceOnUse" patternTransform="translate(25.52 -55.86) rotate(-90)">
                    <rect fill="none" width="100" height="100"></rect>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="-25" x2="125"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="-25" y1="5.56" x2="125" y2="5.56"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="-25" y1="11.11" x2="125" y2="11.11"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="-25" y1="16.67" x2="125" y2="16.67"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="-25" y1="22.22" x2="125" y2="22.22"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="-25" y1="27.78" x2="125" y2="27.78"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="-25" y1="33.33" x2="125" y2="33.33"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="-25" y1="38.89" x2="125" y2="38.89"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="-25" y1="44.44" x2="125" y2="44.44"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="-25" y1="50" x2="125" y2="50"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="-25" y1="55.56" x2="125" y2="55.56"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="-25" y1="61.11" x2="125" y2="61.11"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="-25" y1="66.67" x2="125" y2="66.67"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="-25" y1="72.22" x2="125" y2="72.22"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="-25" y1="77.78" x2="125" y2="77.78"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="-25" y1="83.33" x2="125" y2="83.33"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="-25" y1="88.89" x2="125" y2="88.89"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="-25" y1="94.44" x2="125" y2="94.44"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="-25" y1="100" x2="125" y2="100"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" y1="125" y2="-25"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="5.56" y1="125" x2="5.56" y2="-25"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="11.11" y1="125" x2="11.11" y2="-25"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="16.67" y1="125" x2="16.67" y2="-25"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="22.22" y1="125" x2="22.22" y2="-25"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="27.78" y1="125" x2="27.78" y2="-25"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="33.33" y1="125" x2="33.33" y2="-25"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="38.89" y1="125" x2="38.89" y2="-25"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="44.44" y1="125" x2="44.44" y2="-25"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="50" y1="125" x2="50" y2="-25"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="55.56" y1="125" x2="55.56" y2="-25"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="61.11" y1="125" x2="61.11" y2="-25"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="66.67" y1="125" x2="66.67" y2="-25"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="72.22" y1="125" x2="72.22" y2="-25"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="77.78" y1="125" x2="77.78" y2="-25"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="83.33" y1="125" x2="83.33" y2="-25"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="88.89" y1="125" x2="88.89" y2="-25"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="94.44" y1="125" x2="94.44" y2="-25"></line>
                    <line style="stroke:${color};stroke-linecap:round;stroke-miterlimit:10;stroke-width:2px;" x1="100" y1="125" x2="100" y2="-25"></line>
                    </pattern>`,

                    `<pattern id="texture-${name}" x="0" y="0" width="100" height="100" patternUnits="userSpaceOnUse">
                    <rect width="100" height="100" fill="none"></rect>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="-53.03" y1="53.03" x2="53.03" y2="-53.03"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="-47.48" y1="58.59" x2="58.59" y2="-47.48"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="-41.92" y1="64.14" x2="64.14" y2="-41.92"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="-36.37" y1="69.7" x2="69.7" y2="-36.37"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="-30.81" y1="75.26" x2="75.26" y2="-30.81"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="-25.26" y1="80.81" x2="80.81" y2="-25.26"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="-19.7" y1="86.37" x2="86.37" y2="-19.7"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="-14.14" y1="91.92" x2="91.92" y2="-14.14"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="-8.59" y1="97.48" x2="97.48" y2="-8.59"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="-3.03" y1="103.03" x2="103.03" y2="-3.03"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="2.52" y1="108.59" x2="108.59" y2="2.52"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="8.08" y1="114.14" x2="114.14" y2="8.08"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="13.63" y1="119.7" x2="119.7" y2="13.63"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="19.19" y1="125.26" x2="125.26" y2="19.19"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="24.74" y1="130.81" x2="130.81" y2="24.74"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="30.3" y1="136.37" x2="136.37" y2="30.3"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="35.86" y1="141.92" x2="141.92" y2="35.86"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="41.41" y1="147.48" x2="147.48" y2="41.41"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="46.97" y1="153.03" x2="153.03" y2="46.97"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="153.03" y1="53.03" x2="46.97" y2="-53.03"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="147.48" y1="58.59" x2="41.41" y2="-47.48"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="141.92" y1="64.14" x2="35.86" y2="-41.92"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="136.37" y1="69.7" x2="30.3" y2="-36.37"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="130.81" y1="75.26" x2="24.74" y2="-30.81"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="125.26" y1="80.81" x2="19.19" y2="-25.26"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="119.7" y1="86.37" x2="13.63" y2="-19.7"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="114.14" y1="91.92" x2="8.08" y2="-14.14"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="108.59" y1="97.48" x2="2.52" y2="-8.59"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="103.03" y1="103.03" x2="-3.03" y2="-3.03"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="97.48" y1="108.59" x2="-8.59" y2="2.52"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="91.92" y1="114.14" x2="-14.14" y2="8.08"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="86.37" y1="119.7" x2="-19.7" y2="13.63"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="80.81" y1="125.26" x2="-25.26" y2="19.19"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="75.26" y1="130.81" x2="-30.81" y2="24.74"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="69.7" y1="136.37" x2="-36.37" y2="30.3"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="64.14" y1="141.92" x2="-41.92" y2="35.86"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="58.59" y1="147.48" x2="-47.48" y2="41.41"></line>
                    <line style="stroke:${color};stroke-miterlimit:10;stroke-width:2px;" x1="53.03" y1="153.03" x2="-53.03" y2="46.97"></line>
                    </pattern>`,
                ];
                const texture = textures[index];
                // that.textureContainer.append(texture);
                that.textures.push(texture);
                return `url(#texture-${name})`;
            };

            const assignColor = function(node, depth) {
                if (typeof(node)==='string') return;
                const nodename = node.name;
                const basecolor = colors[nodename].fill;
                const baseopacity = colors[nodename].opacity;
                if (depth===0) {
                    const childcnt = node.children.length;
                    let opacity = baseopacity;
                    const minOpacity = baseopacity>0.4?0.4:0;
                    const opacityStep = childcnt>1?(opacity-minOpacity)/(childcnt-1):0;
                    for (const child of node.children) {
                        const childname = typeof(child)==='string'?child:child.name;
                        colors[childname] = {
                            fill: basecolor,
                            opacity: opacity,
                        };
                        opacity -= opacityStep;
                        assignColor(child, depth+1);
                    }
                } else if (depth===1) {
                    for (let i=0; i<node.children.length; i++) {
                        const child = node.children[i];
                        const childname = typeof(child)==='string'?child:child.name;
                        const color = generateTexture(basecolor, i, childname);
                        colors[childname] = {
                            fill: color,
                            opacity: baseopacity,
                        };
                    }
                } else {
                    for (const child of node.children) {
                        const childname = typeof(child)==='string'?child:child.name;
                        colors[childname] = {
                            fill: basecolor,
                            opacity: baseopacity,
                        };
                    }
                }
            };


            for (let i=0; i<hierarchy.length; i++) {
                const nodename = typeof(hierarchy[i])==='string'?hierarchy[i]:hierarchy[i].name;
                colors[nodename] = {
                    fill: basecolors[i],
                    opacity: 1,
                };
                assignColor(hierarchy[i], 0);
            }
            return colors;
        },
    },
    router: router,
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
  margin: 0 50px 0 20px;
  float: left;
}

.router-link {
  text-decoration: none;
}
</style>
