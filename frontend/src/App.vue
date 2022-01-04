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
    <router-view></router-view>
  </div>
</template>

<script>
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
            const basecolors = ['#8dd3c7', '#ffffb3', '#fb8072', '#80b1d3',
                '#fdb462', '#b3de69', '#fccde5', '#bc80bd', '#ccebc5', '#ffed6f'];
            const colors = {}; // fill and opacity

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
                } else if (depth>=1) {
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
