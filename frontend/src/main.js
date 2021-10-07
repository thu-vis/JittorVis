// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue';
import App from './App.vue';
import store from './store/store.js';
import vuescroll from 'vuescroll/dist/vuescroll-native';
// import the css file
import 'vuescroll/dist/vuescroll.css';
Vue.use(vuescroll);

Vue.config.productionTip = false;

/* eslint-disable no-new */
new Vue({
    el: '#app',
    store: store,
    components: {App},
    template: '<App/>',
});
