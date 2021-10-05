<template>
    <svg id="network-all" ref="networkAll">
        <g id="network-main">
            <network-layout :width="width" :height="height"></network-layout>
        </g>
    </svg>
</template>

<script>
import {mapGetters} from 'vuex';
import NetworkLayout from './NetworkLayout.vue';

export default {
    name: 'network',
    components: {NetworkLayout},
    computed: {
        ...mapGetters([
            'network',
        ]),
    },
    data: function() {
        return {
            width: 0,
            height: 0,
        };
    },
    methods: {
        resize: function() {
            this.width = this.$refs.networkAll.clientWidth;
            this.height = this.$refs.networkAll.clientHeight;
        },
    },
    mounted: function() {
        this.resize();
    },
};
</script>

<style>
#network-all {
    width: 100%;
    height: 100%;
}
</style>
