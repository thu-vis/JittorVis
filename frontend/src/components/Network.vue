<template>
<vue-scroll :ops="scrollOptions">
    <svg id="network-all" ref="networkAll" v-bind:style="{ height: height+'px' }">
        <g id="network-main">
            <network-layout :width="width" v-on:reheight='reheight'></network-layout>
        </g>
    </svg>
</vue-scroll>
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
            height: '100%',
            scrollOptions: {
                bar: {
                    background: '#c6bebe',
                },
            },
        };
    },
    methods: {
        setsize: function() {
            this.width = this.$refs.networkAll.clientWidth;
            this.height = this.$refs.networkAll.clientHeight;
        },
        reheight: function(height) {
            this.height = height;
        },
    },
    mounted: function() {
        this.setsize();
    },
};
</script>

<style>
#network-all {
    width: 100%;
    height: 100%;
}
</style>
