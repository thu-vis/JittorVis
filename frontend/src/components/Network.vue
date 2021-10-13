<template>
<div id="network-all">
    <div id="network-tools-bar">
        <svg class="network-tools-bar-icon" @mouseenter="zoomInOpacity=hoverOpacity"
            @mouseleave="zoomInOpacity=iconOpacity" :width="iconSize" :height="iconSize">
            <image xlink:href="/static/images/zoomin.svg" @click="zoomIn" :width="iconSize" :height="iconSize" :opacity="zoomInOpacity"></image>
        </svg>
        <svg class="network-tools-bar-icon" @mouseenter="zoomOutOpacity=hoverOpacity"
            @mouseleave="zoomOutOpacity=iconOpacity" :width="iconSize" :height="iconSize">
            <image xlink:href="/static/images/zoomout.svg" @click="zoomOut" :width="iconSize" :height="iconSize" :opacity="zoomOutOpacity"></image>
        </svg>
    </div>
    <div id="network-main-container">
        <vue-scroll :ops="scrollOptions">
            <div class="network-view-all">
                <network-layout :scale="scale"></network-layout>
            </div>
        </vue-scroll>
    </div>
</div>
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
            iconSize: 20,
            scrollOptions: {
                bar: {
                    background: '#c6bebe',
                },
            },
            hoverOpacity: 1,
            iconOpacity: 0.5,
            zoomInOpacity: 0.5,
            zoomOutOpacity: 0.5,
            scale: 1,
        };
    },
    methods: {
        zoomIn: function() {
            this.scale = Math.min(2, this.scale+0.1);
        },
        zoomOut: function() {
            this.scale = Math.max(0.5, this.scale-0.1);
        },
    },
};
</script>

<style>
#network-all {
    display: flex;
    flex-direction: column;
    height: 100%;
}

#network-tools-bar {
    width: 100%;
    height: 30px;
    display: flex;
    align-items: center;
    justify-content: center;
}

#network-main-container {
    width: 100%;
    height: 100%;
    overflow: hidden;
}

.network-tools-bar-icon {
    margin: 0 0 0 5px;
    cursor: pointer;
}

.network-view-all {
    display: flex;
    justify-content: center;
}
</style>
