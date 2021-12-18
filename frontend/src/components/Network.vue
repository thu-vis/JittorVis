<template>
<div id="network-all" ref="AllNetwork" @resize="setsize">
    <div id="network-tools-bar">
        <svg class="network-tools-bar-icon" @mouseenter="zoomInOpacity=hoverOpacity"
            @mouseleave="zoomInOpacity=iconOpacity" :width="iconSize" :height="iconSize">
            <image xlink:href="/static/images/zoomin.svg" @click="zoomIn" :width="iconSize" :height="iconSize" :opacity="zoomInOpacity"></image>
        </svg>
        <svg class="network-tools-bar-icon" @mouseenter="zoomOutOpacity=hoverOpacity"
            @mouseleave="zoomOutOpacity=iconOpacity" :width="iconSize" :height="iconSize">
            <image xlink:href="/static/images/zoomout.svg" @click="zoomOut" :width="iconSize" :height="iconSize" :opacity="zoomOutOpacity"></image>
        </svg>
        <!--<svg class="network-tools-bar-icon" @mouseenter="splitOpacity=hoverOpacity"
            @mouseleave="splitOpacity=iconOpacity" :width="iconSize" :height="iconSize">
            <image xlink:href="/static/images/split-screen.png"
                @click="splitPage" :width="iconSize" :height="iconSize" :opacity="splitOpacity"></image>
        </svg>-->
    </div>
    <div id="network-main-container">
        <div class="each-network" v-for="networkid in displayNetworkID" :key="networkid" :style="`width: ${width/displayNetworkID.length}px`">
            <svg class="network-tools-bar-icon close-btn" @mouseenter="networkCloseBtnOpacity[networkid]=hoverOpacity"
                @mouseleave="networkCloseBtnOpacity[networkid]=iconOpacity" :width="iconSize" :height="iconSize" v-if="displayNetworkID.length>1">
                <image xlink:href="/static/images/close.png"
                    @click="closePage(networkid)" :width="iconSize" :height="iconSize" :opacity="networkCloseBtnOpacity[networkid]"></image>
            </svg>
            <div class="network-view-all">
                <network-layout :scale="scale" :id="networkBaseID+networkid"></network-layout>
            </div>
        </div>
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
            networkBaseID: 'network-all-',
            displayNetworkID: [0],
            networkCloseBtnOpacity: {0: 0.5},
            hoverOpacity: 1,
            iconOpacity: 0.5,
            zoomInOpacity: 0.5,
            zoomOutOpacity: 0.5,
            splitOpacity: 0.5,
            scale: 1,
            width: 0,
        };
    },
    mounted: function() {
        this.setsize();
    },
    methods: {
        zoomIn: function() {
            this.scale = Math.min(2, this.scale+0.1);
        },
        zoomOut: function() {
            this.scale = Math.max(0.5, this.scale-0.1);
        },
        splitPage: function() {
            const id = Date.now();
            this.displayNetworkID.push(id);
            this.networkCloseBtnOpacity[id] = 0.5;
        },
        closePage: function(id) {
            const idx = this.displayNetworkID.indexOf(id);
            if (idx!==-1) {
                this.displayNetworkID.splice(idx, 1);
                this.networkCloseBtnOpacity[id] = undefined;
            }
        },
        setsize: function() {
            this.width = this.$refs.AllNetwork.clientWidth;
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
    display: flex;
    justify-content: center;
}

.network-tools-bar-icon {
    margin: 0 0 0 5px;
    cursor: pointer;
}

.network-view-all {
    display: flex;
    justify-content: center;
    height: 100%;
    width:100%;
}

.each-network {
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    align-items: center;
}

.close-btn {
    align-self: center;
}
</style>
