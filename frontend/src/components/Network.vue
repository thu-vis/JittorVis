<template>
    <svg id="network-all">
        <g id="network-main">
            <network-layout></network-layout>
        </g>
    </svg>
</template>

<script>
import { mapGetters } from 'vuex'
import NetworkLayout from './NetworkLayout.vue'

export default {
    name: 'network',
    components: {NetworkLayout},
    computed: {
        ...mapGetters([
            'network'
        ])
    }
}
</script>

<style>
#network-all {
    width: 100%;
    height: 100%;
}
</style>
