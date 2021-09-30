<template>
    <v-chart class="chart" :option="options" />
</template>

<script>
import { mapGetters } from 'vuex'

import { use } from 'echarts/core'
import { SVGRenderer } from 'echarts/renderers'
import { LineChart } from 'echarts/charts'
import {
    LegendComponent,
    GridComponent
} from 'echarts/components'
import VChart from 'vue-echarts'

use([
    SVGRenderer,
    LineChart,
    LegendComponent,
    GridComponent
])

export default {
    name: 'statistic',
    components: {
        VChart
    },
    data: function () {
        return {

        }
    },
    computed: {
        ...mapGetters([
            'statistic'
        ]),
        options: function () {
            let loss = this.statistic.loss
            let option = {
                xAxis: {
                    type: 'value',
                    axisLabel: {
                        interval: 1
                    }
                },
                yAxis: {
                    type: 'value'
                },
                grid: {
                    left: '5%',
                    right: '5%',
                    top: '5%',
                    bottom: '10%'
                },
                series: [{
                    data: loss,
                    type: 'line',
                    encode: {
                        x: 0,
                        y: 1
                    }
                }]
            }
            return option
        }
    }
}
</script>

<style>

</style>
