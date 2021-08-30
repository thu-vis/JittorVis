<template>
    <v-chart class="chart" :option="options" />
</template>

<script>
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
    options: function () {
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
          data: [[0, 150], [1, 230], [2, 224], [3, 218], [4, 135], [5, 147], [6, 260]],
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
