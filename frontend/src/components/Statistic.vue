<template>
    <div style="width: 100%; height: 100%;">
        <div :id="'statistic-chart-'+dataName" class="statistic-chart"></div>
    </div>
</template>

<script>
import * as echarts from 'echarts/core';
import {SVGRenderer} from 'echarts/renderers';
import {LineChart} from 'echarts/charts';
import {
    LegendComponent,
    GridComponent,
    TooltipComponent,
    TitleComponent,
} from 'echarts/components';
import {Select, Option} from 'element-ui';
import Vue from 'vue';

Vue.use(Select);
Vue.use(Option);

echarts.use([
    SVGRenderer,
    LineChart,
    LegendComponent,
    GridComponent,
    TooltipComponent,
    TitleComponent,
]);

export default {
    name: 'statistic',
    components: {
    },
    props: {
        statisticData: {
            default: [],
        },
        dataName: {
            default: '',
        },
    },
    data: function() {
        return {
            statisticChart: null,
        };
    },
    mounted: function() {
        this.statisticChart = echarts.init(document.getElementById('statistic-chart-'+this.dataName));
        this.statisticChart.setOption(this.chartOptions);
    },
    computed: {
        chartOptions: function() {
            const option = {
                title: {
                    text: '—— '+this.dataName+' ——',
                    top: 0,
                    left: 'center',
                    textStyle: {
                        fontWeight: 400,
                        fontFamily: 'Lucida Sans Typewriter',
                    },
                },
                tooltip: {
                    show: true,
                    trigger: 'axis',
                },
                xAxis: {
                    type: 'value',
                },
                yAxis: {
                    type: 'value',
                },
                grid: {
                    left: '35px',
                    right: '10px',
                    top: '40px',
                    bottom: '20px',
                },
                series: [{
                    data: this.statisticData,
                    type: 'line',
                    showSymbol: false,
                    encode: {
                        x: 0,
                        y: 1,
                    },
                    smooth: true,
                }],
            };
            return option;
        },
    },
    methods: {
        changeKey: function(event) {
            this.statisticChart.setOption(this.chartOptions);
        },
    },
};
</script>

<style scoped>
.el-select {
    margin-left: 5px;
}
.el-select >>> .el-input__inner {
    font-size: 5px;
    line-height: 20px;
    height: 20px;
    width: 80px;
    padding-left: 5px;
    padding-right: 15px;
}
.el-select >>> .el-select-dropdown__item {
    font-size: 5px;
}
.el-select >>> .el-input__icon {
    line-height: 20px;
    width: 15px;
}
.el-select >>> .el-input__suffix {
    right: 3px;
}
.statistic-chart {
    width: 100%;
    height: calc(100% - 20px);
}
</style>
