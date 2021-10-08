<template>
    <div style="width: 100%; height: 100%;">
        <el-select id="statistic-select" :popper-append-to-body="false" v-on:change="changeKey" v-model="curKey" placeholder="请选择">
            <el-option
                v-for="item in selectOptions"
                :key="item"
                :label="item"
                :value="item">
            </el-option>
        </el-select>
        <div id="statistic-chart"></div>
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
} from 'echarts/components';
import {mapGetters} from 'vuex';
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
]);

export default {
    name: 'statistic',
    components: {
    },
    data: function() {
        return {
            curKey: '',
            statisticChart: null,
        };
    },
    mounted: function() {
        this.statisticChart = echarts.init(document.getElementById('statistic-chart'));
        this.curKey = Object.keys(this.statistic)[0];
        this.statisticChart.setOption(this.chartOptions);
    },
    computed: {
        ...mapGetters([
            'statistic',
        ]),
        selectOptions: function() {
            return Object.keys(this.statistic);
        },
        chartOptions: function() {
            const option = {
                tooltip: {
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
                    top: '10px',
                    bottom: '20px',
                },
                series: [{
                    data: this.statistic[this.curKey],
                    type: 'line',
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
#statistic-chart {
    width: 100%;
    height: calc(100% - 20px);
}
</style>
