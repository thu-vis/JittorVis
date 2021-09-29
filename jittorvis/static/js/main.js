function loadStatisticVisWidget(statistics, op_groups, schedule) {
        return VIS.CHART.WIDGET.statisticVisWidget({
            element: d3.select('#StatisticComponent'),
            statistics: statistics,
            op_groups: op_groups,
            schedule: schedule
        });
}

function loadNetworkVisWidget(data) {
        return VIS.CHART.WIDGET.networkVisWidget({
            element: d3.select('#NetworkComponent'),
            data: data
        });
}


$(document).ready(function () {
    let last_resize_time = Date.now();
    $(window).resize(function () {
        var cur_time = Date.now();
        if (cur_time > last_resize_time + REFRESH_PERIOD) {
            window_resize(window.innerWidth, window.innerHeight);
            last_resize_time = cur_time;
        }
    });
    $(document).on("contextmenu", function () {
        return false;
    });

    color_manager = new ColorManager();
    statistic_component = loadStatisticVisWidget({}, [], []);
    network_component = loadNetworkVisWidget({});
    load_data_handler();
    init_interactions();
});