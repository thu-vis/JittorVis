var VIS = VIS || {};
VIS.CHART = VIS.CHART || {};
VIS.CHART.WIDGET = VIS.CHART.WIDGET || {};
VIS.CHART.WIDGET.statisticVisWidget = function (options) {

    let defaultVal = {
        margin: { top: 25, right: 55, bottom: 25, left: 30 },
        events:{}
    };

    function StatisticVis(options) {
        let self = this;
        self.element = options.element;
        self.statistics = options.statistics;
        self.op_groups = options.op_groups;
        self.curr_all_schedule = options.schedule;
        self.schedule = [];
        self.settings = $.extend({}, defaultVal, options);
        self.iter_lines = [];
        self.values = [];
        self.selected_value = [];
        self.spring_delta_y = 5;
        self.spring_num = 300;
        self.value_to_label = {};
        self.multi_value_select = false;
        self.used_labels = new Array(30).fill(0);
        self.check_point_id_manager = id_manager();
        self.highlight_op_index_manager = id_manager();
        self.duration = 500;
        self.remove_duration = 200;
        self.op_schedule_checkpoints_width = 8;
        self.focus_op_schedule_checkpoint_index = -1;
        self.focus_op_schedule_range_bar_index = -1;
        self.focus_op_schedule_range_bar_part = '';
        self.op_schedule_range_del_btn_size = 10;
        self.highlight_op_index = {};
        self.focus_op_index = -1;
        self.op_group_highlight_click_manager = click_manager(self.duration);
        self.zoomed_duration = 20;
        self.zoomed_manager = my_throttle(self.zoomed_duration);
        self.row_gap_y = 20;
        self.op_group_circle_r = 6;
        self.show_dependency_op_groups = [];
        self.curr_time_range = [0, 0];
        self.drag_element = undefined;
    }

    StatisticVis.prototype = {
        init: function () {
            this._init_chart();
        },
        redraw:function(statistics, op_groups, schedule){
            let self = this;
            if (statistics || op_groups || schedule){
                self.statistics = statistics;
                self.settings.statistics = statistics;
                self.op_groups = op_groups;
                self.settings.op_groups = op_groups;
                self.curr_all_schedule = schedule;
                self.schedule = [];

                self.settings.schedule = schedule;
                self.highlight_op_index = {};
                self.focus_op_index = -1;
                self.showing_detail_info = -1;

                if (statistics['iter'] !== undefined) {
                    let iter = statistics['iter'];
                    let iter_num = iter.length;

                    self.time_to_x_scale = d3.scaleLinear()
                        .domain([iter[0], iter[iter_num - 1]])
                        .range([self.settings.margin.left, self.settings.margin.left + self.statistic_width]);
                    self.value_to_y_scale = d3.scaleLinear()
                        .domain([0, 1])
                        .range([self.settings.margin.top + self.statistic_height, self.settings.margin.top]);
                    self.time_range_to_spring_num = d3.scaleLinear()
                        .domain([0, iter[iter_num - 1]])
                        .range([0, self.spring_num]);

                    self.iter_lines = iter.map((x, i) => {
                        return {
                            'index': i,
                            'time': x
                        };
                    });
                    self.min_end_time = iter[0]
                    self.max_end_time = iter[iter_num - 1];
                    self.focus_ranges_count = 0;
                    self.iter_ranges = iter.map((x, i) => {
                        let previous_x = i === 0 ? 0 : iter[i - 1];
                        return {
                            'index': i,
                            'time': [previous_x, x],
                            'focus': false
                        }
                    });
                    let values = {};

                    for (let k = 0;k < statistics['value']['iter'].length;k++) {
                        let key = statistics['value']['iter'][k];
                        values[key] = statistics[key].map((x, i) => {
                            return {
                                'index': i,
                                'iter': i,
                                'x': self.time_to_x_scale(self.iter_lines[i]['time']),
                                'y': self.value_to_y_scale(x)
                            };
                        });
                    }
                    for (let k = 0; k < statistics['value']['time'].length;k++) {
                        let key = statistics['value']['time'][k];
                        values[key] = [];
                        for (let i = 0;i < statistics[key].length;i++) {
                            values[key][i] = statistics[key][i].map((x, j) => {
                                return {
                                    'index': j,
                                    'iter': i,
                                    'time': i === 0 ? j: iter[i - 1] + j,
                                    // 'x': i !== 0? liner_scale(iter[i - 1] + j, [iter[i - 1], iter[i]],
                                    //     [self.time_to_x_scale(self.iter_lines[i - 1]['time']) - self.time_to_x_scale(self.iter_lines[i]['time']), 0])
                                    //     : liner_scale(j, [0, iter[i]], [self.time_to_x_scale(0) - self.time_to_x_scale(self.iter_lines[i]['time']), 0]),
                                    'y': self.value_to_y_scale(x)
                                };
                            });
                        }
                    }
                    //console.log("values", values)
                    self.values = values;
                    self.max_op_end_time = 0;
                    self.op_groups.forEach(function (op_group) {
                        if (op_group.end_time > self.max_op_end_time) {
                            self.max_op_end_time = op_group.end_time;
                        }
                    });

                    let op_schedule_spring_path_data = [{'start_time': 0}];
                    self.op_schedule_checkpoints.forEach(function (d, i) {
                        op_schedule_spring_path_data[i].end_time = d.time;
                        op_schedule_spring_path_data.push({
                            'start_time': d.time
                        });
                    });
                    op_schedule_spring_path_data[op_schedule_spring_path_data.length - 1]['end_time'] = self.max_op_end_time;
                    self.op_schedule_spring_path_data = op_schedule_spring_path_data;
                    let op_schedule_range_data = [];
                    for (let i = 0;i < self.op_schedule_checkpoints.length;i += 2) {
                        op_schedule_range_data.push({
                            index: self.op_schedule_checkpoints[i].id,
                            start_time: self.op_schedule_checkpoints[i].time,
                            end_time: self.op_schedule_checkpoints[i + 1].time
                        });
                    }
                    self.op_schedule_range_data = op_schedule_range_data;
                    self.curr_time_range = [0, self.max_op_end_time];
                    let time_slices = [0], x_slices = [self.settings.margin.left + 6];
                    time_slices = time_slices.concat(self.op_schedule_checkpoints.map(check_point=>check_point.time));
                    time_slices.push(self.max_op_end_time);

                    x_slices = x_slices.concat(self.op_schedule_checkpoints.map(check_point=>check_point.x));
                    x_slices.push(self.settings.margin.left + self.statistic_width);

                    self.op_group_time_to_x_scale = d3.scaleLinear()
                        .domain(time_slices)
                        .range(x_slices);
                    self.op_group_x_to_time_scale = d3.scaleLinear()
                        .domain(x_slices)
                        .range(time_slices);

                    self.op_group_detail_time_to_x_scale = d3.scaleLinear()
                        .domain(self.curr_time_range)
                        .range([self.settings.margin.left + 6, self.settings.margin.left + self.statistic_width]);

                    self.op_schedule_spring_path_data.forEach(function (d) {
                        d.start_x = self.op_group_time_to_x_scale(d.start_time);
                        d.end_x = self.op_group_time_to_x_scale(d.end_time);
                    });

                    self.op_schedule_range_data.forEach(function (d) {
                        d.start_x = self.op_group_time_to_x_scale(d.start_time);
                        d.end_x = self.op_group_time_to_x_scale(d.end_time);
                    });
                    // self.schedule = process_connected_op_groups(self.op_groups, self.curr_all_schedule, self.op_group_time_to_x_scale);
                    self.schedule = process_op_groups(self.op_groups, self.curr_all_schedule);
                }
                self._init_overview();
                self._generate_chart();
            }
        },
        resize:function(){
            let self = this;
            self.chart_width = WINDOW_WIDTH - 20;
            self.chart_height = 100;
            self.overview_chart_height = 50;
            self.statistic_width = self.chart_width - self.settings.margin.left - self.settings.margin.right;
            self.statistic_height = self.chart_height + self.overview_chart_height - self.settings.margin.top - self.settings.margin.bottom;
            self.op_group_overview_height = self.overview_chart_height - self.settings.margin.top;
            self.op_group_detail_height = self.chart_height - self.settings.margin.bottom;
            self.chart.attr('width', self.chart_width)
                        .attr('height', self.chart_height + self.overview_chart_height);

            let statistics = self.statistics;
            if (statistics['iter'] !== undefined) {
                let iter = statistics['iter'];
                let iter_num = iter.length;

                self.time_to_x_scale = d3.scaleLinear()
                    .domain([0, iter[iter_num - 1]])
                    .range([self.settings.margin.left, self.settings.margin.left + self.statistic_width]);
                self.value_to_y_scale = d3.scaleLinear()
                    .domain([0, 1])
                    .range([self.settings.margin.top + self.statistic_height, self.settings.margin.top]);
                self.time_range_to_spring_num = d3.scaleLinear()
                    .domain([0, iter[iter_num - 1]])
                    .range([0, self.spring_num]);

                self.iter_lines = iter.map((x, i) => {
                    return {
                        'index': i,
                        'time': x
                    };
                });
                self.max_end_time = iter[iter_num - 1];
                self.focus_ranges_count = 0;
                self.iter_ranges = iter.map((x, i) => {
                    let previous_x = i === 0 ? 0 : iter[i - 1];
                    return {
                        'index': i,
                        'time': [previous_x, x],
                        'focus': false
                    }
                });
                let values = {};

                for (let k = 0;k < statistics['value']['iter'].length;k++) {
                    let key = statistics['value']['iter'][k];
                    values[key] = statistics[key].map((x, i) => {
                        return {
                            'index': i,
                            'iter': i,
                            'x': self.time_to_x_scale(self.iter_lines[i]['time']),
                            'y': self.value_to_y_scale(x)
                        };
                    });
                }
                for (let k = 0; k < statistics['value']['time'].length;k++) {
                    let key = statistics['value']['time'][k];
                    values[key] = [];
                    for (let i = 0;i < statistics[key].length;i++) {
                        values[key][i] = statistics[key][i].map((x, j) => {
                            return {
                                'index': j,
                                'iter': i,
                                'time': i === 0? j: iter[i - 1] + j,
                                // 'x': i !== 0? liner_scale(iter[i - 1] + j, [iter[i - 1], iter[i]],
                                //     [self.time_to_x_scale(self.iter_lines[i - 1]['time']) - self.time_to_x_scale(self.iter_lines[i]['time']), 0])
                                //     : liner_scale(j, [0, iter[i]], [self.time_to_x_scale(0) - self.time_to_x_scale(self.iter_lines[i]['time']), 0]),
                                'y': self.value_to_y_scale(x)
                            };
                        });
                    }
                }
                self.values = values;
                self.max_op_end_time = 0;
                self.op_groups.forEach(function (op_group) {
                    if (op_group.end_time > self.max_op_end_time) {
                        self.max_op_end_time = op_group.end_time;
                    }
                });

                let op_schedule_spring_path_data = [{'start_time': 0}];
                self.op_schedule_checkpoints.forEach(function (d, i) {
                    op_schedule_spring_path_data[i].end_time = d.time;
                    op_schedule_spring_path_data.push({
                        'start_time': d.time
                    });
                });
                op_schedule_spring_path_data[op_schedule_spring_path_data.length - 1]['end_time'] = self.max_op_end_time;
                self.op_schedule_spring_path_data = op_schedule_spring_path_data;

                let time_slices = [0], x_slices = [self.settings.margin.left + 6];
                time_slices = time_slices.concat(self.op_schedule_checkpoints.map(check_point=>check_point.time));
                time_slices.push(self.max_op_end_time);

                x_slices = x_slices.concat(self.op_schedule_checkpoints.map(check_point=>check_point.x));
                x_slices.push(self.settings.margin.left + self.statistic_width);

                self.op_group_time_to_x_scale = d3.scaleLinear()
                    .domain(time_slices)
                    .range(x_slices);
                self.op_group_x_to_time_scale = d3.scaleLinear()
                    .domain(x_slices)
                    .range(time_slices);

                self.op_schedule_spring_path_data.forEach(function (d) {
                    d.start_x = self.op_group_time_to_x_scale(d.start_time);
                    d.end_x = self.op_group_time_to_x_scale(d.end_time);
                });
                // self.schedule = process_connected_op_groups(self.op_groups, self.curr_all_schedule, self.op_group_time_to_x_scale);
                self.schedule = process_op_groups(self.op_groups, self.curr_all_schedule);
            }
            self._generate_chart();
        },
        _init_chart: function(){
            let self = this;
            self.chart_width = WINDOW_WIDTH - 20;
            self.chart_height = 100;
            self.overview_chart_height = 50;
            self.statistic_width = self.chart_width - self.settings.margin.left - self.settings.margin.right;
            self.statistic_height = self.chart_height + self.overview_chart_height - self.settings.margin.top - self.settings.margin.bottom;
            self.op_group_overview_height = self.overview_chart_height - self.settings.margin.top;
            self.op_group_detail_height = self.chart_height - self.settings.margin.bottom;
            self.mouse_pos = {
                x: 0,
                y: 0
            };
            self.info_rect_width = 0;
            self.chart = self.element
                .append('svg')
                .attr('id','statisticView')
                .attr('width', self.chart_width)
                .attr('height', self.chart_height + self.overview_chart_height);
            self.chart
                .on("mousedown", function () {
                    if (self.mode === 'op_schedule') {
                        if(d3.event.which === 1) {
                            if (self.checkpoints_update_state === 'none') {
                                if (self.focus_op_schedule_checkpoint_index !== -1) {
                                    for (let i = 0;i < self.op_schedule_checkpoints.length;i++) {
                                        if (self.op_schedule_checkpoints[i].id === self.focus_op_schedule_checkpoint_index) {
                                            self.mouse_down_element.index = i;
                                            self.mouse_down_element.element_type = 'checkpoint';
                                            self.mouse_down_element.part = '';
                                            break;
                                        }
                                    }
                                }
                                else if (self.focus_op_schedule_range_bar_index !== -1) {
                                    self.mouse_down_element.index = self.focus_op_schedule_range_bar_index;
                                    self.mouse_down_element.element_type = 'range_bar';
                                    self.mouse_down_element.part = self.focus_op_schedule_range_bar_part;
                                }
                                else {
                                    self.mouse_down_element.index = -1;
                                    self.mouse_down_element.element_type = '';
                                    self.mouse_down_element.part = '';
                                }
                            }
                            else if (self.checkpoints_update_state === 'add') {
                                if (self.temp_op_schedule_range.start_x !== self.temp_op_schedule_range.end_x) {
                                    let new_check_points = [];
                                    let i = 0;
                                    while (i < self.op_schedule_checkpoints.length) {
                                        if (self.op_schedule_checkpoints[i].x < self.temp_op_schedule_range.start_x) {
                                            new_check_points.push(self.op_schedule_checkpoints[i]);
                                        }
                                        else {
                                            break;
                                        }
                                        i++;
                                    }
                                    new_check_points.push({
                                        'time': self.op_group_x_to_time_scale(self.temp_op_schedule_range.start_x),
                                        'x': self.temp_op_schedule_range.start_x,
                                        'id': self.check_point_id_manager.get_id()
                                    });
                                    new_check_points.push({
                                        'time': self.op_group_x_to_time_scale(self.temp_op_schedule_range.end_x),
                                        'x': self.temp_op_schedule_range.end_x,
                                        'id': self.check_point_id_manager.get_id()
                                    });
                                    while (i < self.op_schedule_checkpoints.length) {
                                        new_check_points.push(self.op_schedule_checkpoints[i]);
                                        i++;
                                    }

                                    self.op_schedule_checkpoints = new_check_points;

                                    let time_slices = [0], x_slices = [self.settings.margin.left + 6];
                                    time_slices = time_slices.concat(self.op_schedule_checkpoints.map(check_point=>check_point.time));
                                    time_slices.push(self.max_op_end_time);

                                    x_slices = x_slices.concat(self.op_schedule_checkpoints.map(check_point=>check_point.x));
                                    x_slices.push(self.settings.margin.left + self.statistic_width);

                                    self.op_group_time_to_x_scale = d3.scaleLinear()
                                        .domain(time_slices)
                                        .range(x_slices);
                                    self.op_group_x_to_time_scale = d3.scaleLinear()
                                        .domain(x_slices)
                                        .range(time_slices);

                                    let op_schedule_spring_path_data = [{'start_time': 0}];
                                    self.op_schedule_checkpoints.forEach(function (d, i) {
                                        op_schedule_spring_path_data[i].end_time = d.time;
                                        op_schedule_spring_path_data.push({
                                            'start_time': d.time
                                        });
                                    });
                                    op_schedule_spring_path_data[op_schedule_spring_path_data.length - 1]['end_time'] = self.max_op_end_time;
                                    self.op_schedule_spring_path_data = op_schedule_spring_path_data;

                                    self.op_schedule_spring_path_data.forEach(function (d) {
                                        d.start_x = self.op_group_time_to_x_scale(d.start_time);
                                        d.end_x = self.op_group_time_to_x_scale(d.end_time);
                                    });

                                    let op_schedule_range_data = [];
                                    for (let i = 0;i < self.op_schedule_checkpoints.length;i += 2) {
                                        op_schedule_range_data.push({
                                            index: self.op_schedule_checkpoints[i].id,
                                            start_time: self.op_schedule_checkpoints[i].time,
                                            end_time: self.op_schedule_checkpoints[i + 1].time
                                        });
                                    }
                                    self.op_schedule_range_data = op_schedule_range_data;

                                    self.op_schedule_range_data.forEach(function (d) {
                                        d.start_x = self.op_group_time_to_x_scale(d.start_time);
                                        d.end_x = self.op_group_time_to_x_scale(d.end_time);
                                    });

                                    // self.schedule = process_connected_op_groups(self.op_groups, self.curr_all_schedule, self.op_group_time_to_x_scale);
                                    self.schedule = process_op_groups(self.op_groups, self.curr_all_schedule);
                                    self._generate_chart();
                                }
                            }
                        }
                    }
                })
                .on("mousemove", function () {
                    self.mouse_pos = {
                        x: d3.event.offsetX,
                        y: d3.event.offsetY
                    };
                    let x = self.mouse_pos.x + 15;
                    if (self.statistic_width - x < self.info_rect_width && x > self.statistic_width / 2) {
                        x -= self.info_rect_width + 30;
                    }
                    self.op_schedule_info_group
                        .attr('transform', `translate(${x}, 0)`);
                    self.focus_op_schedule_range_bar_index = -1;
                    if (self.focus_op_schedule_checkpoint_index === -1 && self.settings.margin.top + self.statistic_height - self.op_bar_height <= d3.event.offsetY
                        && d3.event.offsetY <= self.settings.margin.top + self.statistic_height) {
                        let i = 0;
                        while (i < self.op_schedule_range_data.length) {
                            if (self.op_schedule_range_data[i].start_x < d3.event.offsetX
                                && d3.event.offsetX < self.op_schedule_range_data[i].end_x) {
                                self.focus_op_schedule_range_bar_index = i;
                                break;
                            }
                            i++;
                        }
                    }

                    let i = 0;
                    if (self.checkpoints_update_state === 'none') {
                        while (i < self.op_schedule_range_data.length) {
                            if (self.op_schedule_range_data[i].start_x - 5 < d3.event.offsetX
                                && d3.event.offsetX < self.op_schedule_range_data[i].end_x + 5) {
                                break;
                            }
                            i++;
                        }
                        self.op_schedule_background_group.selectAll('.op_schedule_range_bar_del_btn_g')
                                .style('opacity', 0);
                        if (i !== self.op_schedule_range_data.length) {
                            self.op_schedule_background_group.selectAll(`#op_schedule_range_bar_del_btn_g_${self.op_schedule_range_data[i]['index']}`)
                                .style('opacity', 1);
                        }
                    }

                    i = 0;
                    while (i < self.op_schedule_range_data.length) {
                        if (self.op_schedule_range_data[i].start_x - 20 < d3.event.offsetX
                            && d3.event.offsetX < self.op_schedule_range_data[i].end_x + 20) {
                            break;
                        }
                        i++;
                    }
                    if (i === self.op_schedule_range_data.length) {
                        self.temp_op_schedule_range.start_x = d3.event.offsetX - 10;
                        self.temp_op_schedule_range.end_x = d3.event.offsetX + 10;
                        let temp_data = [{
                                id: 0,
                                x: self.temp_op_schedule_range.start_x
                            }, {
                                id: 1,
                                x: self.temp_op_schedule_range.end_x
                            }];
                        self.op_schedule_checkpoints_group.selectAll('.op_schedule_temp_checkpoints_path').data(temp_data, function (d) {
                            return d.id;
                        })
                            .attr('d', function (d) {
                                return path_d([
                                    [d.x,
                                        self.settings.margin.top + self.op_group_overview_height],
                                    [d.x,
                                        self.settings.margin.top + self.statistic_height - self.op_bar_height]]);
                            })
                            .style('opacity', function (d) {
                                if (self.checkpoints_update_state === 'add') {
                                    return 1;
                                }
                                return 0;
                            });


                        self.op_schedule_background_group
                            .selectAll('.op_schedule_temp_range_background_rect')
                            .data([self.temp_op_schedule_range])
                            .attr('x', function (d) {
                                return d.start_x;
                            })
                            .attr('y', self.settings.margin.top)
                            .attr('width', function (d) {
                                return d.end_x - d.start_x;
                            })
                            .attr('height', self.op_group_detail_height)
                            .style('opacity', function (d) {
                                if (self.checkpoints_update_state === 'add') {
                                    return 1;
                                }
                                return 0;
                            });

                        self.op_schedule_background_group
                            .selectAll('.op_schedule_temp_range_bar_rect')
                            .data([self.temp_op_schedule_range])
                            .attr('x', function (d) {
                                return d.start_x;
                            })
                            .attr('y', self.settings.margin.top + self.op_group_detail_height - self.op_bar_height)
                            .attr('width', function (d) {
                                return d.end_x - d.start_x;
                            })
                            .attr('height', self.op_bar_height)
                            .style('opacity', function (d) {
                                if (self.checkpoints_update_state === 'add') {
                                    return 0.9;
                                }
                                return 0;
                            });
                    }
                    else {
                        self.temp_op_schedule_range.start_x = 0;
                        self.temp_op_schedule_range.end_x = 0;
                        let temp_data = [{
                                id: 0,
                                x: self.temp_op_schedule_range.start_x
                            }, {
                                id: 1,
                                x: self.temp_op_schedule_range.end_x
                            }];
                        self.op_schedule_checkpoints_group.selectAll('.op_schedule_temp_checkpoints_path').data(temp_data, function (d) {
                            return d.id;
                        })
                            .style('opacity', function (d) {
                                return 0;
                            });

                        self.op_schedule_background_group
                            .selectAll('.op_schedule_temp_range_background_rect')
                            .data([self.temp_op_schedule_range])
                            .style('opacity', 0);

                        self.op_schedule_background_group
                            .selectAll('.op_schedule_temp_range_bar_rect')
                            .data([self.temp_op_schedule_range])
                            .style('opacity', 0);
                    }
                    if (self.drag_element !== undefined) {
                        if (self.drag_element.element === 'left_top') {
                            self.curr_time_range[0] = Math.min(self.curr_time_range[1] - 1, Math.max(0, self.op_group_x_to_time_scale(self.mouse_pos.x)));
                        }
                        else if (self.drag_element.element === 'right_top') {
                            self.curr_time_range[1] = Math.max(self.curr_time_range[0] + 1, Math.min(self.max_op_end_time, self.op_group_x_to_time_scale(self.mouse_pos.x)));
                        }
                        else {
                            let curr_time = self.op_group_x_to_time_scale(self.mouse_pos.x);
                            let del_time = self.curr_time_range[1] - self.curr_time_range[0];
                            curr_time = Math.max(self.drag_element.time, Math.min(self.max_op_end_time - del_time + self.drag_element.time, curr_time));
                            self.curr_time_range[0] = curr_time - self.drag_element.time;
                            self.curr_time_range[1] = self.curr_time_range[0] + del_time;
                        }
                        self._update_curr_time_range();
                    }

                })
                .on("mouseleave", function () {
                    let temp_data = [{
                            id: 0,
                            x: self.temp_op_schedule_range.start_x
                        }, {
                            id: 1,
                            x: self.temp_op_schedule_range.end_x
                        }];
                    self.op_schedule_checkpoints_group.selectAll('.op_schedule_temp_checkpoints_path').data(temp_data, function (d) {
                        return d.id;
                    })
                        .style('opacity', function (d) {
                            return 0;
                        });

                    self.op_schedule_background_group
                        .selectAll('.op_schedule_temp_range_background_rect')
                        .data([self.temp_op_schedule_range])
                        .style('opacity', 0);

                    self.op_schedule_background_group
                        .selectAll('.op_schedule_temp_range_bar_rect')
                        .data([self.temp_op_schedule_range])
                        .style('opacity', 0);
                    if (self.checkpoints_update_state === 'none') {
                        self.op_schedule_background_group.selectAll('.op_schedule_range_bar_del_btn_g')
                                    .style('opacity', 0);
                    }
                    self.temp_op_schedule_range.start_x = - 100;
                    self.temp_op_schedule_range.end_x = -100;
                })
                .on("click", function () {
                    if (self.mode === 'op_schedule') {
                        self.drag_element = undefined;
                    }
                });


            self.statistics_group = self.chart.append('g').attr('class','statistics_group');
            self.op_schedule_group = self.chart.append('g').attr('class','op_schedule_group')
                .attr('transform', `translate(${self.chart_width},0)`)
                .style('opacity', 0);
            self.op_schedule_background_group = self.op_schedule_group.append('g').attr('class','op_schedule_background_group').attr('transform', `translate(0,${self.settings.margin.top + self.op_group_overview_height})`);
            self.op_schedule_node_group = self.op_schedule_group.append('g').attr('class','op_schedule_node_group').attr('transform', `translate(0,${self.settings.margin.top + self.op_group_overview_height})`);
            self.op_schedule_static_group = self.op_schedule_group.append('g').attr('class','op_schedule_static_group');
            self.op_schedule_info_group = self.op_schedule_group.append('g').attr('class','op_schedule_info_group')
                .style('opacity', 0).attr('transform', 'translate(0,0)');
            self.op_schedule_node_overview_group = self.op_schedule_group.append('g').attr('class','op_schedule_node_overview_group');
            self.op_schedule_node_overview_cover_group = self.op_schedule_node_overview_group.append('g').attr('class','op_schedule_node_overview_cover_group');

            self.op_schedule_checkpoints_group = self.op_schedule_group.append('g').attr('class','op_schedule_checkpoints_group');
            self.op_schedule_spring_group = self.op_schedule_group.append('g').attr('class','op_schedule_spring_group');
            self.op_schedule_dependency_group = self.op_schedule_group.append('g').attr('class','op_schedule_dependency_group');
            self.op_schedule_info_group.append('rect')
                .attr('class', 'op_schedule_info_text')
                .attr('id', 'op_schedule_info_background')
                .attr('x', 0)
                .attr('y', 0)
                .attr('width', 0)
                .attr('height', 0)
                .style('fill', 'white')
                .style('stroke', color_manager.default_color)
                .style('stroke-width', 1);
            self.op_schedule_info_group.append('text')
                .attr('class', 'op_schedule_info_text')
                .attr('id', 'index-info')
                .text('index:')
                .attr('dominant-baseline', 'hanging')
                .attr('x', 0)
                .attr('y', 0);
            self.op_schedule_info_group.append('text')
                .attr('class', 'op_schedule_info_text')
                .attr('id', 'file_path-info')
                .text('file_path:')
                .attr('dominant-baseline', 'hanging')
                .attr('x', 0)
                .attr('y', 0);
            self.op_schedule_info_group.append('text')
                .attr('class', 'op_schedule_info_text')
                .attr('id', 'jit_key-info')
                .text('jit_key:')
                .attr('dominant-baseline', 'hanging')
                .attr('x', 0)
                .attr('y', 0);


            self.op_schedule_node_overview_group.append('path')
                .attr('class', 'op_schedule_overview_border')
                .attr('id', 'op_schedule_overview_border')
                .attr('d', line([{
                    x: self.settings.margin.left, y: self.overview_chart_height
                }, {
                    x: self.chart_width, y: self.overview_chart_height
                }]))
                .style('stroke', color_manager.text_color)
                .style('stroke-width', '1px')
                .style('fill', 'none');

            self.op_schedule_node_overview_cover_group.append('rect')
                .attr('class', 'op_schedule_overview_cover_rect')
                .attr('id', 'op_schedule_overview_cover_rect_left')
                .attr('x', self.settings.margin.left + 6)
                .attr('y', 0)
                .attr('width', 0)
                .attr('height', self.overview_chart_height)
                .style('stroke', 'none')
                .style('fill', 'black')
                .style('opacity', 0.3);

            self.op_schedule_node_overview_cover_group.append('rect')
                .attr('class', 'op_schedule_overview_cover_rect')
                .attr('id', 'op_schedule_overview_cover_rect_right')
                .attr('x', self.settings.margin.left + self.statistic_width)
                .attr('y', 0)
                .attr('width', 0)
                .attr('height', self.overview_chart_height)
                .style('stroke', 'none')
                .style('fill', 'black')
                .style('opacity', 0.3);

            self.op_schedule_node_overview_cover_group.append('path')
                .attr('class', 'op_schedule_overview_path')
                .attr('id', 'op_schedule_overview_path_left')
                .attr('d', line([{
                    x: self.settings.margin.left + 6, y: 0
                }, {
                    x: self.settings.margin.left + 6, y: self.overview_chart_height
                }]))
                .style('stroke', color_manager.text_color)
                .style('stroke-width', '1px')
                .style('fill', 'none');

            self.op_schedule_node_overview_cover_group.append('path')
                .attr('class', 'op_schedule_overview_path')
                .attr('id', 'op_schedule_overview_path_right')
                .attr('d', line([{
                    x: self.settings.margin.left + self.statistic_width, y: 0
                }, {
                    x: self.settings.margin.left + self.statistic_width, y: self.overview_chart_height
                }]))
                .style('stroke', color_manager.text_color)
                .style('stroke-width', '1px')
                .style('fill', 'none');

            self.op_schedule_node_overview_cover_group.append('rect')
                .attr('class', 'op_schedule_overview_cover_rect')
                .attr('id', 'op_schedule_overview_cover_rect_top')
                .attr('x', self.settings.margin.left + 6)
                .attr('y', 0)
                .attr('width', self.statistic_width - 6)
                .attr('height', 10)
                .style('stroke', 'none')
                .style('fill', 'transparent')
                .attr('cursor', "grab")
                .style('opacity', 1)
                .on("mousedown", function () {
                    if (self.mode === 'op_schedule') {
                        if(d3.event.which === 1) {
                            self.drag_element = {
                                element: 'top',
                                time: self.op_group_x_to_time_scale(self.mouse_pos.x) - self.curr_time_range[0]
                            };
                        }
                    }
                })
                .on("click", function () {
                    if (self.mode === 'op_schedule') {
                        self.drag_element = undefined;
                    }
                });

            self.op_schedule_node_overview_cover_group.append('rect')
                .attr('class', 'op_schedule_overview_cover_rect')
                .attr('id', 'op_schedule_overview_cover_rect_left_top')
                .attr('x', self.settings.margin.left + 6 - 2.5)
                .attr('y', 0)
                .attr('width', 5)
                .attr('height', 10)
                .style('stroke', 'white')
                .style('fill', 'black')
                .attr('cursor', "pointer")
                .style('opacity', 1)
                .on("mousedown", function () {
                    if (self.mode === 'op_schedule') {
                        if(d3.event.which === 1) {
                            self.drag_element = {
                                element: 'left_top',
                                time: self.op_group_x_to_time_scale(self.mouse_pos.x)
                            };
                        }
                    }
                })
                .on("click", function () {
                    if (self.mode === 'op_schedule') {
                        self.drag_element = undefined;
                    }
                });

            self.op_schedule_node_overview_cover_group.append('rect')
                .attr('class', 'op_schedule_overview_cover_rect')
                .attr('id', 'op_schedule_overview_cover_rect_right_top')
                .attr('x', self.settings.margin.left + self.statistic_width - 2.5)
                .attr('y', 0)
                .attr('width', 5)
                .attr('height', 10)
                .style('stroke', 'white')
                .style('fill', 'black')
                .attr('cursor', "pointer")
                .style('opacity', 1)
                .on("mousedown", function () {
                    if (self.mode === 'op_schedule') {
                        if(d3.event.which === 1) {
                            self.drag_element = {
                                element: 'right_top',
                                time: self.op_group_x_to_time_scale(self.mouse_pos.x)
                            };
                        }
                    }
                })
                .on("click", function () {
                    if (self.mode === 'op_schedule') {
                        self.drag_element = undefined;
                    }
                });

            self.legend_group = self.chart.append('g').attr('class','legend_group');
            self.legend_hide = false;
            self.mode = 'statistics';
            self.op_schedule_checkpoints = [];
            self.op_schedule_spring_path_data = [];
            self.temp_op_schedule_range = {
                start_x: 0,
                end_x: 0
            };
            self.op_schedule_range_data = [];
            self.checkpoints_update_state = 'none';

            self.max_op_end_time = 0;
            self.op_bar_height = 10;
            self.mouse_down_element = {
                index: -1,
                element_type: '',
                part: ''
            };

            self.transform = {
                x: 0,
                y: 0,
                k: 1
            };
            self.zoom = d3.zoom()
                            .on("zoom", zoomed);

            function zoomed() {
                if (d3.event.transform.k === self.transform.k) {
                    if (self.mouse_down_element.index !== -1) {
                        let index = self.mouse_down_element.index;
                        if (self.mouse_down_element.element_type === 'checkpoint') {
                            self.op_schedule_checkpoints[index].x += d3.event.transform.x - self.transform.x;
                            if (index > 0) {
                                self.op_schedule_checkpoints[index].x = Math.max(self.op_schedule_checkpoints[index].x,
                                    self.op_schedule_checkpoints[index - 1].x + self.op_schedule_checkpoints_width);
                            }
                            else {
                                self.op_schedule_checkpoints[index].x = Math.max(self.op_schedule_checkpoints[index].x,
                                    self.op_schedule_checkpoints_width);
                            }
                            if (index < self.op_schedule_checkpoints.length - 1) {
                                self.op_schedule_checkpoints[index].x = Math.min(self.op_schedule_checkpoints[index].x,
                                    self.op_schedule_checkpoints[index + 1].x - self.op_schedule_checkpoints_width);
                            }
                            else {
                                self.op_schedule_checkpoints[index].x = Math.min(self.op_schedule_checkpoints[index].x,
                                    self.settings.margin.left + self.statistic_width - self.op_schedule_checkpoints_width);
                            }
                            self.op_schedule_spring_path_data[index].end_x = self.op_schedule_checkpoints[index].x;
                            self.op_schedule_spring_path_data[index + 1].start_x = self.op_schedule_checkpoints[index].x;
                            let range_index = Math.floor(index / 2);
                            if (index % 2 === 0) {
                                self.op_schedule_range_data[range_index].start_x = self.op_schedule_checkpoints[index].x;
                            }
                            else {
                                self.op_schedule_range_data[range_index].end_x = self.op_schedule_checkpoints[index].x;
                            }
                        }
                        else if (self.mouse_down_element.element_type === 'range_bar') {
                            if (self.mouse_down_element.part === 'center') {
                                let index = self.mouse_down_element.index;
                                let delta_x = d3.event.transform.x - self.transform.x;
                                let spring_data = self.op_schedule_spring_path_data[index * 2];
                                if (delta_x > 0) {
                                    spring_data = self.op_schedule_spring_path_data[index * 2 + 2];
                                }
                                let range_data = self.op_schedule_range_data[index];
                                let delta_time = (spring_data.end_time - spring_data.start_time) / (spring_data.end_x - spring_data.start_x) * delta_x;

                                let max_time = self.max_op_end_time,
                                    min_time = 0;
                                if (index > 0) {
                                    min_time = self.op_schedule_range_data[index - 1].end_time;
                                }
                                if (index < self.op_schedule_range_data.length - 1) {
                                    max_time = self.op_schedule_range_data[index + 1].start_time;
                                }

                                if (range_data.end_time + delta_time > max_time - 0.1) {
                                    delta_time = max_time - 0.1 - range_data.end_time;
                                }
                                if (range_data.start_time + delta_time < min_time + 0.1) {
                                    delta_time = min_time + 0.1 - range_data.start_time;
                                }

                                delta_x = delta_time / (spring_data.end_time - spring_data.start_time) * (spring_data.end_x - spring_data.start_x);

                                self.op_schedule_checkpoints[index * 2].x += delta_x;
                                self.op_schedule_checkpoints[index * 2].time += delta_time;

                                self.op_schedule_checkpoints[index * 2 + 1].x += delta_x;
                                self.op_schedule_checkpoints[index * 2 + 1].time += delta_time;

                                self.op_schedule_spring_path_data[index * 2].end_x = self.op_schedule_checkpoints[index * 2].x;
                                self.op_schedule_spring_path_data[index * 2 + 1].start_x = self.op_schedule_checkpoints[index * 2].x;
                                self.op_schedule_spring_path_data[index * 2 + 1].end_x = self.op_schedule_checkpoints[index * 2 + 1].x;
                                self.op_schedule_spring_path_data[index * 2 + 2].start_x = self.op_schedule_checkpoints[index * 2 + 1].x;

                                self.op_schedule_spring_path_data[index * 2].end_time = self.op_schedule_checkpoints[index * 2].time;
                                self.op_schedule_spring_path_data[index * 2 + 1].start_time = self.op_schedule_checkpoints[index * 2].time;
                                self.op_schedule_spring_path_data[index * 2 + 1].end_time = self.op_schedule_checkpoints[index * 2 + 1].time;
                                self.op_schedule_spring_path_data[index * 2 + 2].start_time = self.op_schedule_checkpoints[index * 2 + 1].time;

                                self.op_schedule_range_data[index].start_x += delta_x;
                                self.op_schedule_range_data[index].end_x += delta_x;
                                self.op_schedule_range_data[index].start_time += delta_time;
                                self.op_schedule_range_data[index].end_time += delta_time;
                            }
                            else if (self.mouse_down_element.part === 'left') {
                                let index = self.mouse_down_element.index;
                                let delta_x = d3.event.transform.x - self.transform.x;
                                let range_data = self.op_schedule_range_data[index];

                                let min_time = 0,
                                    min_x = self.settings.margin.left + 6;
                                if (index > 0) {
                                    min_x = self.op_schedule_range_data[index - 1].end_x;
                                    min_time = self.op_schedule_range_data[index - 1].end_time;
                                }

                                if (range_data.start_x + delta_x < min_x + 20) {
                                    delta_x = min_x + 20 - range_data.start_x;
                                }

                                let delta_time = (range_data.end_time - range_data.start_time) / (range_data.end_x - range_data.start_x) * delta_x;

                                if (range_data.start_time + delta_time < min_time + 0.1) {
                                    delta_time = min_time + 0.1 - range_data.start_time;
                                }

                                delta_x = delta_time / (range_data.end_time - range_data.start_time) * (range_data.end_x - range_data.start_x);

                                self.op_schedule_checkpoints[index * 2].x += delta_x;
                                self.op_schedule_checkpoints[index * 2].time += delta_time;

                                self.op_schedule_spring_path_data[index * 2].end_x = self.op_schedule_checkpoints[index * 2].x;
                                self.op_schedule_spring_path_data[index * 2 + 1].start_x = self.op_schedule_checkpoints[index * 2].x;

                                self.op_schedule_spring_path_data[index * 2].end_time = self.op_schedule_checkpoints[index * 2].time;
                                self.op_schedule_spring_path_data[index * 2 + 1].start_time = self.op_schedule_checkpoints[index * 2].time;

                                self.op_schedule_range_data[index].start_x += delta_x;
                                self.op_schedule_range_data[index].start_time += delta_time;
                            }
                            else if (self.mouse_down_element.part === 'right') {
                                let index = self.mouse_down_element.index;
                                let delta_x = d3.event.transform.x - self.transform.x;
                                let range_data = self.op_schedule_range_data[index];

                                let max_time = self.max_op_end_time,
                                    max_x = self.settings.margin.left + self.statistic_width;
                                if (index < self.op_schedule_range_data.length - 1) {
                                    max_x = self.op_schedule_range_data[index + 1].start_x;
                                    max_time = self.op_schedule_range_data[index + 1].start_time;
                                }

                                if (range_data.end_x + delta_x > max_x - 20) {
                                    delta_x = max_x - 20 - range_data.end_x;
                                }

                                let delta_time = (range_data.end_time - range_data.start_time) / (range_data.end_x - range_data.start_x) * delta_x;

                                if (range_data.end_time + delta_time > max_time - 0.1) {
                                    delta_time = max_time - 0.1 - range_data.end_time;
                                }

                                delta_x = delta_time / (range_data.end_time - range_data.start_time) * (range_data.end_x - range_data.start_x);

                                self.op_schedule_checkpoints[index * 2 + 1].x += delta_x;
                                self.op_schedule_checkpoints[index * 2 + 1].time += delta_time;

                                self.op_schedule_spring_path_data[index * 2 + 1].end_x = self.op_schedule_checkpoints[index * 2 + 1].x;
                                self.op_schedule_spring_path_data[index * 2 + 2].start_x = self.op_schedule_checkpoints[index * 2 + 1].x;

                                self.op_schedule_spring_path_data[index * 2 + 1].end_time = self.op_schedule_checkpoints[index * 2 + 1].time;
                                self.op_schedule_spring_path_data[index * 2 + 2].start_time = self.op_schedule_checkpoints[index * 2 + 1].time;

                                self.op_schedule_range_data[index].end_x += delta_x;
                                self.op_schedule_range_data[index].end_time += delta_time;
                            }
                        }


                        let time_slices = [0], x_slices = [self.settings.margin.left + 6];
                        time_slices = time_slices.concat(self.op_schedule_checkpoints.map(check_point=>check_point.time));
                        time_slices.push(self.max_op_end_time);

                        x_slices = x_slices.concat(self.op_schedule_checkpoints.map(check_point=>check_point.x));
                        x_slices.push(self.settings.margin.left + self.statistic_width);

                        self.op_group_time_to_x_scale = d3.scaleLinear()
                            .domain(time_slices)
                            .range(x_slices);
                        self.op_group_x_to_time_scale = d3.scaleLinear()
                            .domain(x_slices)
                            .range(time_slices);

                        // self.schedule = process_connected_op_groups(self.op_groups, self.curr_all_schedule, self.op_group_time_to_x_scale);
                        self.schedule = process_op_groups(self.op_groups, self.curr_all_schedule);
                    }
                }
                else if (self.focus_op_schedule_range_bar_index !== -1) {
                    let index = self.focus_op_schedule_range_bar_index;
                    let delta_k = d3.event.transform.k / self.transform.k;
                    let spring_rect_data = self.op_schedule_spring_path_data[index];
                    let delta_time = (spring_rect_data.end_time - spring_rect_data.start_time) * (delta_k - 1) / delta_k;
                    if (spring_rect_data.end_time - spring_rect_data.start_time - delta_time < 0.1) {
                        delta_time = spring_rect_data.end_time - spring_rect_data.start_time - 0.1;
                    }
                    let delta_times = [delta_time / 2, -delta_time / 2];


                    if (spring_rect_data.end_time + delta_times[1] > self.max_op_end_time - 0.1) {
                        delta_times[1] = self.max_op_end_time - 0.1 - spring_rect_data.end_time;
                        delta_times[0] = -delta_times[1];
                    }
                    if (spring_rect_data.start_time + delta_times[0] < 0.1) {
                        delta_times[0] = 0.1 - spring_rect_data.start_time;
                        delta_times[1] = -delta_times[0];
                    }

                    let delta_scale = d3.scaleLinear()
                            .domain([0, spring_rect_data.start_time, spring_rect_data.end_time, self.max_op_end_time])
                            .range([0, spring_rect_data.start_time + delta_times[0], spring_rect_data.end_time + delta_times[1], self.max_op_end_time]);

                    self.op_schedule_checkpoints.forEach(function (d) {
                        d.time = delta_scale(d.time);
                    });
                    self.op_schedule_spring_path_data.forEach(function (d) {
                        d.start_time = delta_scale(d.start_time);
                        d.end_time = delta_scale(d.end_time);
                    });

                    let time_slices = [0], x_slices = [self.settings.margin.left + 6];
                    time_slices = time_slices.concat(self.op_schedule_checkpoints.map(check_point=>check_point.time));
                    time_slices.push(self.max_op_end_time);

                    x_slices = x_slices.concat(self.op_schedule_checkpoints.map(check_point=>check_point.x));
                    x_slices.push(self.settings.margin.left + self.statistic_width);

                    self.op_group_time_to_x_scale = d3.scaleLinear()
                        .domain(time_slices)
                        .range(x_slices);
                    self.op_group_x_to_time_scale = d3.scaleLinear()
                        .domain(x_slices)
                        .range(time_slices);

                    // self.schedule = process_connected_op_groups(self.op_groups, self.curr_all_schedule, self.op_group_time_to_x_scale);
                    self.schedule = process_op_groups(self.op_groups, self.curr_all_schedule);
                }
                self.zoomed_manager.run(()=>{
                    self._generate_chart_for_zoomed();
                });

                self.transform = d3.event.transform;
            }
            self.chart.call(self.zoom);

            // y axis
            self.statistics_group.append('path')
                .attr('class', 'axis-y')
                .attr('id', 'axis-y-0')
                .attr('d', path_d([
                    [self.settings.margin.left,
                        3],
                    [self.settings.margin.left,
                        self.statistic_height + self.settings.margin.top]]))
                .style('stroke', color_manager.text_color)
                .style('stroke-width', '2px');
                
            self.statistics_group.append('path')
                .attr('class', 'axis-y')
                .attr('id', 'axis-y-1')
                .attr('d', path_d([
                    [self.settings.margin.left,
                        self.settings.margin.top],
                    [self.settings.margin.left + 6,
                        self.settings.margin.top]]))
                .style('stroke', color_manager.text_color)
                .style('stroke-width', '2px')
                
            self.statistics_group.append('path')
                .attr('class', 'axis-y')
                .attr('id', 'axis-y-2')
                .attr('d', path_d([
                    [self.settings.margin.left,
                        self.statistic_height + self.settings.margin.top],
                    [self.settings.margin.left + 6,
                        self.statistic_height + self.settings.margin.top]]))
                .style('stroke', color_manager.text_color)
                .style('stroke-width', '2px');
            self.statistics_group.append('path')
                .attr('id', 'axis-y-arrow')
                .attr('d', arrow_path_d(self.settings.margin.left - 3, 3, 6, 9, 'top'))
                .style('stroke', color_manager.text_color)
                .style('fill', color_manager.text_color)
                .style('stroke-width', '2px');
            self.statistics_group.append('text')
                .attr('id', 'axis-y-label')
                .text(self.selected_value.length > 0? self.selected_value[0]: '')
                .attr('x', self.settings.margin.left + 6)
                .attr('y', 0)
                .style("user-select", "none")
                .attr('font-size', '15px')
                .attr('font-family', 'Arial')
                .attr('dominant-baseline', 'hanging')
                .style('fill', color_manager.text_color);

            // x axis
            self.statistics_group.append('path')
                .attr('id', 'op_schedule_axis-x')
                .attr('d', path_d([
                    [self.statistic_width + self.settings.margin.left,
                        self.statistic_height + self.settings.margin.top],
                    [self.statistic_width + self.settings.margin.left + 20,
                        self.statistic_height + self.settings.margin.top]]))
                .style('stroke', color_manager.text_color)
                .style('stroke-width', '2px');
            self.statistics_group.append('path')
                .attr('id', 'op_schedule_axis-x-arrow')
                .attr('d', arrow_path_d(self.statistic_width + self.settings.margin.left + 11, self.statistic_height + self.settings.margin.top - 3, 9, 6, 'right'))
                .style('stroke', color_manager.text_color)
                .style('fill', color_manager.text_color)
                .style('stroke-width', '2px');
            self.statistics_group.append('text')
                .attr('id', 'op_schedule_axis-x-label')
                .text('Iteration')
                .attr('x', self.settings.margin.left + self.statistic_width - 10)
                .attr('y', self.settings.margin.top + self.statistic_height - 15)
                .style("user-select", "none")
                .attr('font-size', '15px')
                .attr('font-family', 'Arial')
                .attr('dominant-baseline', 'central')
                .style('fill', color_manager.text_color);

            self.op_schedule_static_group.append('rect')
                .attr('id', `op_schedule_outside_cover`)
                .attr('x', 0)
                .attr('y', self.overview_chart_height)
                .attr('width', self.settings.margin.left)
                .attr('height', self.chart_height)
                .style('stroke', 'white')
                .style('stroke-width', '1px')
                .style('fill', 'white');

            // y axis
            self.op_schedule_static_group.append('path')
                .attr('class', 'axis-y')
                .attr('id', 'axis-y-3')
                .attr('d', path_d([
                    [self.settings.margin.left,
                        3],
                    [self.settings.margin.left,
                        self.statistic_height + self.settings.margin.top]]))
                .style('stroke', color_manager.text_color)
                .style('stroke-width', '2px');
            // self.op_schedule_group.append('path')
            //     .attr('class', 'axis-y')
            //     .attr('d', path_d([
            //         [self.settings.margin.left,
            //             self.settings.margin.top],
            //         [self.settings.margin.left + 6,
            //             self.settings.margin.top]]))
            //     .style('stroke', color_manager.text_color)
            //     .style('stroke-width', '2px');
            self.op_schedule_static_group.append('path')
                .attr('id', 'op_schedule_axis-y-arrow')
                .attr('d', arrow_path_d(self.settings.margin.left - 3, 3, 6, 9, 'top'))
                .style('stroke', color_manager.text_color)
                .style('fill', color_manager.text_color)
                .style('stroke-width', '2px');
            self.op_schedule_static_group.append('text')
                .attr('id', 'op_schedule_axis-y-label')
                .attr('x', self.settings.margin.left + 6)
                .attr('y', 0)
                .text('gpu id')
                .style("user-select", "none")
                .attr('font-size', '15px')
                .attr('font-family', 'Arial')
                .attr('dominant-baseline', 'hanging')
                .style('fill', color_manager.text_color);

            // x axis
            self.op_schedule_static_group.append('path')
                .attr('id', 'op_schedule_axis-x')
                .attr('d', path_d([
                    [self.statistic_width + self.settings.margin.left,
                        self.statistic_height + self.settings.margin.top],
                    [self.statistic_width + self.settings.margin.left + 20,
                        self.statistic_height + self.settings.margin.top]]))
                .style('stroke', color_manager.text_color)
                .style('stroke-width', '2px');

            self.op_schedule_static_group.append('path')
                .attr('id', 'op_schedule_axis-x-arrow')
                .attr('d', arrow_path_d(self.statistic_width + self.settings.margin.left + 11, self.statistic_height + self.settings.margin.top - 3, 9, 6, 'right'))
                .style('stroke', color_manager.text_color)
                .style('fill', color_manager.text_color)
                .style('stroke-width', '2px');

            self.op_schedule_static_group.append('text')
                .attr('id', 'op_schedule_axis-x-label')
                .text('Iteration')
                .attr('x', self.settings.margin.left + self.statistic_width - 10)
                .attr('y', self.settings.margin.top + self.statistic_height - 15)
                .style("user-select", "none")
                .attr('font-size', '15px')
                .attr('font-family', 'Arial')
                .attr('dominant-baseline', 'central')
                .style('fill', color_manager.text_color);

            self.op_schedule_static_group.append('text')
                .attr('id', 'op_schedule_iter_label')
                .attr('x', self.settings.margin.left + self.statistic_width)
                .attr('y', 0)
                .text('Iteration 0')
                .style("user-select", "none")
                .attr('font-size', '15px')
                .attr('font-family', 'Arial')
                .attr('dominant-baseline', 'hanging')
                .style('text-anchor', 'end')
                .style('fill', color_manager.text_color);

            self.op_schedule_spring_group.append('rect')
                .attr('id', `op_schedule_spring_background`)
                .attr('x', self.settings.margin.left + 6)
                .attr('y', self.settings.margin.top + self.statistic_height + 3)
                .attr('width', self.statistic_width - 6)
                .attr('height', self.spring_delta_y * 2 + 20)
                .style('stroke', 'white')
                .style('stroke-width', '1px')
                .style('fill', 'white');

            self.op_schedule_spring_group.append('path')
                .attr('id', `op_schedule_spring_path`)
                .style('stroke', color_manager.default_color)
                .style('stroke-width', '1px')
                .style('fill', 'none');

            self.statistics_group.append('path')
                .attr('id', `spring_path`)
                .style('stroke', color_manager.text_color)
                .style('stroke-width', '2px')
                .style('fill', 'none');

            self.op_schedule_background_group.append('rect')
                .attr('id', 'op_schedule_background_rect')
                .attr('x', self.settings.margin.left)
                .attr('y', self.settings.margin.top)
                .attr('width', self.statistic_width)
                .attr('height', self.op_group_detail_height)
                .style('opacity', 0)
                .style('fill', color_manager.default_color);

            let temp_data = [{
                            id: 0,
                            x: self.temp_op_schedule_range.start_x
                        }, {
                            id: 1,
                            x: self.temp_op_schedule_range.end_x
                        }];

            let op_schedule_temp_checkpoints_path = self.op_schedule_checkpoints_group
                .selectAll('.op_schedule_temp_checkpoints_path')
                .data(temp_data, function (d) {
                return d.id;
            });

            op_schedule_temp_checkpoints_path.enter()
                .append('path')
                .attr('class', 'op_schedule_temp_checkpoints_path')
                .attr('id', function (d) {
                    return `op_schedule_temp_checkpoints_path_${d.id}`;
                })
                .attr('cursor', "pointer")
                .attr('d', function (d, i) {
                    return path_d([
                        [d.x,
                            self.settings.margin.top],
                        [d.x,
                            self.settings.margin.top + self.statistic_height - self.op_bar_height]]);
                })
                .style('stroke', color_manager.default_color)
                .style('stroke-width', '1px')
                .style('opacity', 0);

            let op_schedule_temp_range_background_rect = self.op_schedule_background_group
                .selectAll('.op_schedule_temp_range_background_rect')
                .data([self.temp_op_schedule_range]);

            op_schedule_temp_range_background_rect.enter()
                .append('rect')
                .attr('class', 'op_schedule_temp_range_background_rect')
                .attr('id', function (d) {
                    return `op_schedule_temp_range_background_rect`;
                })
                .attr('cursor', "pointer")
                .attr('x', function (d) {
                    return d.start_x;
                })
                .attr('y', self.settings.margin.top)
                .attr('width', function (d) {
                    return d.end_x - d.start_x;
                })
                .attr('height', self.op_group_detail_height)
                .style('fill', 'white')
                .style('opacity', 0);

            let op_schedule_temp_range_bar_rect = self.op_schedule_background_group
                .selectAll('.op_schedule_temp_range_bar_rect')
                .data([self.temp_op_schedule_range]);

            op_schedule_temp_range_bar_rect.enter()
                .append('rect')
                .attr('class', 'op_schedule_temp_range_bar_rect')
                .attr('id', function (d) {
                    return `op_schedule_temp_range_bar_rect`;
                })
                .attr('cursor', "pointer")
                .attr('x', function (d) {
                    return d.start_x;
                })
                .attr('y', self.settings.margin.top + self.op_group_detail_height - self.op_bar_height)
                .attr('rx', self.op_bar_height / 2)
                .attr('ry', self.op_bar_height / 2)
                .attr('width', function (d) {
                    return d.end_x - d.start_x;
                })
                .attr('height', self.op_bar_height)
                .style('opacity', 0)
                .style('fill', color_manager.node_border_color)
                .style('stroke', 'none')
                .style('stroke-width', 0);
        },
        _generate_chart:function(){
            let self = this;
            self._remove();
            setTimeout(function () {
                self._create();
                init_group_click_menu();
                self._update(self.duration);
            }, self.remove_duration);
        },
        _add_selected_value:function (value_name) {
            let self = this;
            if (!self.multi_value_select) {
                self.used_labels = new Array(30).fill(0);
                self.value_to_label = {};
                self.selected_value = [];
            }
            if (self.selected_value.indexOf(value_name) === -1) {
                self.selected_value.push(value_name);
                self._generate_chart();
            }
        },
        _delete_selected_value:function (value_name) {
            let self = this;
            let selected_value = self.selected_value;
            if (selected_value.indexOf(value_name) !== -1) {
                self.selected_value.remove(value_name);
                d3.selectAll(`#value_line_${value_name}`).remove();
                self.used_labels[self.value_to_label[value_name]] = 0;
                delete self.value_to_label[value_name];
            }
        },
        _remove: function () {
            let self = this;
            let lines = self.statistics_group.selectAll('.iter_line')
                .data(self.iter_lines, function (d) {
                    return d.index;
                });

            let texts = self.statistics_group.selectAll('.iter_line_text')
                .data(self.iter_lines, function (d) {
                    return d.index;
                });

            let ranges = self.statistics_group.selectAll('.iter_range')
                .data(self.iter_ranges, d => d.index);

            let current_values = self.selected_value.map(x=>{
                return {
                    'name': x,
                    'value': self.values[x]
                };
            });
            let paths = self.statistics_group.selectAll('.value_line').data(current_values);

            let show_dependency_op_groups = self.show_dependency_op_groups.map(x=>self.op_groups[x]);
            let op_node_dependency_group = self.op_schedule_dependency_group.selectAll('.op_node_dependency_group').data(show_dependency_op_groups);

            let op_node_dependency_path = self.op_schedule_dependency_group.selectAll('.op_node_dependency_group').selectAll('.op_node_dependency_path')
                .data(function (d) {
                    let sorted_index = [];
                    d.depend_op_index.forEach(depend_index=>{
                        let depend_d = self.op_groups[depend_index];
                        let cx1 = self.op_group_circle_r, cx2 = self.op_group_time_to_x_scale(d.end_time)
                            - self.op_group_time_to_x_scale(d.start_time)
                            - self.op_group_circle_r;
                        if (cx1 > cx2) {
                            cx1 = (cx1 + cx2) / 2;
                            cx2 = cx1;
                        }

                        let start_x = self.op_group_time_to_x_scale(d.start_time) + cx1;
                        cx1 = self.op_group_circle_r;
                        cx2 = self.op_group_time_to_x_scale(depend_d.end_time)
                            - self.op_group_time_to_x_scale(depend_d.start_time)
                            - self.op_group_circle_r;
                        if (cx1 > cx2) {
                            cx1 = (cx1 + cx2) / 2;
                            cx2 = cx1;
                        }
                        let end_x = self.op_group_time_to_x_scale(depend_d.start_time) + cx2;
                        if (start_x !== end_x) {
                            sorted_index.push(depend_index);
                        }
                    });

                    sorted_index.sort(function(a, b) {
                        let depend_a = self.op_groups[a], depend_b = self.op_groups[b];
                        if (depend_a.end_time !== depend_b.end_time) {
                            return depend_a.end_time - depend_b.end_time;
                        }
                        else {
                            return depend_a.schedule_index - depend_b.schedule_index;
                        }
                    });
                    return d.depend_op_index.map(depend_index=>{
                        let depend_d = self.op_groups[depend_index];
                        let cx1 = self.op_group_circle_r, cx2 = self.op_group_time_to_x_scale(d.end_time)
                            - self.op_group_time_to_x_scale(d.start_time)
                            - self.op_group_circle_r;
                        if (cx1 > cx2) {
                            cx1 = (cx1 + cx2) / 2;
                            cx2 = cx1;
                        }

                        let start_x = self.op_group_time_to_x_scale(d.start_time) + cx1;
                        cx1 = self.op_group_circle_r;
                        cx2 = self.op_group_time_to_x_scale(depend_d.end_time)
                            - self.op_group_time_to_x_scale(depend_d.start_time)
                            - self.op_group_circle_r;
                        if (cx1 > cx2) {
                            cx1 = (cx1 + cx2) / 2;
                            cx2 = cx1;
                        }
                        let end_x = self.op_group_time_to_x_scale(depend_d.start_time) + cx2;
                        let margin = 0;
                        let points = [{
                                x: start_x + margin,
                                y: self.settings.margin.top + self.row_gap_y * (d.schedule_index + 0.5)
                            }];

                        if (start_x !== end_x) {
                            let y_ratio = (sorted_index.indexOf(depend_index) + 1) / (sorted_index.length + 1);
                            let num = Math.floor((start_x - end_x) / 5);
                            for (let i = 1;i < num;i++) {
                                points.push({
                                    x: (start_x * (num - i) + end_x * i) / num,
                                    y: (self.settings.margin.top - 5) * y_ratio
                                });
                            }
                        }
                        else {
                            points.push({
                                x: start_x + margin / 3,
                                y: self.settings.margin.top + self.row_gap_y * (d.schedule_index + 0.5)
                            });
                            points.push({
                                x: end_x - margin / 3,
                                y: self.settings.margin.top + self.row_gap_y * (depend_d.schedule_index + 0.5)
                            });
                        }
                        points.push({
                                x: end_x - margin,
                                y: self.settings.margin.top + self.row_gap_y * (depend_d.schedule_index + 0.5)
                            });
                        return {
                            'points': points,
                            'name': `${d.index}_${depend_index}`
                        };
                    });
                });

            let gpu_id_labels = self.op_schedule_group.selectAll('.gpu_id_label_text').data(self.schedule);

            let gpu_id_y_axis = self.op_schedule_group.selectAll('.gpu_id_y_axis_path').data(self.schedule);

            let op_schedule_checkpoints_path_g = self.op_schedule_checkpoints_group.selectAll('.op_schedule_checkpoints_path_g').data(self.op_schedule_checkpoints, function (d) {
                return d.id;
            });

            let op_schedule_range_background_g = self.op_schedule_background_group.selectAll('.op_schedule_range_background_g').data(self.op_schedule_range_data, function (d) {
                return d.index;
            });

            let op_gpu_id_row_group = self.op_schedule_node_group.selectAll('.op_gpu_id_row_group').data(self.schedule);

            texts.exit().transition().duration(self.remove_duration).style('opacity', 0);
            lines.exit().transition().duration(self.remove_duration).style('opacity', 0);
            ranges.exit().transition().duration(self.remove_duration).style('opacity', 0);
            paths.exit().transition().duration(self.remove_duration).style('opacity', 0);
            gpu_id_labels.exit().transition().duration(self.remove_duration).style('opacity', 0);
            gpu_id_y_axis.exit().transition().duration(self.remove_duration).style('opacity', 0);
            op_schedule_checkpoints_path_g.exit().transition().duration(self.remove_duration).style('opacity', 0);
            op_schedule_range_background_g.exit().transition().duration(self.remove_duration).style('opacity', 0);
            op_gpu_id_row_group.exit().transition().duration(self.remove_duration).style('opacity', 0);
            op_node_dependency_group.exit().transition().duration(self.remove_duration).style('opacity', 0);
            op_node_dependency_path.exit().transition().duration(self.remove_duration).style('opacity', 0);

            setTimeout(function () {
                texts.exit().remove();
                lines.exit().remove();
                ranges.exit().remove();
                paths.exit().remove();
                gpu_id_labels.exit().remove();
                gpu_id_y_axis.exit().remove();
                op_schedule_checkpoints_path_g.exit().remove();
                op_schedule_range_background_g.exit().remove();
                op_gpu_id_row_group.exit().remove();
                op_node_dependency_group.exit().remove();
                op_node_dependency_path.exit().remove();
            }, self.remove_duration);
        },
        _create: function () {
            let self = this;
            let lines = self.statistics_group.selectAll('.iter_line')
                .data(self.iter_lines, function (d) {
                    return d.index;
                })

            lines.enter()
                .append('path')
                .attr('class', 'iter_line')
                .attr('id', function (d) {
                    return `iter_line_${d.index}`;
                })
                .attr('cursor', "pointer")
                .attr('d', function (d) {
                    return path_d([
                        [self.time_to_x_scale(d.time), self.settings.margin.top + self.statistic_height - 10],
                        [self.time_to_x_scale(d.time), self.settings.margin.top + self.statistic_height]
                    ]);
                })
                .style('stroke', color_manager.text_color)
                .style('stroke-width', '2px')
                //.style('display', 'none')

            let texts = self.statistics_group.selectAll('.iter_line_text')
                .data(self.iter_lines, function (d) {
                    return d.index;
                })

            texts.enter()
                .append('text')
                .attr('class', 'iter_line_text')
                .attr('cursor', "pointer")
                .attr('id', function (d) {
                    return `iter_line_text_${d.index}`;
                })
                .text(function (d) {
                    return `${d.index}`;
                })
                .attr('x', function (d) {
                    return self.time_to_x_scale(d.time);
                })
                .attr('y', self.settings.margin.top + self.statistic_height + 15)
                .style("user-select", "none")
                .attr('text-anchor', 'middle')
                .attr('font-size', '15px')
                .style('fill', color_manager.text_color)
                .attr('font-family', 'Arial');

            let current_values = self.selected_value.map(x=>{
                return {
                    'name': x,
                    'value': self.values[x]
                };
            });
            let paths = self.statistics_group.selectAll('.value_line').data(current_values);
            paths.enter().append('path')
                .attr('class', 'value_line')
                .style('fill', 'none')
                .style('stroke', color_manager.default_color)
                .style('stroke-width', '2px')
                .attr('id', function (d) {
                    return `value_line_${d.name}`;
                });

            let ranges = self.statistics_group.selectAll('.iter_range')
                .data(self.iter_ranges, d => d.index);
            ranges.enter()
                .append('rect')
                .attr('class', 'iter_range')
                .attr('id', d => `iter_line_${d.index}`)
                .attr('cursor', "pointer")
                .attr('x', d => self.time_to_x_scale(d.time[0]))
                .attr('y', self.settings.margin.top)
                .attr('width', d => self.time_to_x_scale(d.time[1]) - self.time_to_x_scale(d.time[0]))
                .attr('height', self.statistic_height)
                .style('fill', 'transparent')
                .on('click', d => {
                    d.focus = !d.focus;
                    if (d.focus) self.focus_ranges_count += 1; else self.focus_ranges_count -= 1;
                    let statistics = self.statistics;
                    if (statistics['iter'] === undefined) {
                        return;
                    }
                    let iter = statistics['iter'];
                    let iter_num = iter.length;
                    if (self.focus_ranges_count === iter_num) {
                        self.focus_ranges_count = 0;
                        self.iter_ranges.forEach(r => {
                            r.focus = false;
                        })
                    }
                    let unfocus_width = 1 / (self.focus_ranges_count * 9 + iter_num) * self.statistic_width,
                        focus_width = 10 * unfocus_width;
                    let range_slices = [self.settings.margin.left];
                    self.iter_ranges.forEach((r, i) => {
                        range_slices.push((r.focus ? focus_width : unfocus_width) + range_slices[i]);
                    });
                    self.time_to_x_scale = d3.scaleLinear()
                        .domain([0].concat(iter))
                        .range(range_slices);
                    self._generate_chart();
                });

            let gpu_id_labels = self.op_schedule_group.selectAll('.gpu_id_label_text').data(self.schedule);
            gpu_id_labels.enter()
                .append('text')
                .attr('class', 'gpu_id_label_text')
                .attr('cursor', "pointer")
                .attr('id', function (d, i) {
                    return `gpu_id_label_text_${i}`;
                })
                .text(function (d) {
                    return `${d.gpu_id}`;
                })
                .attr('x', function (d) {
                    return self.settings.margin.left - 10;
                })
                .attr('y', function (d, i) {
                    return self.settings.margin.top * 2 + self.op_group_overview_height + self.row_gap_y * (i + 0.5);
                })
                .style("user-select", "none")
                .attr('text-anchor', 'end')
                .attr('dominant-baseline', 'central')
                .attr('font-size', '12px')
                .attr('font-family', 'Arial');

            let gpu_id_y_axis = self.op_schedule_group.selectAll('.gpu_id_y_axis_path').data(self.schedule);
            gpu_id_y_axis.enter()
                .append('path')
                .attr('class', 'gpu_id_y_axis_path')
                .attr('d', function (d, i) {
                    return path_d([
                        [self.settings.margin.left,
                            self.row_gap_y * (i + 0.5) + self.settings.margin.top * 2 + self.op_group_overview_height],
                        [self.settings.margin.left + 6,
                            self.row_gap_y * (i + 0.5) + self.settings.margin.top * 2 + self.op_group_overview_height]]);
                })
                .style('stroke', color_manager.text_color)
                .style('stroke-width', '2px');

            let op_schedule_range_background_g = self.op_schedule_background_group.selectAll('.op_schedule_range_background_g').data(self.op_schedule_range_data, function (d) {
                return d.index;
            });

            let op_schedule_range_background_g_enter = op_schedule_range_background_g.enter().append('g')
                .attr('class', 'op_schedule_range_background_g')
                .attr('id', function (d) {
                    return `op_schedule_range_background_g_${d.index}`;
                })
                .attr('cursor', "pointer")
                .attr('transform', function (d) {
                    return `translate(${d.start_x}, ${self.settings.margin.top})`;
                });

            let op_schedule_range_background_rect = op_schedule_range_background_g_enter.selectAll('.op_schedule_range_background_rect').data(function (d) {
                return [d];
            });

            op_schedule_range_background_rect.enter()
                .append('rect')
                .attr('class', 'op_schedule_range_background_rect')
                .attr('id', function (d) {
                    return `op_schedule_range_background_rect_${d.index}`;
                })
                .attr('cursor', "pointer")
                .attr('x', 0)
                .attr('y', 0)
                .attr('width', function (d) {
                    return d.end_x - d.start_x;
                })
                .attr('height', self.statistic_height)
                .style('fill', 'white');

            let op_schedule_range_bar_rect_g = op_schedule_range_background_g_enter.selectAll('.op_schedule_range_bar_rect_g').data(function (d) {
                return [d];
            });

            let op_schedule_range_bar_rect_g_enter = op_schedule_range_bar_rect_g.enter()
                .append('g')
                .attr('class', 'op_schedule_range_bar_rect_g')
                .attr('id', function (d) {
                    return `op_schedule_range_bar_rect_g_${d.index}`;
                })
                .attr('cursor', "pointer");

            let op_schedule_range_bar_del_btn_g = op_schedule_range_background_g_enter.selectAll('.op_schedule_range_bar_del_btn_g').data(function (d) {
                return [d];
            });

            let op_schedule_range_bar_del_btn_g_enter = op_schedule_range_bar_del_btn_g.enter()
                .append('g')
                .attr('class', 'op_schedule_range_bar_del_btn_g')
                .attr('id', function (d) {
                    return `op_schedule_range_bar_del_btn_g_${d.index}`;
                })
                .attr('cursor', "pointer")
                .attr('transform', function (d) {
                    return `translate(${d.end_x - d.start_x - self.op_schedule_range_del_btn_size}, ${- self.op_schedule_range_del_btn_size - 2})`;
                })
                .style('opacity', function (d) {
                    if (self.checkpoints_update_state === 'delete') {
                        return 1;
                    }
                    return 0;
                })
                .on("mouseenter", function (d) {
                    self.op_schedule_background_group.selectAll(`#op_schedule_range_bar_del_btn_rect_${d.index}`)
                        .transition().duration(self.duration)
                        .style('fill', color_manager.node_btn_highlight_background_color);
                })
                .on("mouseleave", function (d) {
                    self.op_schedule_background_group.selectAll(`#op_schedule_range_bar_del_btn_rect_${d.index}`)
                        .transition().duration(self.duration)
                        .style('fill', 'transparent');
                })
                .on("click", function (d) {
                    let range_index= d.index;
                    let new_check_points = [];
                    let flag = 0;

                    self.op_schedule_checkpoints.forEach(checkpoint=> {
                        if (flag === 1) {
                            self.check_point_id_manager.del_id(checkpoint.id);
                            flag++;
                        }
                        else if (flag === 0 && checkpoint.id === range_index) {
                            self.check_point_id_manager.del_id(checkpoint.id);
                            flag++;
                        }
                        else {
                            new_check_points.push(checkpoint);
                        }
                    });

                    self.op_schedule_checkpoints = new_check_points;

                    let time_slices = [0], x_slices = [self.settings.margin.left + 6];
                    time_slices = time_slices.concat(self.op_schedule_checkpoints.map(check_point=>check_point.time));
                    time_slices.push(self.max_op_end_time);

                    x_slices = x_slices.concat(self.op_schedule_checkpoints.map(check_point=>check_point.x));
                    x_slices.push(self.settings.margin.left + self.statistic_width);

                    self.op_group_time_to_x_scale = d3.scaleLinear()
                        .domain(time_slices)
                        .range(x_slices);
                    self.op_group_x_to_time_scale = d3.scaleLinear()
                        .domain(x_slices)
                        .range(time_slices);

                    let op_schedule_spring_path_data = [{'start_time': 0}];
                    self.op_schedule_checkpoints.forEach(function (d, i) {
                        op_schedule_spring_path_data[i].end_time = d.time;
                        op_schedule_spring_path_data.push({
                            'start_time': d.time
                        });
                    });
                    op_schedule_spring_path_data[op_schedule_spring_path_data.length - 1]['end_time'] = self.max_op_end_time;
                    self.op_schedule_spring_path_data = op_schedule_spring_path_data;

                    self.op_schedule_spring_path_data.forEach(function (d) {
                        d.start_x = self.op_group_time_to_x_scale(d.start_time);
                        d.end_x = self.op_group_time_to_x_scale(d.end_time);
                    });

                    let op_schedule_range_data = [];
                    for (let i = 0;i < self.op_schedule_checkpoints.length;i += 2) {
                        op_schedule_range_data.push({
                            index: self.op_schedule_checkpoints[i].id,
                            start_time: self.op_schedule_checkpoints[i].time,
                            end_time: self.op_schedule_checkpoints[i + 1].time
                        });
                    }
                    self.op_schedule_range_data = op_schedule_range_data;

                    self.op_schedule_range_data.forEach(function (d) {
                        d.start_x = self.op_group_time_to_x_scale(d.start_time);
                        d.end_x = self.op_group_time_to_x_scale(d.end_time);
                    });


                    // self.schedule = process_connected_op_groups(self.op_groups, self.curr_all_schedule, self.op_group_time_to_x_scale);
                    self.schedule = process_op_groups(self.op_groups, self.curr_all_schedule);

                    self._generate_chart();
                });

            op_schedule_range_bar_del_btn_g_enter.append('rect')
                .attr('class', 'op_schedule_range_bar_del_btn_rect')
                .attr('id', function (d) {
                    return `op_schedule_range_bar_del_btn_rect_${d.index}`;
                })
                .attr('x', 0)
                .attr('y', 0)
                .attr('rx', 2)
                .attr('ry', 2)
                .attr('width', self.op_schedule_range_del_btn_size - 1)
                .attr('height', self.op_schedule_range_del_btn_size - 1)
                .style('stroke', color_manager.default_color)
                .style('stroke-width', 1)
                .style('fill', 'transparent');

            op_schedule_range_bar_del_btn_g_enter
                .append('path')
                .attr('class', 'op_schedule_range_bar_del_btn_path')
                .attr('id', function (d) {
                    return `op_schedule_range_bar_del_btn_path_${d.index}`;
                })
                .attr('d', function (d) {
                    return delete_path_d(0, 0, self.op_schedule_range_del_btn_size - 1,
                        self.op_schedule_range_del_btn_size - 1, 4);
                })
                .attr('cursor', "pointer")
                .style('fill', color_manager.node_time_cost_unit_color);

            let op_schedule_range_bar_rect = op_schedule_range_bar_rect_g_enter.selectAll('.op_schedule_range_bar_rect').data(function (d) {
                return [d];
            });

            op_schedule_range_bar_rect.enter()
                .append('rect')
                .attr('class', 'op_schedule_range_bar_rect')
                .attr('id', function (d) {
                    return `op_schedule_range_bar_rect_${d.index}`;
                })
                .attr('cursor', "pointer")
                .attr('x', 0)
                .attr('y', self.statistic_height - self.op_bar_height)
                .attr('rx', self.op_bar_height / 2)
                .attr('ry', self.op_bar_height / 2)
                .attr('width', function (d) {
                    return d.end_x - d.start_x;
                })
                .attr('height', self.op_bar_height)
                .style('opacity', 0.9)
                .style('fill', color_manager.node_border_color)
                .style('stroke', 'none')
                .style('stroke-width', 0);

            let op_schedule_range_bar_left_icon = op_schedule_range_bar_rect_g_enter.selectAll('.op_schedule_range_bar_left_icon').data(function (d) {
                return [d];
            });

            op_schedule_range_bar_left_icon.enter()
                .append('path')
                .attr('class', 'op_schedule_range_bar_left_icon')
                .attr('id', function (d) {
                    return `op_schedule_range_bar_left_icon_${d.index}`;
                })
                .attr('cursor', "pointer")
                .attr('d', function (d) {
                    return op_schedule_bar_icon_path_d(0, (d.end_x - d.start_x)  / 10, self.statistic_height - self.op_bar_height, self.statistic_height, 'left');
                })
                .style('opacity', 0)
                .style('fill', color_manager.node_border_color)
                .style('stroke', 'none')
                .style('stroke-width', 0);

            let op_schedule_range_bar_right_icon = op_schedule_range_bar_rect_g_enter.selectAll('.op_schedule_range_bar_right_icon').data(function (d) {
                return [d];
            });

            op_schedule_range_bar_right_icon.enter()
                .append('path')
                .attr('class', 'op_schedule_range_bar_right_icon')
                .attr('id', function (d) {
                    return `op_schedule_range_bar_right_icon_${d.index}`;
                })
                .attr('cursor', "pointer")
                .attr('d', function (d) {
                    return op_schedule_bar_icon_path_d((d.end_x - d.start_x) * 0.9, d.end_x - d.start_x, self.statistic_height - self.op_bar_height, self.statistic_height, 'right');
                })
                .style('opacity', 0)
                .style('fill', color_manager.node_border_color)
                .style('stroke', 'none')
                .style('stroke-width', 0);

            let op_schedule_range_bar_left_rect = op_schedule_range_bar_rect_g_enter.selectAll('.op_schedule_range_bar_left_rect').data(function (d) {
                return [d];
            });

            op_schedule_range_bar_left_rect.enter()
                .append('rect')
                .attr('class', 'op_schedule_range_bar_left_rect')
                .attr('id', function (d) {
                    return `op_schedule_range_bar_left_rect_${d.index}`;
                })
                .attr('cursor', "pointer")
                .attr('x', 0)
                .attr('y', self.statistic_height - self.op_bar_height)
                .attr('rx', self.op_bar_height / 2)
                .attr('ry', self.op_bar_height / 2)
                .attr('width', function (d) {
                    return (d.end_x - d.start_x) / 10 + self.op_bar_height;
                })
                .attr('height', self.op_bar_height)
                .style('opacity', 0)
                .style('fill', color_manager.node_border_color)
                .style('stroke', 'none')
                .style('stroke-width', 0)
                .on('mouseenter', function (d) {
                    self.op_schedule_background_group.selectAll(`#op_schedule_range_bar_rect_${self.focus_op_schedule_range_bar_index}`)
                        .transition().duration(self.duration)
                        .style('fill', color_manager.node_border_color)
                        .style('stroke', 'none')
                        .style('stroke-width', 0);
                    self.op_schedule_background_group.selectAll(`#op_schedule_range_bar_rect_${d.index}`)
                        .transition().duration(self.duration)
                        .style('fill', 'white')
                        .style('stroke', color_manager.node_border_color)
                        .style('stroke-width', 2);
                    self.op_schedule_background_group.selectAll(`#op_schedule_range_bar_left_icon_${self.focus_op_schedule_range_bar_index}`)
                        .transition().duration(self.duration)
                        .style('opacity', 0);
                    self.op_schedule_background_group.selectAll(`#op_schedule_range_bar_left_icon_${d.index}`)
                        .transition().duration(self.duration)
                        .style('opacity', 0.9);
                    self.op_schedule_background_group.selectAll(`#op_schedule_range_bar_right_icon_${d.index}`)
                        .transition().duration(self.duration)
                        .style('opacity', 0);
                    self.op_schedule_background_group.selectAll(`#op_schedule_range_bar_left_rect_${self.focus_op_schedule_range_bar_index}`)
                        .transition().duration(self.duration)
                        .style('opacity', 0);
                    self.op_schedule_background_group.selectAll(`#op_schedule_range_bar_left_rect_${d.index}`)
                        .transition().duration(self.duration)
                        .style('opacity', 0.5);
                    self.op_schedule_background_group.selectAll(`#op_schedule_range_bar_right_rect_${d.index}`)
                        .transition().duration(self.duration)
                        .style('opacity', 0);
                    self.focus_op_schedule_range_bar_index = d.index;
                    self.focus_op_schedule_range_bar_part = 'left';
                })
                .on('mouseleave', function (d) {
                    self.op_schedule_background_group.selectAll(`#op_schedule_range_bar_rect_${d.index}`)
                        .transition().duration(self.duration)
                        .style('fill', color_manager.node_border_color)
                        .style('stroke', 'none')
                        .style('stroke-width', 0);
                    self.op_schedule_background_group.selectAll(`#op_schedule_range_bar_left_icon_${d.index}`)
                        .transition().duration(self.duration)
                        .style('opacity', 0);
                    self.op_schedule_background_group.selectAll(`#op_schedule_range_bar_left_rect_${d.index}`)
                        .transition().duration(self.duration)
                        .style('opacity', 0);
                    self.focus_op_schedule_range_bar_index = -1;
                    self.focus_op_schedule_range_bar_part = '';
                });

            let op_schedule_range_bar_right_rect = op_schedule_range_bar_rect_g_enter.selectAll('.op_schedule_range_bar_right_rect').data(function (d) {
                return [d];
            });

            op_schedule_range_bar_right_rect.enter()
                .append('rect')
                .attr('class', 'op_schedule_range_bar_right_rect')
                .attr('id', function (d) {
                    return `op_schedule_range_bar_right_rect_${d.index}`;
                })
                .attr('cursor', "pointer")
                .attr('x', function (d) {
                    return (d.end_x - d.start_x) * 0.9 - self.op_bar_height;
                })
                .attr('y', self.statistic_height - self.op_bar_height)
                .attr('rx', self.op_bar_height / 2)
                .attr('ry', self.op_bar_height / 2)
                .attr('width', function (d) {
                    return (d.end_x - d.start_x) / 10 + self.op_bar_height;
                })
                .attr('height', self.op_bar_height)
                .style('opacity', 0)
                .style('fill', color_manager.node_border_color)
                .style('stroke', 'none')
                .style('stroke-width', 0)
                .on('mouseenter', function (d) {
                    self.op_schedule_background_group.selectAll(`#op_schedule_range_bar_rect_${self.focus_op_schedule_range_bar_index}`)
                        .transition().duration(self.duration)
                        .style('fill', color_manager.node_border_color)
                        .style('stroke', 'none')
                        .style('stroke-width', 0);
                    self.op_schedule_background_group.selectAll(`#op_schedule_range_bar_rect_${d.index}`)
                        .transition().duration(self.duration)
                        .style('fill', 'white')
                        .style('stroke', color_manager.node_border_color)
                        .style('stroke-width', 2);
                    self.op_schedule_background_group.selectAll(`#op_schedule_range_bar_right_icon_${self.focus_op_schedule_range_bar_index}`)
                        .transition().duration(self.duration)
                        .style('opacity', 0);
                    self.op_schedule_background_group.selectAll(`#op_schedule_range_bar_right_icon_${d.index}`)
                        .transition().duration(self.duration)
                        .style('opacity', 0.9);
                    self.op_schedule_background_group.selectAll(`#op_schedule_range_bar_left_icon_${d.index}`)
                        .transition().duration(self.duration)
                        .style('opacity', 0);
                    self.op_schedule_background_group.selectAll(`#op_schedule_range_bar_right_rect_${self.focus_op_schedule_range_bar_index}`)
                        .transition().duration(self.duration)
                        .style('opacity', 0);
                    self.op_schedule_background_group.selectAll(`#op_schedule_range_bar_right_rect_${d.index}`)
                        .transition().duration(self.duration)
                        .style('opacity', 0.5);
                    self.op_schedule_background_group.selectAll(`#op_schedule_range_bar_left_rect_${d.index}`)
                        .transition().duration(self.duration)
                        .style('opacity', 0);
                    self.focus_op_schedule_range_bar_index = d.index;
                    self.focus_op_schedule_range_bar_part = 'right';
                })
                .on('mouseleave', function (d) {
                    self.op_schedule_background_group.selectAll(`#op_schedule_range_bar_rect_${d.index}`)
                        .transition().duration(self.duration)
                        .style('fill', color_manager.node_border_color)
                        .style('stroke', 'none')
                        .style('stroke-width', 0);
                    self.op_schedule_background_group.selectAll(`#op_schedule_range_bar_right_icon_${d.index}`)
                        .transition().duration(self.duration)
                        .style('opacity', 0);
                    self.op_schedule_background_group.selectAll(`#op_schedule_range_bar_right_rect_${d.index}`)
                        .transition().duration(self.duration)
                        .style('opacity', 0);
                    self.focus_op_schedule_range_bar_index = -1;
                    self.focus_op_schedule_range_bar_part = '';
                });

            let op_schedule_range_bar_center_rect = op_schedule_range_bar_rect_g_enter.selectAll('.op_schedule_range_bar_center_rect').data(function (d) {
                return [d];
            });

            op_schedule_range_bar_center_rect.enter()
                .append('rect')
                .attr('class', 'op_schedule_range_bar_center_rect')
                .attr('id', function (d) {
                    return `op_schedule_range_bar_center_rect_${d.index}`;
                })
                .attr('cursor', "pointer")
                .attr('x', function (d) {
                    return (d.end_x - d.start_x) / 10;
                })
                .attr('y', self.statistic_height - self.op_bar_height)
                .attr('width', function (d) {
                    return (d.end_x - d.start_x) * 0.8;
                })
                .attr('height', self.op_bar_height)
                .style('opacity', 0)
                .style('fill', 'white')
                .style('stroke', 'none')
                .style('stroke-width', 0)
                .on('mouseenter', function (d) {
                    self.op_schedule_background_group.selectAll(`#op_schedule_range_bar_rect_${self.focus_op_schedule_range_bar_index}`)
                        .transition().duration(self.duration)
                        .style('fill', color_manager.node_border_color)
                        .style('stroke', 'none')
                        .style('stroke-width', 0);
                    self.op_schedule_background_group.selectAll(`#op_schedule_range_bar_rect_${d.index}`)
                        .transition().duration(self.duration)
                        .style('fill', 'white')
                        .style('stroke', color_manager.node_border_color)
                        .style('stroke-width', 2);
                    self.op_schedule_background_group.selectAll(`#op_schedule_range_bar_left_icon_${self.focus_op_schedule_range_bar_index}`)
                        .transition().duration(self.duration)
                        .style('opacity', 0);
                    self.op_schedule_background_group.selectAll(`#op_schedule_range_bar_left_icon_${d.index}`)
                        .transition().duration(self.duration)
                        .style('opacity', 0.9);
                    self.op_schedule_background_group.selectAll(`#op_schedule_range_bar_right_icon_${self.focus_op_schedule_range_bar_index}`)
                        .transition().duration(self.duration)
                        .style('opacity', 0);
                    self.op_schedule_background_group.selectAll(`#op_schedule_range_bar_right_icon_${d.index}`)
                        .transition().duration(self.duration)
                        .style('opacity', 0.9);
                    self.focus_op_schedule_range_bar_index = d.index;
                    self.focus_op_schedule_range_bar_part = 'center';
                })
                .on('mouseleave', function (d) {
                    self.op_schedule_background_group.selectAll(`#op_schedule_range_bar_rect_${d.index}`)
                        .transition().duration(self.duration)
                        .style('fill', color_manager.node_border_color)
                        .style('stroke', 'none')
                        .style('stroke-width', 0);
                    self.op_schedule_background_group.selectAll(`#op_schedule_range_bar_left_icon_${d.index}`)
                        .transition().duration(self.duration)
                        .style('opacity', 0);
                    self.op_schedule_background_group.selectAll(`#op_schedule_range_bar_right_icon_${d.index}`)
                        .transition().duration(self.duration)
                        .style('opacity', 0);
                    self.focus_op_schedule_range_bar_index = -1;
                    self.focus_op_schedule_range_bar_part = '';
                });




            let op_schedule_checkpoints_path_g = self.op_schedule_checkpoints_group.selectAll('.op_schedule_checkpoints_path_g').data(self.op_schedule_checkpoints, function (d) {
                return d.id;
            });

            let op_schedule_checkpoints_path_g_enter = op_schedule_checkpoints_path_g.enter()
                .append('g')
                .attr('class', 'op_schedule_checkpoints_path_g')
                .attr('id', function (d) {
                    return `op_schedule_checkpoints_path_g_${d.id}`;
                })
                .attr('cursor', "pointer")
                .attr('transform', function (d) {
                    return `translate(${d.x - self.op_schedule_checkpoints_width / 2}, ${self.settings.margin.top})`;
                })
                .on('mouseenter', function (d) {
                    self.op_schedule_checkpoints_group.selectAll(`#op_schedule_checkpoints_path_rect_${d.id}`).style('opacity', 0.5);
                    self.op_schedule_checkpoints_group.selectAll(`#op_schedule_checkpoints_path_${d.id}`).style('stroke-width', '2px');
                    self.focus_op_schedule_checkpoint_index = d.id;
                })
                .on('mouseleave', function (d) {
                    self.op_schedule_checkpoints_group.selectAll(`#op_schedule_checkpoints_path_rect_${d.id}`).style('opacity', 0);
                    self.op_schedule_checkpoints_group.selectAll(`#op_schedule_checkpoints_path_${d.id}`).style('stroke-width', '1px');
                    self.focus_op_schedule_checkpoint_index = -1;
                });

            let op_schedule_checkpoints_path_rect = op_schedule_checkpoints_path_g_enter.selectAll('.op_schedule_checkpoints_path_rect').data(function (d) {
                return [d];
            });

            op_schedule_checkpoints_path_rect.enter()
                .append('rect')
                .attr('class', 'op_schedule_checkpoints_path_rect')
                .attr('id', function (d) {
                    return `op_schedule_checkpoints_path_rect_${d.id}`;
                })
                .attr('cursor', "pointer")
                .attr('x', 0)
                .attr('y', 0)
                .attr('width', self.op_schedule_checkpoints_width)
                .attr('height', self.statistic_height - self.op_bar_height)
                .style('fill', color_manager.default_color)
                .style('opacity', 0);

            let op_schedule_checkpoints_path = op_schedule_checkpoints_path_g_enter.selectAll('.op_schedule_checkpoints_path').data(function (d) {
                return [d];
            });

            op_schedule_checkpoints_path.enter()
                .append('path')
                .attr('class', 'op_schedule_checkpoints_path')
                .attr('id', function (d) {
                    return `op_schedule_checkpoints_path_${d.id}`;
                })
                .attr('cursor', "pointer")
                .attr('d', function (d, i) {
                    return path_d([
                        [self.op_schedule_checkpoints_width / 2,
                            0],
                        [self.op_schedule_checkpoints_width / 2,
                            self.statistic_height - self.op_bar_height]]);
                })
                .style('stroke', color_manager.default_color)
                .style('stroke-width', '1px');
            let show_dependency_op_groups = self.show_dependency_op_groups.map(x=>self.op_groups[x]);
            let op_node_dependency_group = self.op_schedule_dependency_group.selectAll('.op_node_dependency_group').data(show_dependency_op_groups);
            op_node_dependency_group.enter().append('g')
                .attr('class', 'op_node_dependency_group')
                .attr('id', function (d) {
                    return `op_node_dependency_group_${d.index}`;
                });

            let op_node_dependency_path = self.op_schedule_dependency_group
                .selectAll('.op_node_dependency_group')
                .selectAll('.op_node_dependency_path')
                .data(function (d) {
                    let sorted_index = [];
                    d.depend_op_index.forEach(depend_index=>{
                        let depend_d = self.op_groups[depend_index];
                        let cx1 = self.op_group_circle_r, cx2 = self.op_group_time_to_x_scale(d.end_time)
                            - self.op_group_time_to_x_scale(d.start_time)
                            - self.op_group_circle_r;
                        if (cx1 > cx2) {
                            cx1 = (cx1 + cx2) / 2;
                            cx2 = cx1;
                        }

                        let start_x = self.op_group_time_to_x_scale(d.start_time) + cx1;
                        cx1 = self.op_group_circle_r;
                        cx2 = self.op_group_time_to_x_scale(depend_d.end_time)
                            - self.op_group_time_to_x_scale(depend_d.start_time)
                            - self.op_group_circle_r;
                        if (cx1 > cx2) {
                            cx1 = (cx1 + cx2) / 2;
                            cx2 = cx1;
                        }
                        let end_x = self.op_group_time_to_x_scale(depend_d.start_time) + cx2;
                        if (start_x !== end_x) {
                            sorted_index.push(depend_index);
                        }
                    });

                    sorted_index.sort(function(a, b) {
                        let depend_a = self.op_groups[a], depend_b = self.op_groups[b];
                        if (depend_a.end_time !== depend_b.end_time) {
                            return depend_a.end_time - depend_b.end_time;
                        }
                        else {
                            return depend_a.schedule_index - depend_b.schedule_index;
                        }
                    });
                    return d.depend_op_index.map(depend_index=>{
                        let depend_d = self.op_groups[depend_index];
                        let cx1 = self.op_group_circle_r, cx2 = self.op_group_time_to_x_scale(d.end_time)
                            - self.op_group_time_to_x_scale(d.start_time)
                            - self.op_group_circle_r;
                        if (cx1 > cx2) {
                            cx1 = (cx1 + cx2) / 2;
                            cx2 = cx1;
                        }

                        let start_x = self.op_group_time_to_x_scale(d.start_time) + cx1;
                        cx1 = self.op_group_circle_r;
                        cx2 = self.op_group_time_to_x_scale(depend_d.end_time)
                            - self.op_group_time_to_x_scale(depend_d.start_time)
                            - self.op_group_circle_r;
                        if (cx1 > cx2) {
                            cx1 = (cx1 + cx2) / 2;
                            cx2 = cx1;
                        }
                        let end_x = self.op_group_time_to_x_scale(depend_d.start_time) + cx2;
                        let margin = 0;
                        let points = [{
                                x: start_x + margin,
                                y: self.settings.margin.top + self.row_gap_y * (d.schedule_index + 0.5)
                            }];

                        if (start_x !== end_x) {
                            let y_ratio = (sorted_index.indexOf(depend_index) + 1) / (sorted_index.length + 1);
                            let num = Math.floor((start_x - end_x) / 5);
                            for (let i = 1;i < num;i++) {
                                points.push({
                                    x: (start_x * (num - i) + end_x * i) / num,
                                    y: (self.settings.margin.top - 5) * y_ratio
                                });
                            }
                        }
                        else {
                            points.push({
                                x: start_x + margin / 3,
                                y: self.settings.margin.top + self.row_gap_y * (d.schedule_index + 0.5)
                            });
                            points.push({
                                x: end_x - margin / 3,
                                y: self.settings.margin.top + self.row_gap_y * (depend_d.schedule_index + 0.5)
                            });
                        }
                        points.push({
                                x: end_x - margin,
                                y: self.settings.margin.top + self.row_gap_y * (depend_d.schedule_index + 0.5)
                            });
                        return {
                            'points': points,
                            'name': `${d.index}_${depend_index}`
                        };
                    });
                });

            op_node_dependency_path.enter().append('path')
                .attr('class', 'op_node_dependency_path')
                .attr('id', function (d) {
                    return `op_node_dependency_path_${d.name}`;
                })
                .attr('d', function (d) {
                    return cardinal_line(d.points);
                })
                .style('stroke', color_manager.text_color)
                .style('stroke-width', '1px')
                .style('fill', 'none');

            let op_gpu_id_row_group = self.op_schedule_node_group.selectAll('.op_gpu_id_row_group').data(self.schedule);
            let op_gpu_id_row_group_g = op_gpu_id_row_group.enter()
                .append('g')
                .attr('class', 'op_gpu_id_row_group')
                .attr('id', function (d) {
                    return `op_gpu_id_row_group_${d.gpu_id}`;
                })
                .attr('cursor', "pointer")
                .attr('transform', function (d, i) {
                    return `translate(0, ${self.row_gap_y * (i + 0.5) + self.settings.margin.top})`;
                });

            let op_group_g = op_gpu_id_row_group_g.selectAll('.op_group_g').data(function (d, i) {
                return d.op_groups.map(op_index=>{
                    return {
                        'gpu_id': d.gpu_id,
                        'gpu_index': i,
                        'op_info': self.op_groups[op_index]
                    };
                });
            });
            let op_group_g_enter = op_group_g.enter()
                .append('g')
                .attr('class', function (d) {
                    if (self.show_dependency_op_groups.indexOf(d.op_info.index) === -1) {
                        return 'op_group_g op_group_g_show';
                    }
                    return 'op_group_g op_group_g_hide';
                })
                .attr('id', function (d) {
                    return `op_group_g_${d.op_info.index}`;
                })
                .attr('cursor', "pointer")
                .attr('transform', function (d, i) {
                    return `translate(${self.op_group_time_to_x_scale(d.op_info.start_time)}, 0)`;
                })
                .on("mouseenter", function (d) {
                    self.focus_op_index = d.op_info.index;
                    if (self.highlight_op_index[d.op_info.index] === undefined) {
                        // self.op_schedule_group.selectAll(`#op_group_background_${d.op_info.index}`)
                        // .transition().duration(self.duration)
                        // .style('opacity', 0.3);
                        // self.op_group_highlight_click_manager.try_click(function () {
                        //     highlight_network_dots(self.op_groups[d.op_info.index], -1, 'add');
                        // });
                    }
                    // self.op_schedule_dependency_group
                    //     .selectAll(`#op_node_dependency_group_${d.op_info.index}`)
                    //     .transition().duration(self.duration).style('opacity', 1);

                    // let keys = ['index', 'file_path', 'jit_key'];
                    // let max_size = 0;
                    // keys.forEach((key, i)=>{
                    //     max_size = Math.max(max_size, short_string(key + ':' + d.op_info[key], 1000).length);
                    // });
                    //
                    // self.info_rect_width = max_size * 7.5;
                    // let x = self.mouse_pos.x + 15;
                    // if (self.statistic_width - x < self.info_rect_width && x > self.statistic_width / 2) {
                    //     x -= self.info_rect_width + 30;
                    // }
                    // self.op_schedule_info_group.selectAll('#op_schedule_info_background')
                    //     .transition().duration(self.duration)
                    //     .attr('width', self.info_rect_width)
                    //     .attr('height', self.statistic_height - 5 - self.op_bar_height);
                    // keys.forEach((key, i)=>{
                    //     self.op_schedule_info_group.selectAll(`#${key}-info`)
                    //     .transition().duration(self.duration).text(short_string(key + ':' + d.op_info[key], 1000))
                    //         .attr('x', 2)
                    //         .attr('y', 2 + i * 15);
                    // });
                    //
                    // self.op_schedule_info_group
                    //     .transition().duration(self.duration)
                    //     .attr('transform', `translate(${x}, 0)`)
                    //     .style('opacity', 1);
                })
                .on("mouseleave", function (d) {
                    if (self.highlight_op_index[d.op_info.index] === undefined && d.op_info.index !== self.showing_detail_info) {
                        // self.op_schedule_group.selectAll(`#op_group_background_${d.op_info.index}`)
                        //     .transition().duration(self.duration)
                        //     .style('opacity', 0);
                    }
                    // self.op_schedule_dependency_group
                    //     .selectAll(`#op_node_dependency_group_${d.op_info.index}`)
                    //     .transition().duration(self.duration).style('opacity', 0);


                    // self.info_rect_width = 0;
                    // self.op_schedule_info_group
                    //     .transition().duration(self.duration)
                    //     .style('opacity', 0);
                })
                .on("click", function (d) {
                    if (true || d.op_info.end_time === d.op_info.end_time) {
                        let index = self.highlight_op_index[d.op_info.index];
                        if (index === undefined) {
                            self.highlight_op_index[d.op_info.index] = self.highlight_op_index_manager.get_id();
                            self.op_schedule_group.selectAll(`#op_group_background_${d.op_info.index}`)
                            .transition().duration(self.duration)
                            // .style('opacity', 0.5)
                            .style('fill', color_manager.get_op_group_highlight_color(self.highlight_op_index[d.op_info.index]));
                            self.op_group_highlight_click_manager.try_click(function () {
                                // highlight_network_dots(self.op_groups[d.op_info.index], -1, 'delete');
                                highlight_network_dots(self.op_groups[d.op_info.index], self.highlight_op_index[d.op_info.index], 'add');
                            });
                        }
                        else {
                            self.op_schedule_group.selectAll(`#op_group_background_${d.op_info.index}`)
                            .transition().duration(self.duration)
                            // .style('opacity', 0.3)
                            .style('fill', color_manager.get_color_by_label(d.op_info.jit_type));
                            self.highlight_op_index_manager.del_id(self.highlight_op_index[d.op_info.index]);
                            self.op_group_highlight_click_manager.try_click(function () {
                                highlight_network_dots(self.op_groups[d.op_info.index], self.highlight_op_index[d.op_info.index], 'delete');
                                // highlight_network_dots(self.op_groups[d.op_info.index], -1, 'add');
                                delete self.highlight_op_index[d.op_info.index];
                            });

                        }
                    }
                });

            let op_group_background = op_group_g_enter.selectAll('.op_group_background').data(function (d, i) {
                return [d];
            });

            op_group_background.enter()
                .append('rect')
                .attr('class', 'op_group_background')
                .attr('id', function (d) {
                    return `op_group_background_${d.op_info.index}`;
                })
                .attr('x', 0)
                .attr('y', -self.row_gap_y / 2 + 2)
                // .attr('rx', self.row_gap_y / 2)
                // .attr('ry', self.row_gap_y / 2)
                .attr('width', function (d) {
                    return self.op_group_time_to_x_scale(d.op_info.end_time)
                        - self.op_group_time_to_x_scale(d.op_info.start_time);
                })
                .attr('height', function (d) {
                    return self.row_gap_y - 4;
                })
                .style('fill', function (d) {
                    return color_manager.get_color_by_label(d.op_info.jit_type);
                })
                .style('opacity', function (d) {
                    return 0.5;
                    // let index = self.highlight_op_index[d.op_info.index];
                    // if (index !== undefined) {
                    //     return 0.5;
                    // }
                    // else if (d.op_info.index === self.focus_op_index) {
                    //     return 0.3;
                    // }
                    // return 0;
                });


            // let op_group_number_text = op_group_g_enter.selectAll('.op_group_number_text').data(function (d, i) {
            //     return [d];
            // });
            //
            // op_group_number_text.enter()
            //     .append('text')
            //     .attr('class', 'op_group_number_text')
            //     .attr('id', function (d) {
            //         return `op_group_number_text_${d.op_info.index}`;
            //     })
            //     .text(function (d) {
            //         let end_x = self.op_group_time_to_x_scale(d.op_info.end_time),
            //             start_x = self.op_group_time_to_x_scale(d.op_info.start_time);
            //         if (end_x - start_x > 50) {
            //             return `${d.op_info.connected_number}`;
            //         }
            //         return '';
            //     })
            //     .style('dominant-baseline', 'auto')
            //     .style('text-anchor', 'middle')
            //     .attr('x', function (d) {
            //         let end_x = self.op_group_time_to_x_scale(d.op_info.end_time),
            //             start_x = self.op_group_time_to_x_scale(d.op_info.start_time);
            //         return (end_x - start_x) / 2;
            //     })
            //     .attr('y', -2);
            //
            //
            // let op_group_path = op_group_g_enter.selectAll('.op_group_path').data(function (d, i) {
            //     return [d];
            // });
            //
            // op_group_path.enter()
            //     .append('path')
            //     .attr('class', 'op_group_path')
            //     .attr('id', function (d) {
            //         return `op_group_path_${d.op_info.index}`;
            //     })
            //     .attr('d', function (d) {
            //         return path_d([[self.row_gap_y / 2, 0],
            //             [self.op_group_time_to_x_scale(d.op_info.end_time)
            //             - self.op_group_time_to_x_scale(d.op_info.start_time)
            //             - self.row_gap_y / 2, 0]]);
            //     })
            //     .style('stroke', color_manager.text_color)
            //     .style('stroke-width', '1px')
            //     .style('fill', 'none');
            //
            // let op_group_circle = op_group_g_enter.selectAll('.op_group_circle').data(function (d, i) {
            //     let cx1 = self.op_group_circle_r, cx2 = self.op_group_time_to_x_scale(d.op_info.end_time)
            //             - self.op_group_time_to_x_scale(d.op_info.start_time)
            //             - self.op_group_circle_r;
            //     if (cx1 > cx2) {
            //         cx1 = (cx1 + cx2) / 2;
            //         cx2 = cx1;
            //     }
            //     return [{
            //         'op_index': d.op_info.index,
            //         'cx': cx1,
            //         'cy': 0,
            //         'r': self.op_group_circle_r
            //     }, {
            //         'op_index': d.op_info.index,
            //         'cx': cx2,
            //         'cy': 0,
            //         'r': self.op_group_circle_r
            //     }];
            // });
            //
            // op_group_circle.enter()
            //     .append('circle')
            //     .attr('class', 'op_group_circle')
            //     .attr('id', function (d, i) {
            //         return `op_group_circle_${d.op_index}_${i}`;
            //     })
            //     .attr('cursor', "pointer")
            //     .attr('cx', d=>d.cx)
            //     .attr('cy', d=>d.cy)
            //     .attr('r', d=>d.r)
            //     .style('fill', function (d, i) {
            //         if (i === 0 ) {
            //             return color_manager.node_border_color;
            //         }
            //         return `white`;
            //     })
            //     .style('stroke', color_manager.node_border_color)
            //     .style('stroke-width', 1);

        },
        _update: function (duration) {
            let self = this;
            self._update_axis_stable_text_and_path(duration);
            if (self.mode === 'statistics') {
                self.statistics_group
                    .transition()
                    .duration(duration)
                .attr('transform', `translate(0,0)`)
                .style('opacity', 1);
                self.op_schedule_group
                    .transition()
                    .duration(duration)
                .attr('transform', `translate(${self.chart_width},0)`)
                .style('opacity', 0);
            }
            else {
                self.statistics_group
                    .transition()
                    .duration(duration)
                .attr('transform', `translate(${-self.chart_width},0)`)
                .style('opacity', 0);
                self.op_schedule_group
                    .transition()
                    .duration(duration)
                .attr('transform', `translate(0,0)`)
                .style('opacity', 1);
            }
            let lines = self.statistics_group.selectAll('.iter_line')
                .data(self.iter_lines, function (d) {
                    return d.index;
                });

            lines.transition()
                .duration(duration)
                .attr('d', function (d) {
                    return path_d([
                        [self.time_to_x_scale(d.time), self.settings.margin.top + self.statistic_height - 10],
                        [self.time_to_x_scale(d.time), self.settings.margin.top + self.statistic_height]
                    ]);
                });
            let texts = self.statistics_group.selectAll('.iter_line_text')
                .data(self.iter_lines, function (d) {
                    return d.index;
                });
            texts.transition()
                .duration(duration)
                .attr('x', function (d) {
                    return self.time_to_x_scale(d.time);
                });

            self.statistics_group.selectAll('#spring_path')
                .transition()
                .duration(duration)
                .attr('d', function (d) {
                    return spring_path_d(self.min_end_time, self.max_end_time,
                        self.settings.margin.top + self.statistic_height,
                        0, self.spring_num, self.time_to_x_scale);
                })
                .style('stroke-width', 2)

            let current_values = self.selected_value.map(x=>{
                return {
                    'name': x,
                    'value': self.values[x]
                };
            });
            let paths = self.statistics_group.selectAll('.value_line').data(current_values);
            paths.transition()
                    .duration(duration)
                    .attr('d' , function (d) {
                        if (self.statistics['value']['iter'].indexOf(d.name) !== -1) {
                            let points = [],//[[self.settings.margin.left, self.settings.margin.top + self.statistic_height]],
                                curr_time = self.min_end_time,
                                curr_x = self.settings.margin.left,
                                curr_y = self.settings.margin.top + self.statistic_height;
                            d.value.forEach(function (x, i) {
                                let end_time = self.iter_lines[i].time,
                                    end_x = self.time_to_x_scale(end_time);
                                if (points.length == 0) {
                                    points.push([end_x, x.y])
                                } else {
                                    points = points.concat(insert_point_to_line(curr_x, curr_y, end_x, x.y, end_time - curr_time - 1));
                                }
                                curr_time = end_time;
                                curr_y = x.y;
                                curr_x = end_x;
                            });
                            return path_d(points);
                        }
                        else {
                            return lines_path_d(d.value, self.iter_lines, self.time_to_x_scale);
                        }
                    });

            let ranges = self.statistics_group.selectAll('.iter_range')
                .data(self.iter_ranges, d => d.index);
            ranges.transition()
                .duration(duration)
                .attr('x', d => self.time_to_x_scale(d.time[0]))
                .attr('y', self.settings.margin.top)
                .attr('width', d => self.time_to_x_scale(d.time[1]) - self.time_to_x_scale(d.time[0]))
                .attr('height', self.statistic_height);

            let gpu_id_labels = self.op_schedule_group.selectAll('.gpu_id_label_text').data(self.schedule);
            gpu_id_labels.transition()
                .duration(duration)
                .text(function (d) {
                    return `${d.gpu_id}`;
                })
                .attr('x', function (d) {
                    return self.settings.margin.left - 10;
                })
                .attr('y', function (d, i) {
                    return self.settings.margin.top * 2 + self.op_group_overview_height + self.row_gap_y * (i + 0.5);
                });


            let gpu_id_y_axis = self.op_schedule_group.selectAll('.gpu_id_y_axis_path').data(self.schedule);
            gpu_id_y_axis.transition()
                .duration(duration)
                .attr('d', function (d, i) {
                    return path_d([
                        [self.settings.margin.left,
                            self.row_gap_y * (i + 0.5) + self.settings.margin.top * 2 + self.op_group_overview_height],
                        [self.settings.margin.left + 6,
                            self.row_gap_y * (i + 0.5) + self.settings.margin.top * 2 + self.op_group_overview_height]]);
                });



            self.op_schedule_dependency_group
                .selectAll('.op_node_dependency_group')
                .selectAll('.op_node_dependency_path')
                .data(function (d) {
                    let sorted_index = [];
                    d.depend_op_index.forEach(depend_index=>{
                        let depend_d = self.op_groups[depend_index];
                        let cx1 = self.op_group_circle_r, cx2 = self.op_group_time_to_x_scale(d.end_time)
                            - self.op_group_time_to_x_scale(d.start_time)
                            - self.op_group_circle_r;
                        if (cx1 > cx2) {
                            cx1 = (cx1 + cx2) / 2;
                            cx2 = cx1;
                        }

                        let start_x = self.op_group_time_to_x_scale(d.start_time) + cx1;
                        cx1 = self.op_group_circle_r;
                        cx2 = self.op_group_time_to_x_scale(depend_d.end_time)
                            - self.op_group_time_to_x_scale(depend_d.start_time)
                            - self.op_group_circle_r;
                        if (cx1 > cx2) {
                            cx1 = (cx1 + cx2) / 2;
                            cx2 = cx1;
                        }
                        let end_x = self.op_group_time_to_x_scale(depend_d.start_time) + cx2;
                        if (start_x !== end_x) {
                            sorted_index.push(depend_index);
                        }
                    });

                    sorted_index.sort(function(a, b) {
                        let depend_a = self.op_groups[a], depend_b = self.op_groups[b];
                        if (depend_a.end_time !== depend_b.end_time) {
                            return depend_a.end_time - depend_b.end_time;
                        }
                        else {
                            return depend_a.schedule_index - depend_b.schedule_index;
                        }
                    });
                    return d.depend_op_index.map(depend_index=>{
                        let depend_d = self.op_groups[depend_index];
                        let cx1 = self.op_group_circle_r, cx2 = self.op_group_time_to_x_scale(d.end_time)
                            - self.op_group_time_to_x_scale(d.start_time)
                            - self.op_group_circle_r;
                        if (cx1 > cx2) {
                            cx1 = (cx1 + cx2) / 2;
                            cx2 = cx1;
                        }

                        let start_x = self.op_group_time_to_x_scale(d.start_time) + cx1;
                        cx1 = self.op_group_circle_r;
                        cx2 = self.op_group_time_to_x_scale(depend_d.end_time)
                            - self.op_group_time_to_x_scale(depend_d.start_time)
                            - self.op_group_circle_r;
                        if (cx1 > cx2) {
                            cx1 = (cx1 + cx2) / 2;
                            cx2 = cx1;
                        }
                        let end_x = self.op_group_time_to_x_scale(depend_d.start_time) + cx2;
                        let margin = 0;
                        let points = [{
                                x: start_x + margin,
                                y: self.settings.margin.top + self.row_gap_y * (d.schedule_index + 0.5)
                            }];

                        if (start_x !== end_x) {
                            let y_ratio = (sorted_index.indexOf(depend_index) + 1) / (sorted_index.length + 1);
                            let num = Math.floor((start_x - end_x) / 5);
                            for (let i = 1;i < num;i++) {
                                points.push({
                                    x: (start_x * (num - i) + end_x * i) / num,
                                    y: (self.settings.margin.top - 5) * y_ratio
                                });
                            }
                        }
                        else {
                            points.push({
                                x: start_x + margin / 3,
                                y: self.settings.margin.top + self.row_gap_y * (d.schedule_index + 0.5)
                            });
                            points.push({
                                x: end_x - margin / 3,
                                y: self.settings.margin.top + self.row_gap_y * (depend_d.schedule_index + 0.5)
                            });
                        }
                        points.push({
                                x: end_x - margin,
                                y: self.settings.margin.top + self.row_gap_y * (depend_d.schedule_index + 0.5)
                            });
                        return {
                            'points': points,
                            'name': `${d.index}_${depend_index}`
                        };
                    });
                }).transition()
                .duration(duration)
                .attr('d', function (d) {
                    return cardinal_line(d.points);
                });

            let op_schedule_range_background_g = self.op_schedule_background_group.selectAll('.op_schedule_range_background_g').data(self.op_schedule_range_data, function (d) {
                return d.index;
            });
            op_schedule_range_background_g.transition()
                .duration(duration)
                .attr('transform', function (d) {
                    return `translate(${d.start_x}, ${self.settings.margin.top})`;
                });

            op_schedule_range_background_g.selectAll('.op_schedule_range_background_rect').data(function (d) {
                return [d];
            }).transition()
                .duration(duration)
                .attr('width', function (d) {
                    return d.end_x - d.start_x;
                })
                .attr('height', self.statistic_height);

            let op_schedule_range_bar_del_btn_g = op_schedule_range_background_g.selectAll('.op_schedule_range_bar_del_btn_g').data(function (d) {
                return [d];
            });

            op_schedule_range_bar_del_btn_g.transition()
                .duration(duration)
                .attr('transform', function (d) {
                    return `translate(${d.end_x - d.start_x - self.op_schedule_range_del_btn_size}, ${- self.op_schedule_range_del_btn_size})`;
                });

            let op_schedule_range_bar_rect_g = op_schedule_range_background_g.selectAll('.op_schedule_range_bar_rect_g').data(function (d) {
                return [d];
            });

            op_schedule_range_bar_rect_g.selectAll('.op_schedule_range_bar_center_rect').data(function (d) {
                return [d];
            }).transition()
                .duration(duration)
                .attr('x', function (d) {
                    return (d.end_x - d.start_x) / 10;
                })
                .attr('y', self.statistic_height - self.op_bar_height)
                .attr('width', function (d) {
                    return (d.end_x - d.start_x) * 0.8;
                })
                .attr('height', self.op_bar_height);

            op_schedule_range_bar_rect_g.selectAll('.op_schedule_range_bar_left_icon').data(function (d) {
                return [d];
            }).transition()
                .duration(duration)
                .attr('d', function (d) {
                    return op_schedule_bar_icon_path_d(0, (d.end_x - d.start_x) / 10, self.statistic_height - self.op_bar_height, self.statistic_height, 'left');
                });

            op_schedule_range_bar_rect_g.selectAll('.op_schedule_range_bar_right_icon').data(function (d) {
                return [d];
            }).transition()
                .duration(duration)
                .attr('d', function (d) {
                    return op_schedule_bar_icon_path_d((d.end_x - d.start_x) * 0.9, d.end_x - d.start_x, self.statistic_height - self.op_bar_height, self.statistic_height, 'right');
                });

            op_schedule_range_bar_rect_g.selectAll('.op_schedule_range_bar_left_rect').data(function (d) {
                return [d];
            }).transition()
                .duration(duration)
                .attr('x', 0)
                .attr('y', self.statistic_height - self.op_bar_height)
                .attr('width', function (d) {
                    return (d.end_x - d.start_x) / 10 + self.op_bar_height;
                })
                .attr('height', self.op_bar_height);

            op_schedule_range_bar_rect_g.selectAll('.op_schedule_range_bar_right_rect').data(function (d) {
                return [d];
            }).transition()
                .duration(duration)
                .attr('x', function (d) {
                    return (d.end_x - d.start_x) * 0.9 - self.op_bar_height;
                })
                .attr('y', self.statistic_height - self.op_bar_height)
                .attr('width', function (d) {
                    return (d.end_x - d.start_x) / 10 + self.op_bar_height;
                })
                .attr('height', self.op_bar_height);

            op_schedule_range_bar_rect_g.selectAll('.op_schedule_range_bar_rect').data(function (d) {
                return [d];
            }).transition()
                .duration(duration)
                .attr('y', self.statistic_height - self.op_bar_height)
                .attr('width', function (d) {
                    return d.end_x - d.start_x;
                })
                .attr('height', self.op_bar_height);

            let op_schedule_checkpoints_path_g = self.op_schedule_checkpoints_group.selectAll('.op_schedule_checkpoints_path_g').data(self.op_schedule_checkpoints, function (d) {
                return `${d.id}`;
            });

            op_schedule_checkpoints_path_g.transition()
                .duration(duration)
                .attr('transform', function (d) {
                    return `translate(${d.x - self.op_schedule_checkpoints_width / 2}, ${self.settings.margin.top})`;
                });

            let op_schedule_checkpoints_path_rect = op_schedule_checkpoints_path_g.selectAll('.op_schedule_checkpoints_path_rect').data(function (d) {
                return [d];
            });

            op_schedule_checkpoints_path_rect.transition()
                .duration(duration)
                .style('opacity', 0);

            let op_schedule_checkpoints_path = op_schedule_checkpoints_path_g.selectAll('.op_schedule_checkpoints_path').data(function (d) {
                return [d];
            });

            op_schedule_checkpoints_path.transition()
                .duration(duration)
                .attr('d', function (d, i) {
                    return path_d([
                        [self.op_schedule_checkpoints_width / 2,
                            0],
                        [self.op_schedule_checkpoints_width / 2,
                            self.statistic_height - self.op_bar_height]]);
                })
                .style('stroke-width', '1px');


            let op_schedule_spring_path = self.op_schedule_spring_group
                .selectAll('#op_schedule_spring_path')
                .transition()
                .duration(duration)
                .attr('d', function (d) {
                    return spring_path_d(0, self.max_op_end_time, self.settings.margin.top + self.statistic_height + self.spring_delta_y + 3,
                        self.spring_delta_y, self.spring_num, self.op_group_time_to_x_scale);
                })
                .style('stroke-width', 2)

            let op_gpu_id_row_group = self.op_schedule_node_group
                .selectAll('.op_gpu_id_row_group')
                .data(self.schedule);
            op_gpu_id_row_group.transition()
                .duration(duration)
                .attr('transform', function (d, i) {
                    return `translate(0, ${self.row_gap_y * (i + 0.5) + self.settings.margin.top})`;
                });

            let op_group_g = self.op_schedule_node_group
                .selectAll('.op_gpu_id_row_group')
                .selectAll('.op_group_g')
                .data(function (d, i) {
                    return d.op_groups.map(op_index=>{
                        return {
                            'gpu_id': d.gpu_id,
                            'gpu_index': i,
                            'op_info': self.op_groups[op_index]
                        };
                    });
                });
            op_group_g.transition()
                .duration(duration)
                .attr('transform', function (d, i) {
                    return `translate(${self.op_group_detail_time_to_x_scale(d.op_info.start_time)}, 0)`;
                });

            let op_group_background = self.op_schedule_node_group
                .selectAll('.op_gpu_id_row_group')
                .selectAll('.op_group_g')
                .selectAll('.op_group_background').data(function (d, i) {
                return [d];
            });

            op_group_background.transition()
                .duration(duration)
                .attr('y', -self.row_gap_y / 2 + 2)
                // .attr('rx', self.row_gap_y / 2)
                // .attr('ry', self.row_gap_y / 2)
                .attr('width', function (d) {
                    return self.op_group_time_to_x_scale(d.op_info.end_time)
                        - self.op_group_time_to_x_scale(d.op_info.start_time);
                })
                .attr('height', function (d) {
                    return self.row_gap_y - 4;
                })
                .style('opacity', function (d) {
                    return 0.5;
                    // let index = self.highlight_op_index[d.op_info.index];
                    // if (index !== undefined) {
                    //     return 0.5;
                    // }
                    // else if (d.op_info.index === self.focus_op_index) {
                    //     return 0.3;
                    // }
                    // return 0;
                });


            // let op_group_number_text = self.op_schedule_node_group
            //     .selectAll('.op_gpu_id_row_group')
            //     .selectAll('.op_group_g')
            //     .selectAll('.op_group_number_text').data(function (d, i) {
            //     return [d];
            // });
            //
            // op_group_number_text.transition()
            //     .duration(duration)
            //     .text(function (d) {
            //         let end_x = self.op_group_time_to_x_scale(d.op_info.end_time),
            //             start_x = self.op_group_time_to_x_scale(d.op_info.start_time);
            //         if (end_x - start_x > 50) {
            //             return `${d.op_info.connected_number}`;
            //         }
            //         return '';
            //     })
            //     .attr('x', function (d) {
            //         let end_x = self.op_group_time_to_x_scale(d.op_info.end_time),
            //             start_x = self.op_group_time_to_x_scale(d.op_info.start_time);
            //         return (end_x - start_x) / 2;
            //     })
            //     .attr('y', -2);
            //
            // let op_group_path = self.op_schedule_node_group
            //     .selectAll('.op_gpu_id_row_group')
            //     .selectAll('.op_group_g')
            //     .selectAll('.op_group_path')
            //     .data(function (d, i) {
            //         return [d];
            //     });
            //
            // op_group_path.transition()
            //     .duration(duration)
            //     .attr('d', function (d) {
            //         return path_d([[self.row_gap_y / 2, 0],
            //             [self.op_group_time_to_x_scale(d.op_info.end_time)
            //             - self.op_group_time_to_x_scale(d.op_info.start_time)
            //             - self.row_gap_y / 2, 0]]);
            //     });
            //
            // let op_group_circle = self.op_schedule_node_group.selectAll('.op_gpu_id_row_group')
            //     .selectAll('.op_group_g')
            //     .selectAll('.op_group_circle')
            //     .data(function (d, i) {
            //         let cx1 = self.op_group_circle_r, cx2 = self.op_group_time_to_x_scale(d.op_info.end_time)
            //                 - self.op_group_time_to_x_scale(d.op_info.start_time)
            //                 - self.op_group_circle_r;
            //         if (cx1 > cx2) {
            //             cx1 = (cx1 + cx2) / 2;
            //             cx2 = cx1;
            //         }
            //         return [{
            //             'op_index': d.op_info.index,
            //             'cx': cx1,
            //             'cy': 0,
            //             'r': self.op_group_circle_r
            //         }, {
            //             'op_index': d.op_info.index,
            //             'cx': cx2,
            //             'cy': 0,
            //             'r': self.op_group_circle_r
            //         }];
            //     });
            //
            // op_group_circle.transition()
            //     .duration(duration)
            //     .attr('cx', d=>d.cx)
            //     .attr('cy', d=>d.cy)
            //     .style('fill', function (d, i) {
            //         if (i === 0 ) {
            //             return color_manager.node_border_color;
            //         }
            //         return `white`;
            //     })
            //     .attr('r', d=>d.r);

            self.op_schedule_background_group.selectAll('#op_schedule_background_rect')
                .transition()
                .duration(duration)
                .style('opacity', function () {
                    if (self.checkpoints_update_state === 'add') {
                        return 0.3;
                    }
                    return 0;
                })
                .attr('width', self.statistic_width)
                .attr('height', self.op_group_detail_height);

            let temp_data = [{
                id: 0,
                x: self.temp_op_schedule_range.start_x
            }, {
                id: 1,
                x: self.temp_op_schedule_range.end_x
            }];

            self.op_schedule_checkpoints_group.selectAll('.op_schedule_temp_checkpoints_path').data(temp_data, function (d) {
                return d.id;
            })
                .transition()
                .duration(duration)
                .attr('d', function (d) {
                    return path_d([
                        [d.x,
                            self.settings.margin.top],
                        [d.x,
                            self.settings.margin.top + self.statistic_height - self.op_bar_height]]);
                })
                .style('opacity', function (d) {
                    if (self.checkpoints_update_state === 'add') {
                        return 1;
                    }
                    return 0;
                });

            self.op_schedule_background_group
                .selectAll('.op_schedule_temp_range_background_rect')
                .data([self.temp_op_schedule_range])
                .transition()
                .duration(duration)
                .attr('x', function (d) {
                    return d.start_x;
                })
                .attr('y', self.settings.margin.top)
                .attr('width', function (d) {
                    return d.end_x - d.start_x;
                })
                .attr('height', self.op_group_detail_height)
                .style('opacity', function (d) {
                    if (self.checkpoints_update_state === 'add') {
                        return 1;
                    }
                    return 0;
                });

            self.op_schedule_background_group
                .selectAll('.op_schedule_temp_range_bar_rect')
                .data([self.temp_op_schedule_range])
                .transition()
                .duration(duration)
                .attr('x', function (d) {
                    return d.start_x;
                })
                .attr('y', self.settings.margin.top + self.statistic_height - self.op_bar_height)
                .attr('width', function (d) {
                    return d.end_x - d.start_x;
                })
                .attr('height', self.op_bar_height)
                .style('opacity', function (d) {
                    if (self.checkpoints_update_state === 'add') {
                        return 0.9;
                    }
                    return 0;
                });
        },
        _update_axis_stable_text_and_path: function (duration) {
            let self = this;

            // y axis
            self.statistics_group.selectAll('#axis-y-0')
                .transition().duration(duration)
                .attr('d', path_d([
                    [self.settings.margin.left,
                        3],
                    [self.settings.margin.left,
                        self.statistic_height + self.settings.margin.top]]))
            ;
            self.statistics_group.selectAll('#axis-y-1')
                .transition().duration(duration)
                .attr('d', path_d([
                    [self.settings.margin.left,
                        self.settings.margin.top],
                    [self.settings.margin.left + 6,
                        self.settings.margin.top]]));

            self.statistics_group.selectAll('#axis-y-2')
                .transition().duration(duration)
                .attr('d', path_d([
                    [self.settings.margin.left,
                        self.statistic_height + self.settings.margin.top],
                    [self.settings.margin.left + 6,
                        self.statistic_height + self.settings.margin.top]]));

            self.statistics_group.selectAll('#axis-y-arrow')
                .transition().duration(duration)
                .attr('d',
                    arrow_path_d(self.settings.margin.left - 3, 3, 6, 9, 'top'));

            // x axis
            self.statistics_group.selectAll('#op_schedule_axis-x')
                .transition().duration(duration)
                .attr('d', path_d([
                    [self.statistic_width + self.settings.margin.left,
                        self.statistic_height + self.settings.margin.top],
                    [self.statistic_width + self.settings.margin.left + 20,
                        self.statistic_height + self.settings.margin.top]]));
            self.statistics_group.selectAll('#op_schedule_axis-x-arrow')
                .transition().duration(duration)
                .attr('d',
                    arrow_path_d(self.statistic_width + self.settings.margin.left + 11,
                        self.statistic_height + self.settings.margin.top - 3,
                        9, 6, 'right'));

            self.statistics_group.selectAll('#op_schedule_axis-x-label')
                .transition().duration(duration)
                .attr('x', self.settings.margin.left + self.statistic_width - 10)
                .attr('y', self.settings.margin.top + self.statistic_height - 15)

            // y axis
            self.op_schedule_group.selectAll('#axis-y-3')
                .transition().duration(duration)
                .attr('d', path_d([
                    [self.settings.margin.left,
                        3],
                    [self.settings.margin.left,
                        self.statistic_height + self.settings.margin.top]]));
            self.op_schedule_group.selectAll('#op_schedule_axis-y-arrow')
                .transition().duration(duration)
                .attr('d', arrow_path_d(self.settings.margin.left - 3, 3, 6, 9, 'top'));

            // x axis
            self.op_schedule_group.selectAll('#op_schedule_axis-x')
                .transition().duration(duration)
                .attr('d', path_d([
                    [self.statistic_width + self.settings.margin.left,
                        self.statistic_height + self.settings.margin.top],
                    [self.statistic_width + self.settings.margin.left + 20,
                        self.statistic_height + self.settings.margin.top]]));

            self.op_schedule_group.selectAll('#op_schedule_axis-x-arrow')
                .transition().duration(duration)
                .attr('d', arrow_path_d(
                    self.statistic_width + self.settings.margin.left + 11,
                    self.statistic_height + self.settings.margin.top - 3,
                    9, 6, 'right'));

            self.op_schedule_group.selectAll('#op_schedule_axis-x-label')
                .transition().duration(duration)
                .attr('x', self.settings.margin.left + self.statistic_width - 10)
                .attr('y', self.settings.margin.top + self.statistic_height - 15)

            self.op_schedule_group.selectAll('#op_schedule_iter_label')
                .transition().duration(duration)
                .attr('x', self.settings.margin.left + self.statistic_width);
        },
        _generate_chart_for_zoomed:function(){
            let self = this;
            self._remove_op_schedule_for_zoomed();
            self._create_op_schedule_for_zoomed();
            init_group_click_menu();
            self._update_op_schedule_for_zoomed();
        },
        _remove_op_schedule_for_zoomed: function () {
            let self = this;

            let op_group_g = self.op_schedule_node_group
                .selectAll('.op_gpu_id_row_group')
                .selectAll('.op_group_g')
                .data(function (d, i) {
                    return d.op_groups.map(op_index=>{
                        return {
                            'gpu_id': d.gpu_id,
                            'gpu_index': i,
                            'op_info': self.op_groups[op_index]
                        };
                    });
                });
            op_group_g.exit().remove();
        },
        _create_op_schedule_for_zoomed: function () {
            let self = this;
            let op_gpu_id_row_group = self.op_schedule_node_group.selectAll('.op_gpu_id_row_group').data(self.schedule);
            op_gpu_id_row_group.enter()
                .append('g')
                .attr('class', 'op_gpu_id_row_group')
                .attr('id', function (d) {
                    return `op_gpu_id_row_group_${d.gpu_id}`;
                })
                .attr('cursor', "pointer")
                .attr('transform', function (d, i) {
                    return `translate(0, ${self.row_gap_y * (i + 0.5) + self.settings.margin.top})`;
                });

            let op_group_g = self.op_schedule_node_group
                .selectAll('.op_gpu_id_row_group')
                .selectAll('.op_group_g')
                .data(function (d, i) {
                return d.op_groups.map(op_index=>{
                    return {
                        'gpu_id': d.gpu_id,
                        'gpu_index': i,
                        'op_info': self.op_groups[op_index]
                    };
                });
            });
            let op_group_g_enter = op_group_g.enter()
                .append('g')
                .attr('class', function (d) {
                    if (self.show_dependency_op_groups.indexOf(d.op_info.index) === -1) {
                        return 'op_group_g op_group_g_show';
                    }
                    return 'op_group_g op_group_g_hide';
                })
                .attr('id', function (d) {
                    return `op_group_g_${d.op_info.index}`;
                })
                .attr('cursor', "pointer")
                .attr('transform', function (d, i) {
                    return `translate(${self.op_group_detail_time_to_x_scale(d.op_info.start_time)}, 0)`;
                })
                .on("mouseenter", function (d) {
                    self.focus_op_index = d.op_info.index;
                    if (self.highlight_op_index[d.op_info.index] === undefined) {
                        // self.op_schedule_group.selectAll(`#op_group_background_${d.op_info.index}`)
                        // .transition().duration(self.duration)
                        // .style('opacity', 0.3);
                        // self.op_group_highlight_click_manager.try_click(function () {
                        //     highlight_network_dots(self.op_groups[d.op_info.index], -1, 'add');
                        // });
                    }
                    // self.op_schedule_dependency_group
                    //     .selectAll(`#op_node_dependency_group_${d.op_info.index}`)
                    //     .transition().duration(self.duration).style('opacity', 1);

                    // let keys = ['index', 'file_path', 'jit_key'];
                    // let max_size = 0;
                    // keys.forEach((key, i)=>{
                    //     max_size = Math.max(max_size, short_string(key + ':' + d.op_info[key], 1000).length);
                    // });
                    //
                    // self.info_rect_width = max_size * 7.5;
                    // let x = self.mouse_pos.x + 15;
                    // if (self.statistic_width - x < self.info_rect_width && x > self.statistic_width / 2) {
                    //     x -= self.info_rect_width + 30;
                    // }
                    // self.op_schedule_info_group.selectAll('#op_schedule_info_background')
                    //     .transition().duration(self.duration)
                    //     .attr('width', self.info_rect_width)
                    //     .attr('height', self.statistic_height - 5 - self.op_bar_height);
                    // keys.forEach((key, i)=>{
                    //     self.op_schedule_info_group.selectAll(`#${key}-info`)
                    //     .transition().duration(self.duration).text(short_string(key + ':' + d.op_info[key], 1000))
                    //         .attr('x', 2)
                    //         .attr('y', 2 + i * 15);
                    // });
                    //
                    // self.op_schedule_info_group
                    //     .transition().duration(self.duration)
                    //     .attr('transform', `translate(${x}, 0)`)
                    //     .style('opacity', 1);
                })
                .on("mouseleave", function (d) {
                    if (self.highlight_op_index[d.op_info.index] === undefined && d.op_info.index !== self.showing_detail_info) {
                        // self.op_schedule_group.selectAll(`#op_group_background_${d.op_info.index}`)
                        //     .transition().duration(self.duration)
                        //     .style('opacity', 0);
                    }
                    // self.op_schedule_dependency_group
                    //     .selectAll(`#op_node_dependency_group_${d.op_info.index}`)
                    //     .transition().duration(self.duration).style('opacity', 0);


                    // self.info_rect_width = 0;
                    // self.op_schedule_info_group
                    //     .transition().duration(self.duration)
                    //     .style('opacity', 0);
                })
                .on("click", function (d) {
                    if (d.op_info.end_time === d.op_info.end_time) {
                        let index = self.highlight_op_index[d.op_info.index];
                        if (index === undefined) {
                            self.highlight_op_index[d.op_info.index] = self.highlight_op_index_manager.get_id();
                            self.op_schedule_group.selectAll(`#op_group_background_${d.op_info.index}`)
                            .transition().duration(self.duration)
                            // .style('opacity', 0.5)
                            .style('fill', color_manager.get_op_group_highlight_color(self.highlight_op_index[d.op_info.index]));
                            self.op_group_highlight_click_manager.try_click(function () {
                                // highlight_network_dots(self.op_groups[d.op_info.index], -1, 'delete');
                                highlight_network_dots(self.op_groups[d.op_info.index], self.highlight_op_index[d.op_info.index], 'add');
                            });
                        }
                        else {
                        self.op_schedule_group.selectAll(`#op_group_background_${d.op_info.index}`)
                        .transition().duration(self.duration)
                        // .style('opacity', 0.3)
                        .style('fill', color_manager.get_color_by_label(d.op_info.jit_type));
                        self.highlight_op_index_manager.del_id(self.highlight_op_index[d.op_info.index]);
                        self.op_group_highlight_click_manager.try_click(function () {
                            highlight_network_dots(self.op_groups[d.op_info.index], self.highlight_op_index[d.op_info.index], 'delete');
                            // highlight_network_dots(self.op_groups[d.op_info.index], -1, 'add');
                            delete self.highlight_op_index[d.op_info.index];
                        });

                    }
                    }
                });

            let op_group_background = op_group_g_enter.selectAll('.op_group_background').data(function (d, i) {
                return [d];
            });

            op_group_background.enter()
                .append('rect')
                .attr('class', 'op_group_background')
                .attr('id', function (d) {
                    return `op_group_background_${d.op_info.index}`;
                })
                .attr('x', 0)
                .attr('y', -self.row_gap_y / 2 + 2)
                // .attr('rx', self.row_gap_y / 2)
                // .attr('ry', self.row_gap_y / 2)
                .attr('width', function (d) {
                    return self.op_group_detail_time_to_x_scale(d.op_info.end_time)
                        - self.op_group_detail_time_to_x_scale(d.op_info.start_time);
                })
                .attr('height', function (d) {
                    return self.row_gap_y - 4;
                })
                .style('fill', function (d) {
                    return color_manager.get_color_by_label(d.op_info.jit_type);
                })
                .style('opacity', function (d) {
                    return 0.5;
                    // let index = self.highlight_op_index[d.op_info.index];
                    // if (index !== undefined) {
                    //     return 0.5;
                    // }
                    // else if (d.op_info.index === self.focus_op_index) {
                    //     return 0.3;
                    // }
                    // return 0;
                });

            let op_group_number_text = op_group_g_enter.selectAll('.op_group_number_text').data(function (d, i) {
                return [d];
            });

            op_group_number_text.enter()
                .append('text')
                .attr('class', 'op_group_number_text')
                .attr('id', function (d) {
                    return `op_group_number_text_${d.op_info.index}`;
                })
                .text(function (d) {
                    let end_x = self.op_group_detail_time_to_x_scale(d.op_info.end_time),
                        start_x = self.op_group_detail_time_to_x_scale(d.op_info.start_time);
                    if (end_x - start_x > 50) {
                        return `${d.op_info.connected_number}`;
                    }
                    return '';
                })
                .style('dominant-baseline', 'auto')
                .style('text-anchor', 'middle')
                .attr('x', function (d) {
                    let end_x = self.op_group_detail_time_to_x_scale(d.op_info.end_time),
                        start_x = self.op_group_detail_time_to_x_scale(d.op_info.start_time);
                    return (end_x - start_x) / 2;
                })
                .attr('y', -2);

            // let op_group_path = op_group_g_enter.selectAll('.op_group_path').data(function (d, i) {
            //     return [d];
            // });
            //
            // op_group_path.enter()
            //     .append('path')
            //     .attr('class', 'op_group_path')
            //     .attr('id', function (d) {
            //         return `op_group_path_${d.op_info.index}`;
            //     })
            //     .attr('d', function (d) {
            //         return path_d([[self.row_gap_y / 2, 0],
            //             [self.op_group_time_to_x_scale(d.op_info.end_time)
            //             - self.op_group_time_to_x_scale(d.op_info.start_time)
            //             - self.row_gap_y / 2, 0]]);
            //     })
            //     .style('stroke', color_manager.text_color)
            //     .style('stroke-width', '1px')
            //     .style('fill', 'none');
            //
            // let op_group_circle = op_group_g_enter.selectAll('.op_group_circle').data(function (d, i) {
            //     let cx1 = self.op_group_circle_r, cx2 = self.op_group_time_to_x_scale(d.op_info.end_time)
            //             - self.op_group_time_to_x_scale(d.op_info.start_time)
            //             - self.op_group_circle_r;
            //     if (cx1 > cx2) {
            //         cx1 = (cx1 + cx2) / 2;
            //         cx2 = cx1;
            //     }
            //     return [{
            //         'op_index': d.op_info.index,
            //         'cx': cx1,
            //         'cy': 0,
            //         'r': self.op_group_circle_r
            //     }, {
            //         'op_index': d.op_info.index,
            //         'cx': cx2,
            //         'cy': 0,
            //         'r': self.op_group_circle_r
            //     }];
            // });
            //
            // op_group_circle.enter()
            //     .append('circle')
            //     .attr('class', 'op_group_circle')
            //     .attr('id', function (d, i) {
            //         return `op_group_circle_${d.op_index}_${i}`;
            //     })
            //     .attr('cursor', "pointer")
            //     .attr('cx', d=>d.cx)
            //     .attr('cy', d=>d.cy)
            //     .attr('r', d=>d.r)
            //     .style('fill', function (d, i) {
            //         if (i === 0 ) {
            //             return color_manager.node_border_color;
            //         }
            //         return `white`;
            //     })
            //     .style('stroke', color_manager.node_border_color)
            //     .style('stroke-width', 1);
        },
        _update_op_schedule_for_zoomed: function () {
            let self = this;
            let duration = 100;
            let op_schedule_checkpoints_path_g = self.op_schedule_checkpoints_group.selectAll('.op_schedule_checkpoints_path_g').data(self.op_schedule_checkpoints, function (d) {
                return `${d.id}`;
            });

            op_schedule_checkpoints_path_g
                    .transition()
                    .duration(self.zoomed_duration)
                .attr('transform', function (d) {
                    return `translate(${d.x - self.op_schedule_checkpoints_width / 2}, ${self.settings.margin.top})`;
                });

            self.op_schedule_spring_group
                .selectAll('#op_schedule_spring_path')
                .transition()
                .duration(self.zoomed_duration)
                .attr('d', function (d) {
                    return spring_path_d(0, self.max_op_end_time, self.settings.margin.top + self.statistic_height + self.spring_delta_y + 3,
                        self.spring_delta_y, self.spring_num, self.op_group_time_to_x_scale);
                })
                .style('stroke-width', 2)

            self.op_schedule_node_group
                .selectAll('.op_gpu_id_row_group')
                .data(self.schedule);

            let op_group_g = self.op_schedule_node_group
                .selectAll('.op_gpu_id_row_group')
                .selectAll('.op_group_g')
                .data(function (d, i) {
                    return d.op_groups.map(op_index=>{
                        return {
                            'gpu_id': d.gpu_id,
                            'gpu_index': i,
                            'op_info': self.op_groups[op_index]
                        };
                    });
                });
            op_group_g
                .transition()
                .duration(duration)
                .attr('transform', function (d, i) {
                    return `translate(${self.op_group_detail_time_to_x_scale(d.op_info.start_time)}, 0)`;
                });

            let op_group_background = self.op_schedule_node_group
                .selectAll('.op_gpu_id_row_group')
                .selectAll('.op_group_g')
                .selectAll('.op_group_background').data(function (d, i) {
                return [d];
            });

            op_group_background
                .transition()
                .duration(duration)
                .attr('y', -self.row_gap_y / 2 + 2)
                // .attr('rx', self.row_gap_y / 2)
                // .attr('ry', self.row_gap_y / 2)
                .attr('width', function (d) {
                    return self.op_group_detail_time_to_x_scale(d.op_info.end_time)
                        - self.op_group_detail_time_to_x_scale(d.op_info.start_time);
                })
                .attr('height', function (d) {
                    return self.row_gap_y - 4;
                })
                .style('opacity', function (d) {
                    return 0.5;
                    // let index = self.highlight_op_index[d.op_info.index];
                    // if (index !== undefined) {
                    //     return 0.5;
                    // }
                    // else if (d.op_info.index === self.focus_op_index) {
                    //     return 0.3;
                    // }
                    // return 0;
                });

            let op_group_number_text = self.op_schedule_node_group
                .selectAll('.op_gpu_id_row_group')
                .selectAll('.op_group_g')
                .selectAll('.op_group_number_text').data(function (d, i) {
                return [d];
            });

            op_group_number_text.transition()
                .duration(duration)
                .text(function (d) {
                    let end_x = self.op_group_detail_time_to_x_scale(d.op_info.end_time),
                        start_x = self.op_group_detail_time_to_x_scale(d.op_info.start_time);
                    if (end_x - start_x > 50) {
                        return `${d.op_info.connected_number}`;
                    }
                    return '';
                })
                .attr('x', function (d) {
                    let end_x = self.op_group_detail_time_to_x_scale(d.op_info.end_time),
                        start_x = self.op_group_detail_time_to_x_scale(d.op_info.start_time);
                    return (end_x - start_x) / 2;
                })
                .attr('y', -2);

            self.op_schedule_background_group.selectAll('#op_schedule_background_rect')
                .transition()
                .duration(self.zoomed_duration)
                .attr('width', self.statistic_width)
                .attr('height', self.op_group_detail_height);

            let op_schedule_range_background_g = self.op_schedule_background_group.selectAll('.op_schedule_range_background_g').data(self.op_schedule_range_data, function (d) {
                return d.index;
            });
            op_schedule_range_background_g
                .transition()
                .duration(self.zoomed_duration)
                .attr('transform', function (d) {
                    return `translate(${d.start_x}, ${self.settings.margin.top})`;
                });

            op_schedule_range_background_g.selectAll('.op_schedule_range_background_rect').data(function (d) {
                return [d];
            })
                .transition()
                .duration(self.zoomed_duration)
                .attr('width', function (d) {
                    return d.end_x - d.start_x;
                });

            op_schedule_range_background_g.selectAll('.op_schedule_range_bar_rect').data(function (d) {
                return [d];
            })
                .transition()
                .duration(self.zoomed_duration)
                .attr('y', self.statistic_height - self.op_bar_height)
                .attr('width', function (d) {
                    return d.end_x - d.start_x;
                })
                .attr('height', self.op_bar_height);

            op_schedule_range_background_g.selectAll('.op_schedule_range_bar_del_btn_g').data(function (d) {
                return [d];
            })
                .transition()
                .duration(self.zoomed_duration)
                .attr('transform', function (d) {
                    return `translate(${d.end_x - d.start_x - self.op_schedule_range_del_btn_size}, ${- self.op_schedule_range_del_btn_size})`;
                });

            op_schedule_range_background_g.selectAll('.op_schedule_range_bar_center_rect').data(function (d) {
                return [d];
            })
                .transition()
                .duration(self.zoomed_duration)
                .attr('x', function (d) {
                    return (d.end_x - d.start_x) / 10;
                })
                .attr('y', self.statistic_height - self.op_bar_height)
                .attr('width', function (d) {
                    return (d.end_x - d.start_x) * 0.8;
                })
                .attr('height', self.op_bar_height);

            op_schedule_range_background_g.selectAll('.op_schedule_range_bar_left_icon').data(function (d) {
                return [d];
            })
                .transition()
                .duration(self.zoomed_duration)

                .attr('d', function (d) {
                    return op_schedule_bar_icon_path_d(0, (d.end_x - d.start_x) / 10, self.statistic_height - self.op_bar_height, self.statistic_height, 'left');
                });

            op_schedule_range_background_g.selectAll('.op_schedule_range_bar_right_icon').data(function (d) {
                return [d];
            })
                .transition()
                .duration(self.zoomed_duration)
                .attr('d', function (d) {
                    return op_schedule_bar_icon_path_d((d.end_x - d.start_x) * 0.9, d.end_x - d.start_x, self.statistic_height - self.op_bar_height, self.statistic_height, 'right');
                });

            op_schedule_range_background_g.selectAll('.op_schedule_range_bar_left_rect').data(function (d) {
                return [d];
            })
                .transition()
                .duration(self.zoomed_duration)
                .attr('x', 0)
                .attr('y', self.statistic_height - self.op_bar_height)
                .attr('width', function (d) {
                    return (d.end_x - d.start_x) / 10 + self.op_bar_height;
                })
                .attr('height', self.op_bar_height);

            op_schedule_range_background_g.selectAll('.op_schedule_range_bar_right_rect').data(function (d) {
                return [d];
            })
                .transition()
                .duration(self.zoomed_duration)
                .attr('x', function (d) {
                    return (d.end_x - d.start_x) * 0.9 - self.op_bar_height;
                })
                .attr('y', self.statistic_height - self.op_bar_height)
                .attr('width', function (d) {
                    return (d.end_x - d.start_x) / 10 + self.op_bar_height;
                })
                .attr('height', self.op_bar_height);

            self.op_schedule_dependency_group
                .selectAll('.op_node_dependency_group')
                .selectAll('.op_node_dependency_path')
                .data(function (d) {
                    let sorted_index = [];
                    d.depend_op_index.forEach(depend_index=>{
                        let depend_d = self.op_groups[depend_index];
                        let cx1 = self.op_group_circle_r, cx2 = self.op_group_time_to_x_scale(d.end_time)
                            - self.op_group_time_to_x_scale(d.start_time)
                            - self.op_group_circle_r;
                        if (cx1 > cx2) {
                            cx1 = (cx1 + cx2) / 2;
                            cx2 = cx1;
                        }

                        let start_x = self.op_group_time_to_x_scale(d.start_time) + cx1;
                        cx1 = self.op_group_circle_r;
                        cx2 = self.op_group_time_to_x_scale(depend_d.end_time)
                            - self.op_group_time_to_x_scale(depend_d.start_time)
                            - self.op_group_circle_r;
                        if (cx1 > cx2) {
                            cx1 = (cx1 + cx2) / 2;
                            cx2 = cx1;
                        }
                        let end_x = self.op_group_time_to_x_scale(depend_d.start_time) + cx2;
                        if (start_x !== end_x) {
                            sorted_index.push(depend_index);
                        }
                    });

                    sorted_index.sort(function(a, b) {
                        let depend_a = self.op_groups[a], depend_b = self.op_groups[b];
                        if (depend_a.end_time !== depend_b.end_time) {
                            return depend_a.end_time - depend_b.end_time;
                        }
                        else {
                            return depend_a.schedule_index - depend_b.schedule_index;
                        }
                    });
                    return d.depend_op_index.map(depend_index=>{
                        let depend_d = self.op_groups[depend_index];
                        let cx1 = self.op_group_circle_r, cx2 = self.op_group_time_to_x_scale(d.end_time)
                            - self.op_group_time_to_x_scale(d.start_time)
                            - self.op_group_circle_r;
                        if (cx1 > cx2) {
                            cx1 = (cx1 + cx2) / 2;
                            cx2 = cx1;
                        }

                        let start_x = self.op_group_time_to_x_scale(d.start_time) + cx1;
                        cx1 = self.op_group_circle_r;
                        cx2 = self.op_group_time_to_x_scale(depend_d.end_time)
                            - self.op_group_time_to_x_scale(depend_d.start_time)
                            - self.op_group_circle_r;
                        if (cx1 > cx2) {
                            cx1 = (cx1 + cx2) / 2;
                            cx2 = cx1;
                        }
                        let end_x = self.op_group_time_to_x_scale(depend_d.start_time) + cx2;
                        let margin = 0;
                        let points = [{
                                x: start_x + margin,
                                y: self.settings.margin.top + self.row_gap_y * (d.schedule_index + 0.5)
                            }];

                        if (start_x !== end_x) {
                            let y_ratio = (sorted_index.indexOf(depend_index) + 1) / (sorted_index.length + 1);
                            let num = Math.floor((start_x - end_x) / 5);
                            for (let i = 1;i < num;i++) {
                                points.push({
                                    x: (start_x * (num - i) + end_x * i) / num,
                                    y: (self.settings.margin.top - 5) * y_ratio
                                });
                            }
                        }
                        else {
                            points.push({
                                x: start_x + margin / 3,
                                y: self.settings.margin.top + self.row_gap_y * (d.schedule_index + 0.5)
                            });
                            points.push({
                                x: end_x - margin / 3,
                                y: self.settings.margin.top + self.row_gap_y * (depend_d.schedule_index + 0.5)
                            });
                        }
                        points.push({
                                x: end_x - margin,
                                y: self.settings.margin.top + self.row_gap_y * (depend_d.schedule_index + 0.5)
                            });
                        return {
                            'points': points,
                            'name': `${d.index}_${depend_index}`
                        };
                    });
                })
                .transition()
                .duration(self.zoomed_duration)
                .attr('d', function (d) {
                    return cardinal_line(d.points);
                });
        },
        _generate_dependency_group: function() {
            let self = this;
            let show_dependency_op_groups = self.show_dependency_op_groups.map(x=>self.op_groups[x]);
            let op_node_dependency_group = self.op_schedule_dependency_group.selectAll('.op_node_dependency_group').data(show_dependency_op_groups);

            let op_node_dependency_path = self.op_schedule_dependency_group.selectAll('.op_node_dependency_group').selectAll('.op_node_dependency_path')
                .data(function (d) {
                    let sorted_index = [];
                    d.depend_op_index.forEach(depend_index=>{
                        let depend_d = self.op_groups[depend_index];
                        let cx1 = self.op_group_circle_r, cx2 = self.op_group_time_to_x_scale(d.end_time)
                            - self.op_group_time_to_x_scale(d.start_time)
                            - self.op_group_circle_r;
                        if (cx1 > cx2) {
                            cx1 = (cx1 + cx2) / 2;
                            cx2 = cx1;
                        }

                        let start_x = self.op_group_time_to_x_scale(d.start_time) + cx1;
                        cx1 = self.op_group_circle_r;
                        cx2 = self.op_group_time_to_x_scale(depend_d.end_time)
                            - self.op_group_time_to_x_scale(depend_d.start_time)
                            - self.op_group_circle_r;
                        if (cx1 > cx2) {
                            cx1 = (cx1 + cx2) / 2;
                            cx2 = cx1;
                        }
                        let end_x = self.op_group_time_to_x_scale(depend_d.start_time) + cx2;
                        if (start_x !== end_x) {
                            sorted_index.push(depend_index);
                        }
                    });

                    sorted_index.sort(function(a, b) {
                        let depend_a = self.op_groups[a], depend_b = self.op_groups[b];
                        if (depend_a.end_time !== depend_b.end_time) {
                            return depend_a.end_time - depend_b.end_time;
                        }
                        else {
                            return depend_a.schedule_index - depend_b.schedule_index;
                        }
                    });
                    return d.depend_op_index.map(depend_index=>{
                        let depend_d = self.op_groups[depend_index];
                        let cx1 = self.op_group_circle_r, cx2 = self.op_group_time_to_x_scale(d.end_time)
                            - self.op_group_time_to_x_scale(d.start_time)
                            - self.op_group_circle_r;
                        if (cx1 > cx2) {
                            cx1 = (cx1 + cx2) / 2;
                            cx2 = cx1;
                        }

                        let start_x = self.op_group_time_to_x_scale(d.start_time) + cx1;
                        cx1 = self.op_group_circle_r;
                        cx2 = self.op_group_time_to_x_scale(depend_d.end_time)
                            - self.op_group_time_to_x_scale(depend_d.start_time)
                            - self.op_group_circle_r;
                        if (cx1 > cx2) {
                            cx1 = (cx1 + cx2) / 2;
                            cx2 = cx1;
                        }
                        let end_x = self.op_group_time_to_x_scale(depend_d.start_time) + cx2;
                        let margin = 0;
                        let points = [{
                                x: start_x + margin,
                                y: self.settings.margin.top + self.row_gap_y * (d.schedule_index + 0.5)
                            }];

                        if (start_x !== end_x) {
                            let y_ratio = (sorted_index.indexOf(depend_index) + 1) / (sorted_index.length + 1);
                            let num = Math.floor((start_x - end_x) / 5);
                            for (let i = 1;i < num;i++) {
                                points.push({
                                    x: (start_x * (num - i) + end_x * i) / num,
                                    y: (self.settings.margin.top - 5) * y_ratio
                                });
                            }
                        }
                        else {
                            points.push({
                                x: start_x + margin / 3,
                                y: self.settings.margin.top + self.row_gap_y * (d.schedule_index + 0.5)
                            });
                            points.push({
                                x: end_x - margin / 3,
                                y: self.settings.margin.top + self.row_gap_y * (depend_d.schedule_index + 0.5)
                            });
                        }
                        points.push({
                                x: end_x - margin,
                                y: self.settings.margin.top + self.row_gap_y * (depend_d.schedule_index + 0.5)
                            });
                        return {
                            'points': points,
                            'name': `${d.index}_${depend_index}`
                        };
                    });
                });
            op_node_dependency_group.exit().transition().duration(self.remove_duration).style('opacity', 0);
            op_node_dependency_path.exit().transition().duration(self.remove_duration).style('opacity', 0);
            setTimeout(function () {
                op_node_dependency_group.exit().remove();
                op_node_dependency_path.exit().remove();

                op_node_dependency_group = self.op_schedule_dependency_group.selectAll('.op_node_dependency_group').data(show_dependency_op_groups);
                op_node_dependency_group.enter().append('g')
                    .attr('class', 'op_node_dependency_group')
                    .attr('id', function (d) {
                        return `op_node_dependency_group_${d.index}`;
                    });

                op_node_dependency_path = self.op_schedule_dependency_group
                    .selectAll('.op_node_dependency_group')
                    .selectAll('.op_node_dependency_path')
                    .data(function (d) {
                        let sorted_index = [];
                        d.depend_op_index.forEach(depend_index=>{
                        let depend_d = self.op_groups[depend_index];
                        let cx1 = self.op_group_circle_r, cx2 = self.op_group_time_to_x_scale(d.end_time)
                            - self.op_group_time_to_x_scale(d.start_time)
                            - self.op_group_circle_r;
                        if (cx1 > cx2) {
                            cx1 = (cx1 + cx2) / 2;
                            cx2 = cx1;
                        }

                        let start_x = self.op_group_time_to_x_scale(d.start_time) + cx1;
                        cx1 = self.op_group_circle_r;
                        cx2 = self.op_group_time_to_x_scale(depend_d.end_time)
                            - self.op_group_time_to_x_scale(depend_d.start_time)
                            - self.op_group_circle_r;
                        if (cx1 > cx2) {
                            cx1 = (cx1 + cx2) / 2;
                            cx2 = cx1;
                        }
                        let end_x = self.op_group_time_to_x_scale(depend_d.start_time) + cx2;
                            if (start_x !== end_x) {
                                sorted_index.push(depend_index);
                            }
                        });

                        sorted_index.sort(function(a, b) {
                            let depend_a = self.op_groups[a], depend_b = self.op_groups[b];
                            if (depend_a.end_time !== depend_b.end_time) {
                                return depend_a.end_time - depend_b.end_time;
                            }
                            else {
                                return depend_a.schedule_index - depend_b.schedule_index;
                            }
                        });
                        return d.depend_op_index.map(depend_index=>{
                            let depend_d = self.op_groups[depend_index];
                            let cx1 = self.op_group_circle_r, cx2 = self.op_group_time_to_x_scale(d.end_time)
                                - self.op_group_time_to_x_scale(d.start_time)
                                - self.op_group_circle_r;
                            if (cx1 > cx2) {
                                cx1 = (cx1 + cx2) / 2;
                                cx2 = cx1;
                            }

                            let start_x = self.op_group_time_to_x_scale(d.start_time) + cx1;
                            cx1 = self.op_group_circle_r;
                            cx2 = self.op_group_time_to_x_scale(depend_d.end_time)
                                - self.op_group_time_to_x_scale(depend_d.start_time)
                                - self.op_group_circle_r;
                            if (cx1 > cx2) {
                                cx1 = (cx1 + cx2) / 2;
                                cx2 = cx1;
                            }
                            let end_x = self.op_group_time_to_x_scale(depend_d.start_time) + cx2;
                            let margin = 0;
                            let points = [{
                                    x: start_x + margin,
                                    y: self.settings.margin.top + self.row_gap_y * (d.schedule_index + 0.5)
                                }];

                            if (start_x !== end_x) {
                                let y_ratio = (sorted_index.indexOf(depend_index) + 1) / (sorted_index.length + 1);
                                let num = Math.floor((start_x - end_x) / 5);
                                for (let i = 1;i < num;i++) {
                                    points.push({
                                        x: (start_x * (num - i) + end_x * i) / num,
                                        y: (self.settings.margin.top - 5) * y_ratio
                                    });
                                }
                            }
                            else {
                                points.push({
                                    x: start_x + margin / 3,
                                    y: self.settings.margin.top + self.row_gap_y * (d.schedule_index + 0.5)
                                });
                                points.push({
                                    x: end_x - margin / 3,
                                    y: self.settings.margin.top + self.row_gap_y * (depend_d.schedule_index + 0.5)
                                });
                            }
                            points.push({
                                    x: end_x - margin,
                                    y: self.settings.margin.top + self.row_gap_y * (depend_d.schedule_index + 0.5)
                                });
                            return {
                                'points': points,
                                'name': `${d.index}_${depend_index}`
                            };
                        });
                    });

                op_node_dependency_path.enter().append('path')
                    .attr('class', 'op_node_dependency_path')
                    .attr('id', function (d) {
                        return `op_node_dependency_path_${d.name}`;
                    })
                    .attr('d', function (d) {
                        return cardinal_line(d.points);
                    })
                    .style('stroke', color_manager.text_color)
                    .style('stroke-width', '1px')
                    .style('fill', 'none');
                init_group_click_menu();
                self.op_schedule_dependency_group
                    .selectAll('.op_node_dependency_group')
                    .selectAll('.op_node_dependency_path')
                    .data(function (d) {
                        let sorted_index = [];
                        d.depend_op_index.forEach(depend_index=>{
                            let depend_d = self.op_groups[depend_index];
                            let cx1 = self.op_group_circle_r, cx2 = self.op_group_time_to_x_scale(d.end_time)
                                - self.op_group_time_to_x_scale(d.start_time)
                                - self.op_group_circle_r;
                            if (cx1 > cx2) {
                                cx1 = (cx1 + cx2) / 2;
                                cx2 = cx1;
                            }

                            let start_x = self.op_group_time_to_x_scale(d.start_time) + cx1;
                            cx1 = self.op_group_circle_r;
                            cx2 = self.op_group_time_to_x_scale(depend_d.end_time)
                                - self.op_group_time_to_x_scale(depend_d.start_time)
                                - self.op_group_circle_r;
                            if (cx1 > cx2) {
                                cx1 = (cx1 + cx2) / 2;
                                cx2 = cx1;
                            }
                            let end_x = self.op_group_time_to_x_scale(depend_d.start_time) + cx2;
                            if (start_x !== end_x) {
                                sorted_index.push(depend_index);
                            }
                        });

                        sorted_index.sort(function(a, b) {
                            let depend_a = self.op_groups[a], depend_b = self.op_groups[b];
                            if (depend_a.end_time !== depend_b.end_time) {
                                return depend_a.end_time - depend_b.end_time;
                            }
                            else {
                                return depend_a.schedule_index - depend_b.schedule_index;
                            }
                        });
                        return d.depend_op_index.map(depend_index=>{
                            let depend_d = self.op_groups[depend_index];
                            let cx1 = self.op_group_circle_r, cx2 = self.op_group_time_to_x_scale(d.end_time)
                                - self.op_group_time_to_x_scale(d.start_time)
                                - self.op_group_circle_r;
                            if (cx1 > cx2) {
                                cx1 = (cx1 + cx2) / 2;
                                cx2 = cx1;
                            }

                            let start_x = self.op_group_time_to_x_scale(d.start_time) + cx1;
                            cx1 = self.op_group_circle_r;
                            cx2 = self.op_group_time_to_x_scale(depend_d.end_time)
                                - self.op_group_time_to_x_scale(depend_d.start_time)
                                - self.op_group_circle_r;
                            if (cx1 > cx2) {
                                cx1 = (cx1 + cx2) / 2;
                                cx2 = cx1;
                            }
                            let end_x = self.op_group_time_to_x_scale(depend_d.start_time) + cx2;
                            let margin = 0;
                            let points = [{
                                    x: start_x + margin,
                                    y: self.settings.margin.top + self.row_gap_y * (d.schedule_index + 0.5)
                                }];

                            if (start_x !== end_x) {
                                let y_ratio = (sorted_index.indexOf(depend_index) + 1) / (sorted_index.length + 1);
                                let num = Math.floor((start_x - end_x) / 5);
                                for (let i = 1;i < num;i++) {
                                    points.push({
                                        x: (start_x * (num - i) + end_x * i) / num,
                                        y: (self.settings.margin.top - 5) * y_ratio
                                    });
                                }
                            }
                            else {
                                points.push({
                                    x: start_x + margin / 3,
                                    y: self.settings.margin.top + self.row_gap_y * (d.schedule_index + 0.5)
                                });
                                points.push({
                                    x: end_x - margin / 3,
                                    y: self.settings.margin.top + self.row_gap_y * (depend_d.schedule_index + 0.5)
                                });
                            }
                            points.push({
                                    x: end_x - margin,
                                    y: self.settings.margin.top + self.row_gap_y * (depend_d.schedule_index + 0.5)
                                });
                            return {
                                'points': points,
                                'name': `${d.index}_${depend_index}`
                            };
                        });
                    }).transition()
                    .duration(self.duration)
                    .attr('d', function (d) {
                        return cardinal_line(d.points);
                    });
            }, self.remove_duration);
        },
        _switch_between_statistics_and_op_schedule: function (mode) {
            let self = this;
            if (mode !== self.mode) {
                self.mode = mode;
                self._generate_chart();
            }
        },
        _update_checkpoints_update_state: function (state) {
            let self = this;
            if (state !== self.checkpoints_update_state) {
                self.checkpoints_update_state = state;
                if (state === 'delete') {
                    self.op_schedule_background_group.selectAll('.op_schedule_range_bar_del_btn_g')
                                .style('opacity', 1);
                }
                else {
                    self.op_schedule_background_group.selectAll('.op_schedule_range_bar_del_btn_g')
                                .style('opacity', 0);
                }
                self._generate_chart();
            }
        },
        _show_dependency: function (flag=true) {
            let self = this;
            if (flag && self.focus_op_index !== -1) {

                d3.selectAll(`#op_group_g_${self.focus_op_index}`).classed('op_group_g_show', false);
                d3.selectAll(`#op_group_g_${self.focus_op_index}`).classed('op_group_g_hide', true);
                if (self.show_dependency_op_groups.indexOf(self.focus_op_index) === -1) {
                    self.show_dependency_op_groups.push(self.focus_op_index);
                }

                self._generate_dependency_group();
                // self.op_schedule_dependency_group
                //     .selectAll(`#op_node_dependency_group_${self.focus_op_index}`)
                //     .transition().duration(self.duration).style('opacity', 1);
            }
            else if (!flag && self.focus_op_index !== -1) {
                d3.selectAll(`#op_group_g_${self.focus_op_index}`).classed('op_group_g_show', true);
                d3.selectAll(`#op_group_g_${self.focus_op_index}`).classed('op_group_g_hide', false);
                self.show_dependency_op_groups.remove(self.focus_op_index);
                self._generate_dependency_group();
                // self.op_schedule_dependency_group
                //     .selectAll(`#op_node_dependency_group_${self.focus_op_index}`)
                //     .transition().duration(self.duration).style('opacity', 0);
            }
        },
        _show_detail_info: function () {
            let self = this;
            if (self.focus_op_index !== -1) {
                let op = self.op_groups[self.focus_op_index];
                let keys = ['index', 'start_time', 'time_cost', 'file_path', 'jit_key'];
                let max_size = 100;
                let values = {};
                keys.forEach((key, i)=>{
                    max_size = Math.max(max_size, (key + ': ' + op[key]).length);
                    values[key] = op[key];
                });

                let width = Math.min(1000, max_size * 5);
                let attrs = {
                    top: self.mouse_pos.y,
                    left: self.mouse_pos.x,
                    opacity: 1,
                    width: width,
                    values: values,
                    title: `op_group_${self.focus_op_index}`
                };
                self.showing_detail_info = self.focus_op_index;
                // self.op_schedule_group.selectAll(`#op_group_background_${self.showing_detail_info}`).style('opacity', 0.3);
                set_content_of_tooltip(attrs);
            }
        },
        _hide_detail_info: function () {
            let self = this;
            if (self.showing_detail_info !== -1) {
                // self.op_schedule_group.selectAll(`#op_group_background_${self.showing_detail_info}`)
                //         .transition().duration(self.duration).style('opacity', 0);
                self.showing_detail_info = -1;
            }
        },
        _init_overview: function () {
            let self = this;
            self.op_schedule_node_overview_group.selectAll('.op_overview_gpu_id_row_group')
                .data(self.curr_all_schedule).enter()
                .append('g')
                .attr('class', 'op_overview_gpu_id_row_group')
                .attr('id', function (d) {
                    return `op_overview_gpu_id_row_group_${d.gpu_id}`;
                })
                .attr('transform', function (d, i) {
                    return `translate(0, ${self.row_gap_y / 2 * (i + 0.5) + self.settings.margin.top})`;
                });
            self.op_schedule_node_overview_group
                .selectAll('.op_overview_gpu_id_row_group')
                .selectAll('.op_overview_group_g')
                .data(function (d, i) {
                    return d.op_groups.map(op_index=>{
                        return {
                            'gpu_id': d.gpu_id,
                            'gpu_index': i,
                            'op_info': self.op_groups[op_index]
                        };
                    });
                }).enter()
                .append('g')
                .attr('class', 'op_overview_group_g')
                .attr('id', function (d) {
                    return `op_overview_group_g_${d.op_info.index}`;
                })
                .attr('transform', function (d, i) {
                    return `translate(${self.op_group_time_to_x_scale(d.op_info.start_time)}, 0)`;
                });
            self.op_schedule_node_overview_group
                .selectAll('.op_overview_gpu_id_row_group')
                .selectAll('.op_overview_group_g')
                .selectAll('.op_overview_group_background').data(function (d, i) {
                    return [d];
                }).enter()
                .append('rect')
                .attr('class', 'op_overview_group_background')
                .attr('id', function (d) {
                    return `op_overview_group_background_${d.op_info.index}`;
                })
                .attr('x', 0)
                .attr('y', -self.row_gap_y / 4 + 2)
                .attr('width', function (d) {
                    return self.op_group_time_to_x_scale(d.op_info.end_time)
                        - self.op_group_time_to_x_scale(d.op_info.start_time);
                })
                .attr('height', function (d) {
                    return self.row_gap_y / 2 - 4;
                })
                .style('fill', function (d) {
                    return color_manager.get_color_by_label(d.op_info.jit_type);
                })
                .style('opacity', function (d) {
                    return 0.5;
                });
        },
        _update_curr_time_range: function() {
            let self = this;
            self.op_schedule_node_overview_cover_group.selectAll('#op_schedule_overview_cover_rect_left')
                .attr('width', self.op_group_time_to_x_scale(self.curr_time_range[0]) - self.op_group_time_to_x_scale(0));

            self.op_schedule_node_overview_cover_group.selectAll('#op_schedule_overview_cover_rect_right')
                .attr('x', self.op_group_time_to_x_scale(self.curr_time_range[1]))
                .attr('width', self.op_group_time_to_x_scale(self.max_op_end_time) - self.op_group_time_to_x_scale(self.curr_time_range[1]));

            self.op_schedule_node_overview_cover_group.selectAll('#op_schedule_overview_path_left')
                .attr('d', line([{
                    x: self.op_group_time_to_x_scale(self.curr_time_range[0]), y: 0
                }, {
                    x: self.op_group_time_to_x_scale(self.curr_time_range[0]), y: self.overview_chart_height
                }]));

            self.op_schedule_node_overview_cover_group.selectAll('#op_schedule_overview_path_right')
                .attr('d', line([{
                    x: self.op_group_time_to_x_scale(self.curr_time_range[1]), y: 0
                }, {
                    x: self.op_group_time_to_x_scale(self.curr_time_range[1]), y: self.overview_chart_height
                }]));

            self.op_schedule_node_overview_cover_group.selectAll('#op_schedule_overview_cover_rect_top')
                .attr('x', self.op_group_time_to_x_scale(self.curr_time_range[0]))
                .attr('width', self.op_group_time_to_x_scale(self.curr_time_range[1]) - self.op_group_time_to_x_scale(self.curr_time_range[0]));

            self.op_schedule_node_overview_cover_group.selectAll('#op_schedule_overview_cover_rect_left_top')
                .attr('x', self.op_group_time_to_x_scale(self.curr_time_range[0]) - 2.5);

            self.op_schedule_node_overview_cover_group.selectAll('#op_schedule_overview_cover_rect_right_top')
                .attr('x', self.op_group_time_to_x_scale(self.curr_time_range[1]) - 2.5);
            self.op_group_detail_time_to_x_scale = d3.scaleLinear()
                        .domain(self.curr_time_range)
                        .range([self.settings.margin.left + 6, self.settings.margin.left + self.statistic_width]);
        }
    };
    let statisticVis = new StatisticVis(options);
    statisticVis.init();
    return {
        options: statisticVis.settings,
        redraw: function () {
            statisticVis.redraw.apply(statisticVis, arguments);
        },
        resize: function(){
            statisticVis.resize.apply(statisticVis, arguments);
        },
        reset: function () {
            statisticVis._reset.apply(statisticVis, arguments);
        },
        add_selected_value: function () {
            statisticVis._add_selected_value.apply(statisticVis, arguments);
        },
        delete_selected_value: function () {
            statisticVis._delete_selected_value.apply(statisticVis, arguments);
        },
        switch_between_statistics_and_op_schedule: function () {
            statisticVis._switch_between_statistics_and_op_schedule.apply(statisticVis, arguments);
        },
        update_checkpoints_update_state: function () {
            statisticVis._update_checkpoints_update_state.apply(statisticVis, arguments);
        },
        show_dependency: function () {
            statisticVis._show_dependency.apply(statisticVis, arguments);
        },
        show_detail_info: function () {
            statisticVis._show_detail_info.apply(statisticVis, arguments);
        },
        hide_detail_info: function () {
            statisticVis._hide_detail_info.apply(statisticVis, arguments);
        }
    };
};

