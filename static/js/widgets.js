/**
 * Created by derekxiao on 2017/12/22.
 */


function SimpleLasso(svg, circles, callback) {
    var lasso_start = function() {
        lasso.items()
            .classed("not_possible",true)
            .classed("selected",false);
    };

    var lasso_draw = function() {
        lasso.possibleItems()
            .attr('r', function(d) {
                return dot_size[2] * zoom_scale;
            })
            .style('opacity', 1);
        possible_index = [];
        if (select_mode == 0) {
            selected_index = [];
            lasso.possibleItems().each(function (d, i) {
                        possible_index.push(d.id);
                    })
        }
        else {
            lasso.possibleItems().each(function (d, i) {
                        if (selected_index.indexOf(d.id) != -1) {
                            return;
                        }
                        possible_index.push(d.id);
                    })
        }
        callback(selected_index.concat(possible_index), 'possible');
    };
    var lasso_end = function() {
        lasso.selectedItems()
            .attr('r', function(d) {
                return dot_size[2] * zoom_scale;
            })
            .style('opacity', 1);

        if (select_mode == 0) {
            selected_index = [];
            lasso.selectedItems().each(function (d, i) {
                        selected_index.push(d.id);
                    })
        }
        else {
            lasso.selectedItems().each(function (d, i) {
                        if (selected_index.indexOf(d.id) != -1) {
                            return;
                        }
                        selected_index.push(d.id);
                    })
        }
        callback(selected_index, 'selected');

    };

    var lasso = d3.lasso()
        .closePathSelect(true)
        .closePathDistance(100)
        .items(circles)
        .targetArea(svg)
        .on("start",lasso_start)
        .on("draw",lasso_draw)
        .on("end",lasso_end);
    var selected_index = [];
    var possible_index = [];
    var zoom_scale = 1.0;
    var select_mode = 0;
    svg.call(lasso);
    this.clear = function () {
        selected_index = [];
        possible_index = [];
    };
    this.set_selected_index = function (index) {
        selected_index = index;
        possible_index = [];
    };
    this.set_zoom_scale = function (scale) {
        //zoom_scale = scale;
    };
    this.set_select_mode = function (mode) {
        select_mode = mode;
    };
    this.get_select_mode = function () {
        return select_mode;
    };
    return this;
}


function ImageLasso(svg, circles, callback) {
    var lasso_start = function() {
        lasso.items()
            .classed("not_possible",true)
            .classed("selected",false);
    };

    var lasso_draw = function() {
        possible_index = [];
        if (select_mode == 0) {
            selected_index = [];
            lasso.possibleItems().each(function (d, i) {
                        possible_index.push(d.id);
                    })
        }
        else {
            lasso.possibleItems().each(function (d, i) {
                        if (selected_index.indexOf(d.id) != -1) {
                            return;
                        }
                        possible_index.push(d.id);
                    })
        }
        callback(selected_index.concat(possible_index), 'possible');
    };

    var lasso_end = function() {
        if (select_mode == 0) {
            selected_index = [];
            lasso.selectedItems().each(function (d, i) {
                        selected_index.push(d.id);
                    })
        }
        else {
            lasso.selectedItems().each(function (d, i) {
                        if (selected_index.indexOf(d.id) != -1) {
                            return;
                        }
                        selected_index.push(d.id);
                    })
        }
        callback(selected_index, 'selected');
    };

    var lasso = d3.lasso()
        .closePathSelect(false)
        // .closePathDistance(100)
        .items(circles)
        .targetArea(svg)
        .on("start",lasso_start)
        .on("draw",lasso_draw)
        .on("end",lasso_end);
    var selected_index = [];
    var possible_index = [];
    var zoom_scale = 1.0;
    var select_mode = 0;
    svg.call(lasso);
    this.clear = function () {
        selected_index = [];
        possible_index = [];
    };
    this.set_selected_index = function (index) {
        selected_index = index;
        possible_index = [];
    };
    this.set_zoom_scale = function (scale) {
        //zoom_scale = scale;
    };
    this.set_select_mode = function (mode) {
        select_mode = mode;
    };
    this.get_select_mode = function () {
        return select_mode;
    };
    return this;
}