if (!String.prototype.format) {
    String.prototype.format = function () {
        var args = arguments;
        return this.replace(/{(\d+)}/g, function (match, number) {
            return typeof args[number] != 'undefined'
                ? args[number]
                : match
                ;
        });
    };
}

function window_resize(width, height) {
    WINDOW_HEIGHT = height - 40;
    WINDOW_WIDTH = width;

    if (statistic_component) {
        statistic_component.resize();
    }
    if (network_component) {
        network_component.resize();
    }
}

function list_copy(list) {
    var res = [];
    for (var i = 0;i < list.length;i++) {
        res[i] = list[i];
    }
    return res;
}

function obj_copy(obj){
    if(typeof obj !== 'object'){
        return obj;
    }
    if (obj.length === undefined) {
        var newobj = {};
        for ( var attr in obj) {
            newobj[attr] = obj_copy(obj[attr]);
        }
        return newobj;
    }
    var newobj = [];
    for (var i = 0;i < obj.length;i++) {
        newobj[i] = obj_copy(obj[i]);
    }
    return newobj;
}

function path_d(points) {
    let res = `M${points[0][0]},${points[0][1]}`;
    for (let i = 1;i < points.length;i++) {
        res += `L${points[i][0]},${points[i][1]}`;
    }
    return res;
}

function line(points) {
    let res = `M${points[0].x},${points[0].y}`;
    for (let i = 1;i < points.length;i++) {
        res += `L${points[i].x},${points[i].y}`;
    }
    return res;
}

function spring_path_d(start_time, end_time, y, delta_y, num, time_to_x_scale) {
    let time_per_spring_line = (end_time - start_time) / num / 4;
    let ys = [y + delta_y, y, y - delta_y, y];
    let res = `M${time_to_x_scale(start_time)},${y}`;
    let t = start_time + time_per_spring_line;
    /*
    for (let i = 0;i < num;i++) {
        for (let j = 0;j < 4;j++) {
            res += `L${time_to_x_scale(t)},${ys[j]}`;
            t += time_per_spring_line;
        }
    }*/
    res += `L${time_to_x_scale(end_time)},${y}`;

    return res;
}

function arrow_path_d(start_x, start_y, width, height, direction) {
    let res = '';
    if (direction === 'left') {
        res += `M${start_x + width},${start_y}`;
        res += `L${start_x},${start_y + height / 2}`;
        res += `L${start_x + width},${start_y + height}`;
    }
    else if (direction === 'right') {
        res += `M${start_x},${start_y}`;
        res += `L${start_x + width},${start_y + height / 2}`;
        res += `L${start_x},${start_y + height}`;
    }
    else if (direction === 'top') {
        res += `M${start_x},${start_y + height}`;
        res += `L${start_x + width / 2},${start_y}`;
        res += `L${start_x + width},${start_y + height}`;
    }
    else if (direction === 'bottom') {
        res += `M${start_x},${start_y}`;
        res += `L${start_x + width / 2},${start_y + height}`;
        res += `L${start_x + width},${start_y}`;
    }
    return res;
}

function lines_path_d(lines, iters, x_scale) {
    let res = '';
    for (let i = 0; i < lines.length; i++) {
        let line = lines[i];
        for (let j = 0; j < line.length; j++) {
            res += res === ''? 'M': 'L';
            res += `${x_scale(line[j]['time'])},${line[j]['y']}`;
        }
    }
    return res;
}

function add_statistic_checkbox() {
    let value = processed_data['statistics']['value'];
    let statistics = value['iter'].concat(value['time']);
    for (let i = 0;i < statistics.length;i++) {
        //<a class="dropdown-item" href="#">Link 1</a>

        d3.select('#value_menu').append('a')
            .attr('class', 'dropdown-item')
            .attr('id', `dropdown-item_${i}`)
            .text(statistics[i])
            .on('click', function (d) {
                d3.select('#value_menu_title').text(statistics[i]);
                d3.select('#axis-y-label').text(statistics[i]);
                statistic_component.add_selected_value(statistics[i]);
            });
    }
    if (statistics.length > 0) {
        document.getElementById('dropdown-item_0').click();
    }
}

function add_intro() {
    introJs("#AnalyzerView").setOptions({
        disableInteraction: true,
        steps: [{
          intro: "Welcome to JittorVis!"
        }, {
          element: document.querySelector('#ControlPanel'),
          intro: "Click here to toggle different views"
        },
        {
            element: document.querySelector('#guideview'),
            intro: "This is the navigation view."
        },
        {
            element: document.querySelector('.tree-btn'),
            intro: "Each leaf node represents a computational node in the computational graph."
        },
        {
            element: document.querySelector('.tree-btn'),
            intro: "Click one intermediate node to selected its computational nodes."
        },
        {
            element: document.querySelector('#networkView'),
            intro: "This is the graph structure view, which shows the graph structure of selected computational nodes."
        },
        {
            element: document.querySelector('#networkView'),
            intro: "Each rectangle represents a computational node."
        },
        {
            element: document.querySelector('#networkView'),
            intro: "Each link represents data flows among computational nodes."
        },{
            element: document.querySelector('body'),
            intro: "Start to explore your model in JittorVis!"
        },
        //{
        //    element: document.querySelector('.op_node_group:last-child'),
        //    intro: "Hover on rounded rectangle to explore more"
        //},
        ],
      }).start();
}

// function compute_descendants_pre_next(node, tree, size_threshold, root) {
//     let descendants = compute_descendants(node, tree);
//     if (node['level'] === 0 || node['size'] >= size_threshold) {
//         node['descendants'] = descendants;
//         return;
//     }
//     let pre = [];
//     let next = [];
//     let descendants_flag = new Array(tree[root]['size']).fill(0);
//     for (let i = 0;i < descendants.length;i++) {
//         descendants_flag[i] = 1;
//     }
//     for (let i = 0;i < descendants.length;i++) {
//         let descendant = descendants[i];
//         for (let j = 0; j < descendant['pre'].length;j++) {
//             if (descendants_flag[descendant['pre'][j]] === 0) {
//                 pre.push(descendant['pre'][j]);
//             }
//         }
//         for (let j = 0; j < descendant['next'].length;j++) {
//             if (descendants_flag[descendant['next'][j]] === 0) {
//                 next.push(descendant['next'][j]);
//             }
//         }
//     }
//     node['descendants'] = descendants;
//     node['pre'] = pre;
//     node['next'] = next;
// }

// function compute_descendants(node, tree) {
//     let descendants = [node.index];
//     if (node['level'] === 0) {
//         return descendants;
//     }
//     while (tree[descendants[0]]['level'] > 0) {
//         let temp = [];
//         for (let i = 0;i < descendants.length;i++) {
//             temp = temp.concat(tree[descendants[i]]['children']);
//         }
//         descendants = temp;
//     }
//     return descendants;
// }

function index_set_to_flag_array(index_set, max_index) {
    let flag_array = new Array(max_index).fill(0);
    for (let i = 0;i < index_set.length;i++) {
        flag_array[index_set[i]] = 1;
    }
    return flag_array;
}

function compute_edges(nodes, all_edges, all_nodes=[]) {
    let edges = [];
    let node_indexs = nodes.map(node => node.index)
    let nodeset = new Map()
    for (let idx of node_indexs) {
        nodeset.set(idx, 0)
        all_nodes[idx].status = 'inner'
    }
    let in_nodes = []
    let out_nodes = []
    let edgeset = new Set()
    let lca = node_indexs.slice(0)
    while (1) {
        let flag = 1
        for (let i = 0; i + 1 < lca.length; ++i) {
            if (lca[i] != lca[i + 1]) {
                flag = 0
                break
            }
        }
        if (flag) break
        for (let i = 0; i < lca.length; ++i) lca[i] = all_nodes[lca[i]].parent
    }
    lca = lca[0]
    const lca_depth = lca == -1 ? 0 : all_nodes[lca].depth

    for (let edge_index = 0; edge_index < all_edges.length; ++edge_index) {
        let edge = all_edges[edge_index]
        let start = edge.start
        let end = edge.end
        while (start != -1 && all_nodes[start].depth > lca_depth && (!nodeset.has(start) || nodeset.get(start) != 0)) {
            start = all_nodes[start].parent
        }
        while (end != -1 && all_nodes[end].depth > lca_depth && (!nodeset.has(end) || nodeset.get(end) != 0)) {
            end = all_nodes[end].parent
        }
        if ((!nodeset.has(start) || nodeset.get(start) != 0) && (!nodeset.has(end) || nodeset.get(end) != 0)) {
            continue
        }
        if (start == lca || end == lca || start == end) continue
        if (edgeset.has(start + ' ' + end)) continue
        edgeset.add(start + ' ' + end)
        if (!nodeset.has(start)) {
            nodeset.set(start, 1)
            in_nodes.push(start)
        }
        if (!nodeset.has(end)) {
            nodeset.set(end, 1)
            out_nodes.push(end)
        }
        let e = { start, end }
        edges.push(e)
    }
    const is_ancestor = (x, y) => {
        while (x != -1 && x != y) x = all_nodes[x].parent
        return x == y
    }
    const shrink = (nodes) => {
        let preserve_nodes = nodes.filter(d => is_ancestor(d, all_nodes[lca].parent))
        let left_nodes = nodes.filter(d => !is_ancestor(d, all_nodes[lca].parent))
        edges.forEach(d => {
            for (let e of left_nodes) {
                if (d.start == e && all_nodes[e].parent != -1) d.start = all_nodes[e].parent
                if (d.end == e && all_nodes[e].parent != -1) d.end = all_nodes[e].parent
            }
        })
        left_nodes = [...new Set(left_nodes.map(d => all_nodes[d].parent).filter(d => d != -1))]
        return preserve_nodes.concat(left_nodes)
    }
    if (in_nodes.length > 1) {
        in_nodes = shrink(in_nodes)
    }
    if (out_nodes.length > 1) {
        out_nodes = shrink(out_nodes)
    }
    
    in_nodes = in_nodes.map(d => all_nodes[d])
    out_nodes = out_nodes.map(d => all_nodes[d])
    in_nodes = in_nodes.concat(out_nodes)
    in_nodes.forEach(d => { d.status = 'outer' })
    nodes = nodes.concat(in_nodes)
    edgeset = new Set()
    let newedges = []
    for (let e of edges) {
        if (edgeset.has(e.start + ' ' + e.end)) continue
        edgeset.add(e.start + ' ' + e.end)
        newedges.push(e)
    }
    edges = newedges
    //console.log(nodes)
    //}
    return [nodes, edges]
}

/*
function compute_edges(nodes, all_edges, delete_edge_index=[], include_outside_edge=false, all_node_indexes=[]) {
    let edges = [];
    let delete_edge_index_map = new Array(all_edges.length).fill(0);
    delete_edge_index.forEach(function (d) {
        delete_edge_index_map[d] = 1;
    });

    if (include_outside_edge) {
        let node_indexs = nodes.map(node=>node.index);

        nodes.forEach(function (node) {
            node.pre.forEach(function (edge_index) {
                let edge = all_edges[edge_index];
                if (!delete_edge_index_map[edge_index] && all_node_indexes.indexOf(edge.start) !== -1) {
                    edges.push(edge);
                }
            });
        });

        nodes.forEach(function (node) {
            node.next.forEach(function (edge_index) {
                let edge = all_edges[edge_index];
                if (!delete_edge_index_map[edge_index] && node_indexs.indexOf(edge.end) === -1 && all_node_indexes.indexOf(edge.end) !== -1) {
                    edges.push(edge);
                }
            });
        });
    }
    else {
        let node_indexs = nodes.map(node => node.index);
        nodes.forEach(function (node) {
            node.next.forEach(function (edge_index) {
                let edge = all_edges[edge_index];
                if (!delete_edge_index_map[edge_index] && node_indexs.indexOf(edge.start) !== -1 && node_indexs.indexOf(edge.end) !== -1) {
                    edges.push(edge);
                }
            });
        });
    }

    return edges;
}
*/
function spline(points) {
    let _spline = d3.line()
        .x(function (d) {
            return d.x;
        })
        .y(function (d) {
            return d.y;
        });
//         .curve(d3.curveCardinal.tension(0.8));

    let newPoints = [];
    for (let i = 0; i < points.length; i++) {
        newPoints.push(points[i]);
    }
    return _spline(newPoints);
}

function cardinal_line(points) {
    let _spline = d3.line()
        .x(function (d) {
            return d.x;
        })
        .y(function (d) {
            return d.y;
        })
        .curve(d3.curveMonotoneX);

    let newPoints = [];
    for (let i = 0; i < points.length; i++) {
        newPoints.push(points[i]);
    }
    return _spline(newPoints);
}

function bundle_line(points) {
    let _spline = d3.line()
        .x(function (d) {
            return d.x;
        })
        .y(function (d) {
            return d.y;
        })
        .curve(d3.curveBundle);

    let newPoints = handle_points(points);
    return _spline(newPoints);
}

function B_spline(points) {
    let step_num = 100;

    let factorials = [];
    for (let i = 0;i < points.length;i++) {
        if (i === 0) {
            factorials.push(1);
        }
        else {
            factorials.push(factorials[factorials.length - 1] * i);
        }
    }

    let _B = function(i, n, t) {
        return factorials[n] / factorials[i] / factorials[n - i] * Math.pow(t, i) * Math.pow(1 - t, n - i);
    };

    let _point = function (t) {
        let point = {
            x: 0,
            y: 0
        };

        for (let i = 0;i < points.length;i++) {
            let _b = _B(i, points.length - 1, t);
            point.x += points[i].x * _b;
            point.y += points[i].y * _b;
        }

        return point;
    };


    let _spline = d3.line()
        .x(function (d) {
            return d.x;
        })
        .y(function (d) {
            return d.y;
        });

    let d_points = [];

    for (let t = 0;t <= step_num;t++) {
        d_points.push(_point(1 - t / step_num));
    }

    return _spline(d_points);
}

function handle_points(points) {
    let new_points = [];
    let size = points.length;
    let start_y = points[0].y, end_y = points[size - 1].y;
    if (start_y <= end_y) {
        points.forEach(point=>{
            if (point.y > end_y) {
                point.y = end_y;
            }
            if (point.y >= start_y) {
                start_y = point.y;
            }
            else {
                point.y = start_y;
            }
            new_points.push(point);
        });
    }
    else {
        points.forEach(point=>{
            if (point.y < end_y) {
                point.y = end_y;
            }
            if (point.y <= start_y) {
                start_y = point.y;
            }
            else {
                point.y = start_y;
            }
            new_points.push(point);
        });
    }
    return new_points;
}

function create_units(time_cost, max_time_cost, width, max_unit_num, start_x) {
    let unit_width = width / max_unit_num;
    let time_width = time_cost / max_time_cost * width;
    let unit_num = Math.ceil(time_width / unit_width);
    let unit = [];
    for (let i = 0;i < unit_num - 1;i++) {
        unit.push({
            'x': start_x + i * unit_width,
            'width': unit_width - 1
        });
    }
    unit.push({
        'x': start_x + (unit_num - 1) * unit_width,
        'width': Math.max(1, unit_width - 1 - (unit_num * unit_width - time_width))
    });
    return unit;
}

function minus_path_d(start_x, start_y, width, height, _k) {
    let k = 1 / _k;
    let x = [start_x, start_x + width];
    let y = [start_y + (1 - k) / 2 * height, start_y + (1 + k) / 2 * height];
    let d = `M${x[0]},${y[0]}`;
    d += `L${x[1]},${y[0]}`;
    d += `L${x[1]},${y[1]}`;
    d += `L${x[0]},${y[1]}`;
    d += `L${x[0]},${y[0]}`;
    return d;
}

function delete_path_d(start_x, start_y, width, height, _k) {
    let k = 1 / _k;
    let x = [start_x, start_x + width];
    let y = [start_y + (1 - k) / 2 * height, start_y + (1 + k) / 2 * height];
    let d = `M${x[0]},${y[0]}`;
    d += `L${x[1]},${y[0]}`;
    d += `L${x[1]},${y[1]}`;
    d += `L${x[0]},${y[1]}`;
    d += `L${x[0]},${y[0]}`;
    return d;
}

function plus_path_d(start_x, start_y, width, height, k) {
    let sum_k = 2 * k + 1;
    let x = [start_x, start_x + k / sum_k * width, start_x + (k + 1) / sum_k * width, start_x + width];
    let y = [start_y, start_y + k / sum_k * height, start_y + (k + 1) / sum_k * height, start_y + height];
    let d = `M${x[0]},${y[1]}`;
    d += `L${x[1]},${y[1]}`;
    d += `L${x[1]},${y[0]}`;
    d += `L${x[2]},${y[0]}`;
    d += `L${x[2]},${y[1]}`;
    d += `L${x[3]},${y[1]}`;
    d += `L${x[3]},${y[2]}`;
    d += `L${x[2]},${y[2]}`;
    d += `L${x[2]},${y[3]}`;
    d += `L${x[1]},${y[3]}`;
    d += `L${x[1]},${y[2]}`;
    d += `L${x[0]},${y[2]}`;
    d += `L${x[0]},${y[1]}`;
    return d;
}

function sgn(x) {
    if (x > 0) {
        return 1;
    }
    else if(x === 0) {
        return 0;
    }
    else {
        return -1;
    }
}

function insert_point_to_line(start_x, start_y, end_x, end_y, num_points) {
    let res = [];
    let x_width = (end_x - start_x) / (num_points + 1);
    let y_width = (end_y - start_y) / (num_points + 1);
    for (let i = 0;i < num_points + 2;i++) {
        res[i] = [start_x + i * x_width, start_y + i * y_width];
    }
    return res
}

function compute_exploring_nodes(all_nodes, all_edges, exploring_nodes, left, right) {
    for (let i = 0;i < exploring_nodes.length;i++) {
        let node = exploring_nodes[exploring_nodes.length - 1 - i];
        let seg_rects = [];
        let exploring_level = 1;
        let descendants = [];
        let max_seg_right_most_index= -1;
        node.children.forEach(child => {
            let child_node = all_nodes[child];
            let left, right, top, bottom;
            if (child_node.exploring) {
                merge_seg_rects(seg_rects, child_node.position_info.seg_rects);
                descendants = descendants.concat(child_node.descendants);
                exploring_level = Math.max(child_node.position_info.exploring_level + 1 , exploring_level);
                max_seg_right_most_index = get_max_seg_right_most_index(max_seg_right_most_index, child_node.max_seg_right_most_index, all_nodes);
            }
            else {
                descendants.push(child);
                max_seg_right_most_index = get_max_seg_right_most_index(max_seg_right_most_index, child, all_nodes);
                while (seg_rects.length - 1 < child_node.seg) {
                    seg_rects.push({
                        'min_x': Infinity,
                        'min_y': Infinity,
                        'max_x': -Infinity,
                        'max_y': -Infinity
                    });
                }
                left = child_node.x - child_node.w / 2;
                right = child_node.x + child_node.w / 2;
                top = child_node.y - child_node.h / 2;
                bottom = child_node.y + child_node.h / 2;
                seg_rects[child_node.seg].min_x = Math.min(seg_rects[child_node.seg].min_x, left);
                seg_rects[child_node.seg].min_y = Math.min(seg_rects[child_node.seg].min_y, top);
                seg_rects[child_node.seg].max_x = Math.max(seg_rects[child_node.seg].max_x, right);
                seg_rects[child_node.seg].max_y = Math.max(seg_rects[child_node.seg].max_y, bottom);
            }
        });

        node.position_info = {
            'seg_rects': seg_rects,
            'exploring_level': exploring_level
        };
        node.descendants = descendants;
        node.max_seg_right_most_index = max_seg_right_most_index;
    }
    exploring_nodes.forEach(node=>{
        let new_seg_rects = [];
        for (let i = 0;i < node.position_info.seg_rects.length;i++) {
            let rect = node.position_info.seg_rects[i];
            if (rect.min_x !== Infinity) {
                rect.position = 'middle';
                new_seg_rects.push(rect);
            }
        }

        if (new_seg_rects.length === 1) {
            new_seg_rects[0].position = 'total';
        }
        else {
            new_seg_rects[0].position = 'left';
            new_seg_rects[new_seg_rects.length - 1].position = 'right';

        }
        for (let i = 0;i < new_seg_rects.length;i++) {
            if (i > 0) {
                new_seg_rects[i].min_x = left;
            }
            if (i < new_seg_rects.length - 1) {
                new_seg_rects[i].max_x = right;
            }
        }
        if (node.position_info.exploring_level > 3) {
            node.position_info.exploring_height = -1;
        }
        else if (node.parent === -1) {
            node.position_info.exploring_height = 0;
        }
        else {
            node.position_info.exploring_height = all_nodes[node.parent].position_info.exploring_height + 1;
        }
        node.position_info.seg_rects = new_seg_rects;
        get_exploring_edges(all_nodes, all_edges, node);
    });
}

function filter_exploring_nodes(exploring_nodes) {
    let new_exploring_nodes = [];
    exploring_nodes.forEach(function (d) {
        if (d.exploring_height >= 0) {
            new_exploring_nodes.push(d);
        }
    });
    return new_exploring_nodes;
}

function merge_seg_rects(rect1, rect2) {
    for (let i = 0;i < rect2.length;i++) {
        let rect = rect2[i];
       if (rect1.length <= i) {
           rect1.push({
               'min_x': rect.min_x,
               'min_y': rect.min_y,
               'max_x': rect.max_x,
               'max_y': rect.max_y
           });
       }
       else {
           rect1[i].min_x = Math.min(rect.min_x, rect1[i].min_x);
           rect1[i].min_y = Math.min(rect.min_y, rect1[i].min_y);
           rect1[i].max_x = Math.max(rect.max_x, rect1[i].max_x);
           rect1[i].max_y = Math.max(rect.max_y, rect1[i].max_y);
       }
    }
}

function get_childs_and_exploring_childs(root, tree) {
    if (!root.exploring) {
        return {
            'nodes': [root.index],
            'exploring_nodes': []
        }
    }
    let childs = [], exploring_childs = [root.index];
    let children = root.children;
    children.forEach(function (d) {
        let child_node = tree[d];
        if (child_node.exploring) {
            let {nodes, exploring_nodes} = get_childs_and_exploring_childs(child_node, tree);
            childs = childs.concat(nodes);
            exploring_childs = exploring_childs.concat(exploring_nodes);
        }
        else {
            childs.push(d);
        }
    });
    return {
            'nodes': childs,
            'exploring_nodes': exploring_childs
        };
}

function path_d_exploring_node_background(seg_rect) {
    let path_d = `M${seg_rect._min_x},${seg_rect._min_y}`;
    path_d += `L${seg_rect._max_x},${seg_rect._min_y}`;
    if (seg_rect.position === 'right' || seg_rect.position === 'total') {
       path_d += 'L';
    }
    else {
        path_d += 'M';
    }
    path_d += `${seg_rect._max_x},${seg_rect._max_y}`;
    path_d += `L${seg_rect._min_x},${seg_rect._max_y}`;
    if (seg_rect.position === 'left' || seg_rect.position === 'total') {
       path_d += 'L';
    }
    else {
        path_d += 'M';
    }
    path_d += `${seg_rect._min_x},${seg_rect._min_y}`;
    return path_d;
}

function init_interactions() {
    d3.selectAll('#checkpoints_menu_ul').style('display', 'none')
    document.getElementById("PanelRow-1").style.display = "flex"
    window.show_topbar = 1
    d3.selectAll('#hidden_btn').classed('active', false);
    d3.selectAll('#statistics_btn').classed('active', true);
    d3.selectAll('#execute_op_info_btn').classed('active', false);
    d3.selectAll('#value_menu_ul').style('display', 'flex');
    d3.selectAll('#checkpoints_menu_ul').style('display', 'none');
    d3.selectAll('#checkpoints_add_btn').classed('active', false);
    d3.selectAll('#checkpoints_del_btn').classed('active', false);
    $('#hidden_btn').on('click', function () {
        document.getElementById("PanelRow-1").style.display = "none"
        window.show_topbar = 0
        d3.selectAll('#hidden_btn').classed('active', true);
        d3.selectAll('#statistics_btn').classed('active', false);
        d3.selectAll('#execute_op_info_btn').classed('active', false);
        d3.selectAll('#value_menu_ul').style('display', 'none');
        statistic_component.hidden_all()
    });
    $('#statistics_btn').on('click', function () {
        document.getElementById("PanelRow-1").style.display = "flex"
        window.show_topbar = 1
        d3.selectAll('#hidden_btn').classed('active', false);
        d3.selectAll('#statistics_btn').classed('active', true);
        d3.selectAll('#execute_op_info_btn').classed('active', false);
        d3.selectAll('#value_menu_ul').style('display', 'flex');
        d3.selectAll('#checkpoints_menu_ul').style('display', 'none');
        d3.selectAll('#checkpoints_add_btn').classed('active', false);
        d3.selectAll('#checkpoints_del_btn').classed('active', false);
        statistic_component.update_checkpoints_update_state('none');
        statistic_component.switch_between_statistics_and_op_schedule('statistics');
    });
    $('#execute_op_info_btn').on('click', function () {
        document.getElementById("PanelRow-1").style.display = "flex"
        window.show_topbar = 1
        d3.selectAll('#hidden_btn').classed('active', false);
        d3.selectAll('#statistics_btn').classed('active', false);
        d3.selectAll('#execute_op_info_btn').classed('active', true);
        d3.selectAll('#value_menu_ul').style('display', 'none');
        d3.selectAll('#checkpoints_menu_ul').style('display', 'flex');
        statistic_component.switch_between_statistics_and_op_schedule('op_schedule');
    });
    $('#checkpoints_add_btn').on('click', function () {
        if (d3.selectAll('#checkpoints_add_btn').classed('active')) {
            d3.selectAll('#checkpoints_add_btn').classed('active', false);
            statistic_component.update_checkpoints_update_state('none');
        }
        else {
            d3.selectAll('#checkpoints_add_btn').classed('active', true);
            d3.selectAll('#checkpoints_del_btn').classed('active', false);
            statistic_component.update_checkpoints_update_state('add');
        }
    });
    $('#checkpoints_del_btn').on('click', function () {
        if (d3.selectAll('#checkpoints_del_btn').classed('active')) {
            d3.selectAll('#checkpoints_del_btn').classed('active', false);
            statistic_component.update_checkpoints_update_state('none');
        }
        else {
            d3.selectAll('#checkpoints_add_btn').classed('active', false);
            d3.selectAll('#checkpoints_del_btn').classed('active', true);
            statistic_component.update_checkpoints_update_state('delete');
        }
    });

    $('#info-remove-btn').on('click', function () {
        statistic_component.hide_detail_info();
        d3.selectAll('#info-tip-div').transition().duration(self.duration).style('opacity', 0)
            .style('width', `0px`)
            .style('top', `-100px`)
            .style('left', `-100px`);
        d3.selectAll('#info-tip-content')
            .transition().duration(500)
            .style('width', `0px`);
        network_component.hide_detail_info();
    });

    init_drag_for_info_tip_div();
}

function init_drag_for_info_tip_div() {
    let dv = document.getElementById('info-tip-title');
    let tdv = document.getElementById('info-tip-div');
    let dv1 = document.getElementById('info-tip-title-text');
    let x = 0;
    let y = 0;
    let l = 0;
    let t = 0;
    let isDown = false;
    //鼠标按下事件
    dv.onmousedown = function(e) {
        //获取x坐标和y坐标
        x = e.clientX;
        y = e.clientY;

        //获取左部和顶部的偏移量
        l = tdv.offsetLeft;
        t = tdv.offsetTop;
        //开关打开
        isDown = true;
        //设置样式
        dv.style.cursor = 'grabing';
        dv1.style.cursor = 'grabing';
    };
    //鼠标移动
    window.onmousemove = function(e) {
        if (isDown == false) {
            return;
        }
        //获取x和y
        var nx = e.clientX;
        var ny = e.clientY;
        //计算移动后的左偏移量和顶部的偏移量
        var nl = nx - (x - l);
        var nt = ny - (y - t);

        tdv.style.left = nl + 'px';
        tdv.style.top = nt + 'px';
    };
    //鼠标抬起事件
    window.onmouseup = function() {
        //开关关闭
        isDown = false;
        dv.style.cursor = 'grab';
        dv1.style.cursor = 'grab';
    };
    d3.selectAll('#info-tip-div')
        .style('opacity', 0);
}

function init_single_group_click_menu(op_index, show_dependency=true) {
    let group_menu = [{
            name:'detail info',
            title:'detail info',
            fun:function(){
                statistic_component.show_detail_info();
            }
        }, {
            name: show_dependency? 'show dependency': 'hide dependency',
            title: show_dependency? 'show dependency': 'hide dependency',
            fun:function(){
                statistic_component.show_dependency(show_dependency);
            }
        }];
    $(`#op_group_g_${op_index}`).contextMenu(group_menu, {'mouseClick':'right','triggerOn':'click'});
}

function init_group_click_menu() {
    let group_menu_show = [{
            name:'detail info',
            title:'detail info',
            fun:function(){
                statistic_component.show_detail_info();
            }
        }, {
            name:'show dependency',
            title:'show dependency',
            fun:function(){
                statistic_component.show_dependency();
            }
        }];
    let group_menu_hide = [{
            name:'detail info',
            title:'detail info',
            fun:function(){
                statistic_component.show_detail_info();
            }
        }, {
            name:'hide dependency',
            title:'hide dependency',
            fun:function(){
                statistic_component.show_dependency(false);
            }
        }];
    d3.selectAll('.iw-contextMenu').remove();
    $('.op_group_g_show').contextMenu(group_menu_show, {'mouseClick':'right','triggerOn':'click'});
    $('.op_group_g_hide').contextMenu(group_menu_hide, {'mouseClick':'right','triggerOn':'click'});
}

function init_network_click_menu() {
    let nextwork_menu = [{
            name:'detail info',
            title:'detail info',
            fun:function(){
                network_component.show_detail_info();
            }
        }];
    $('.node_main_rect_g').contextMenu(nextwork_menu, {'mouseClick':'right','triggerOn':'click'});
    $('.var_node_group').contextMenu(nextwork_menu, {'mouseClick':'right','triggerOn':'click'});
}

function set_content_of_tooltip(attrs) {
    d3.selectAll('#info-tip-div')
        .style('width', `${attrs.width}px`)
        .style('left', `${attrs.left}px`)
        .style('top', `${attrs.top}px`);
    d3.selectAll('#info-tip-content')
        .style('width', `${attrs.width}px`);
    d3.selectAll('#info-tip-title-text')
        .style('width', `${attrs.width - 24}px`);
    d3.selectAll('#info-tip-div')
        .transition().duration(500)
        .style('opacity', attrs.opacity);
    d3.selectAll('#info-tip-content')
        .transition().duration(500);
    $('#info-tip-title-text').text(attrs['title']);
    let values = attrs.values;
    let innerhtml = '';
    for (let key in values) {
        let value = values[key];
        // innerhtml += `${key}:<p style=\"margin-left: 20px;\">${value}</p>`;
        innerhtml += `<p>${key}: ${value}</p>`;
    }
    $('#info-tip-content')[0].innerHTML = innerhtml;
}

function short_title(title, max_size) {
    let short_dic = {
        'array': 'arr',
        'binary': 'bi',
        'subtract': 'sub',
        'divide': 'div',
        'broadcast': 'bcast'
    };
    let short_title = title;
    for (let key in short_dic) {
        short_title = short_title.replace(key, short_dic[key]);
    }
    if (short_title.length > max_size) {
        short_title = short_title.slice(0, max_size - 2) + '··';
    }
    // console.log(title, 'to', short_title);
    return short_title;
}

function short_string(str, max_size) {
    if (str.length > max_size) {
        str = str.slice(0, max_size - 2) + '··';
    }
    return str;
}

function seg_node_path_d(node) {
    let res = 'M0,0L10,0L10,10L0,10L0,0';
    let max_r = 8, min_r = 4;
    if (node.type === 'start') {
        res = `M${-min_r / Math.sqrt(2)},${min_r / Math.sqrt(2)}A${min_r},${min_r},0,1,0,${-min_r / Math.sqrt(2)},${-min_r / Math.sqrt(2)}`;
        res += `L${-max_r / Math.sqrt(2)},${-max_r / Math.sqrt(2)}A${max_r},${max_r},0,1,1,${-max_r / Math.sqrt(2)},${max_r / Math.sqrt(2)}`;
    }
    else if (node.type === 'end') {
        res = `M${-min_r},${0}A${min_r},${min_r},0,1,0,${min_r},${0}`;
        res += `M${-min_r},${0}A${min_r},${min_r},0,1,1,${min_r},${0}`;
    }
    return res;
}

function square_distance(point1, point2) {
    return (point1.x - point2.x) * (point1.x - point2.x) + (point1.y - point2.y) * (point1.y - point2.y);
}

function get_exploring_edges(all_nodes, all_edges, exploring_node) {
    let descendants = exploring_node.descendants;
    let exploring_edges = [], exploring_seg_edges = [], exploring_seg_nodes = [];
    descendants.forEach(node_index=> {
        let node = all_nodes[node_index];
        node.pre.forEach(edge_index=>{
            let edge = all_edges[edge_index];
            if (descendants.indexOf(parseInt(edge.start)) !== -1) {
                exploring_edges.push(edge_index);
                exploring_seg_nodes = exploring_seg_nodes.concat(edge.seg_nodes);
                exploring_seg_edges = exploring_seg_edges.concat(edge.seg_edges);
            }
        });
    });
    exploring_node.exploring_edges = exploring_edges;
    exploring_node.exploring_seg_edges = exploring_seg_edges;
    exploring_node.exploring_seg_nodes = exploring_seg_nodes;

    // let river_edges = [];
    // let index_to_cluster = {};
    // let cluster_to_indexes = [];
    //
    // descendants.forEach(node_index=> {
    //     if (index_to_cluster[node_index] === undefined) {
    //         index_to_cluster[node_index] = cluster_to_indexes.length;
    //         cluster_to_indexes.push([node_index]);
    //         let node_queue = [node_index];
    //         let queue_index = 0;
    //         while (queue_index < node_queue.length) {
    //             let node = all_nodes[node_queue[queue_index]];
    //             node.pre.forEach(edge_index=>{
    //                 let edge = all_edges[edge_index];
    //                 if (descendants.indexOf(parseInt(edge.start)) !== -1 && index_to_cluster[parseInt(edge.start)] === undefined) {
    //                     node_queue.push(parseInt(edge.start));
    //                     index_to_cluster[parseInt(edge.start)] = index_to_cluster[node_queue[queue_index]];
    //                     cluster_to_indexes[index_to_cluster[node_queue[queue_index]]].push(parseInt(edge.start));
    //                 }
    //             });
    //             node.next.forEach(edge_index=>{
    //                 let edge = all_edges[edge_index];
    //                 if (descendants.indexOf(parseInt(edge.end)) !== -1 && index_to_cluster[parseInt(edge.end)] === undefined) {
    //                     node_queue.push(parseInt(edge.end));
    //                     index_to_cluster[parseInt(edge.end)] = index_to_cluster[node_queue[queue_index]];
    //                     cluster_to_indexes[index_to_cluster[node_queue[queue_index]]].push(parseInt(edge.end));
    //                 }
    //             });
    //             queue_index++;
    //         }
    //     }
    // });
    //
    // let distance_matrix = new Array(cluster_to_indexes.length);
    // for (let i = 0;i < cluster_to_indexes.length;i++) {
    //     distance_matrix[i] = new Array(cluster_to_indexes.length);
    //     for (let j = 0;j < cluster_to_indexes.length;j++) {
    //         distance_matrix[i][j] = {
    //             start: -1,
    //             end: -1,
    //             square_distance: Infinity
    //         };
    //         if (i === j) {
    //             distance_matrix[i][j].square_distance = 0;
    //         }
    //     }
    // }
    //
    // let edge_queue = [];
    // for (let cluster_index1 = 0;cluster_index1 < cluster_to_indexes.length;cluster_index1++) {
    //     let indexes1 = cluster_to_indexes[cluster_index1];
    //     for (let cluster_index2 = cluster_index1 + 1;cluster_index2 < cluster_to_indexes.length;cluster_index2++) {
    //         let indexes2 = cluster_to_indexes[cluster_index2];
    //         indexes1.forEach(index1=>{
    //             indexes2.forEach(index2=>{
    //                 let distance = square_distance(all_nodes[index1], all_nodes[index2]);
    //                 if (distance < distance_matrix[cluster_index1][cluster_index2].square_distance) {
    //                     distance_matrix[cluster_index1][cluster_index2].square_distance = distance;
    //                     distance_matrix[cluster_index1][cluster_index2].start = index1;
    //                     distance_matrix[cluster_index1][cluster_index2].end = index2;
    //                     distance_matrix[cluster_index2][cluster_index1].square_distance = distance;
    //                     distance_matrix[cluster_index2][cluster_index1].start = index1;
    //                     distance_matrix[cluster_index2][cluster_index1].end = index2;
    //                 }
    //             });
    //         });
    //         edge_queue.push([cluster_index1, cluster_index2]);
    //     }
    // }
    //
    // edge_queue.sort(function (a, b) {
    //     return distance_matrix[a[0]][a[1]].square_distance - distance_matrix[b[0]][b[1]].square_distance;
    // });
    //
    // let cluster_merge_func = [];
    // for (let i = 0;i < cluster_to_indexes.length;i++) {
    //     cluster_merge_func[i] = i;
    // }
    //
    //
    // for (let i = 0;i < edge_queue.length;i++) {
    //     let cluster_index0 = edge_queue[i][0], cluster_index1 = edge_queue[i][1];
    //     if (cluster_merge_func[cluster_index0] !== cluster_merge_func[cluster_index1]) {
    //         river_edges.push(distance_matrix[cluster_index0][cluster_index1]);
    //         for (let i = 0;i < cluster_to_indexes.length;i++) {
    //             if (cluster_merge_func[i] === cluster_merge_func[cluster_index1]) {
    //                 cluster_merge_func[i] = cluster_merge_func[cluster_index0];
    //             }
    //         }
    //         if (river_edges.length === cluster_to_indexes.length - 1) {
    //             break;
    //         }
    //     }
    // }
    //
    // exploring_node.river_edges = river_edges;
}

function get_max_seg_right_most_index(curr_index, index, all_nodes) {
    if (curr_index === -1) {
        return index;
    }
    let curr_node = all_nodes[curr_index], node = all_nodes[index];
    let res = curr_index;
    if (node.seg > curr_node.seg) {
        res = index;
    }
    else if (node.seg === curr_node.seg) {
        if (node.x + node.w / 2 > curr_node.x + curr_node.w / 2) {
            res = index;
        }
        else if (node.x + node.w / 2 === curr_node.x + curr_node.w / 2
            && node.y - node.h / 2 < curr_node.y - curr_node.h / 2) {
            res = index;
        }
    }
    return res;
}

function get_intersection_of_line_and_rect(node, node1, node2, time_cost_height=0) {
    let margin = 0;
    let x1 = node1.x, x2 = node2.x, y1 = node1.y, y2 = node2.y;
    // line (y2 - y1)x + (x1 - x2)y = x1y2 - x2y1
    //        y = (x1y2 - x2y1 - (y2 - y1)x) / (x1 - x2)
    //        x = (x1y2 - x2y1 - (x1 - x2)y) / (y2 - y1)
    // node rect: (node.x - node.w / 2, node.y - node.h / 2)                              (node.x + node.w / 2, node.y - node.h / 2)
    //            (node.x - node.w / 2, node.y + node.h / 2)                              (node.x + node.w / 2, node.y + node.h / 2)
    // node top line: y = node.y - node.h / 2
    // node bottom line: y = node.y + node.h / 2
    // node left line: x = node.x - node.w / 2
    // node right line: x = node.x + node.w / 2
    let w = node.w + 2 * margin, h = node.h + 2 * margin;
    let top = node.y - h / 2,
        bottom = node.y + h / 2,
        left = node.x - w / 2,
        right = node.x + w / 2;
    if (!node.is_var) {
        bottom -= time_cost_height;
    }
    let inter_node = [];
     // node top line intersection
    if (y1 !== y2) {
        let top_x = (x1 * y2 - x2 * y1 - (x1 - x2) * top) / (y2 - y1);
        if (top_x <= right && top_x >= left) {
            inter_node.push({
                x: top_x,
                y: top
            });
        }
        let bottom_x = (x1 * y2 - x2 * y1 - (x1 - x2) * bottom) / (y2 - y1);
        if (bottom_x <= right && bottom_x >= left) {
            inter_node.push({
                x: bottom_x,
                y: bottom
            });
        }
    }
    if (x1 !== x2) {
        let left_y = (x1 * y2 - x2 * y1 - (y2 - y1) * left) / (x1 - x2);
        if (left_y <= bottom && left_y >= top) {
            inter_node.push({
                x: left,
                y: left_y
            });
        }
        let right_y = (x1 * y2 - x2 * y1 - (y2 - y1) * right) / (x1 - x2);
        if (right_y <= bottom && right_y >= top) {
            inter_node.push({
                x: right,
                y: right_y
            });
        }
    }
    let res_node = undefined;
    if (inter_node.length === 1) {
        res_node = inter_node[0];
    }
    else if (inter_node.length === 2) {
        let mean_point = {
            x: (x1 + x2) / 2,
            y: (y1 + y2) / 2
        };
        let distance = inter_node.map(point=>square_distance(point, mean_point));
        if (distance[0] >= distance[1]) {
            res_node = inter_node[1];
        }
        else {
            res_node = inter_node[0];
        }
    }
    else {
        console.log('too much intersection');
    }
    if (res_node === undefined) {
        return undefined;
    }
    else {
        let min_x = Math.min(node1.x, node2.x) - 0.001,
            min_y = Math.min(node1.y, node2.y) - 0.001,
            max_x = Math.max(node1.x, node2.x) + 0.001,
            max_y = Math.max(node1.y, node2.y) + 0.001;
        if (res_node.x >= min_x && res_node.x <= max_x && res_node.y >= min_y && res_node.y <= max_y) {
            return [res_node, 'replace'];
        }
        else {
            return [res_node, 'push'];
        }
    }
}

function one_edge(points) {
    // return bundle_line(points);
    // return line(points);
    // return B_spline(points);

//     points = handle_points(points);
    let len = points.length;
    if (len === 0) { return "" }
    let start = `M ${points[0].x} ${points[0].y}`,
        vias = [];

    const getInter = (p1, p2, n) => {
        return `${p1.x * n + p2.x * (1 - n)} ${p1.y * n + p2.y * (1 - n)}`
    };

    const getCurve = (points) => {
        let vias = [],
            len = points.length;
        const ratio = 0.5;
        for (let i = 0; i < len - 2; i++) {
            let p1, p2, p3, p4, p5;
            if (i === 0) {
                p1 = `${points[i].x} ${points[i].y}`
            } else {
                p1 = getInter(points[i], points[i + 1], ratio)
            }
            p2 = getInter(points[i], points[i + 1], 1 - ratio);
            p3 = `${points[i + 1].x} ${points[i + 1].y}`;
            p4 = getInter(points[i + 1], points[i + 2], ratio);
            if (i === len - 3) {
                p5 = `${points[i + 2].x} ${points[i + 2].y}`
            } else {
                p5 = getInter(points[i + 1], points[i + 2], 1 - ratio)
            }
            let cPath = `M ${p1} L${p2} Q${p3} ${p4} L${p5}`;
            vias.push(cPath);
        }
        return vias
    };
    vias = getCurve(points);
    let pathData = `${start}  ${vias.join(' ')}`;
    return pathData;
}

function adjust_points(edge, threshold, left, right) {
    if (edge.start === 6752 && edge.end === 6770) {
        console.log('xxxx');
    }
    if (left === 0) {
        adjust_points(edge, threshold, right, right + 1);
    }
    else if (right === edge.points.length) {
        return;
    }
    if (right - left < threshold) {
        if (edge.points[right].y === edge.points[right - 1].y) {
            adjust_points(edge, threshold, left, right + 1);
        }
        else if (edge.points[left - 1].y < edge.points[left].y && edge.points[right].y < edge.points[right - 1].y) {
            if (edge.points[left - 1].y < edge.points[right].y) {
                for (let k = left;k < right;k++) {
                    edge.points[k].y = edge.points[right].y;
                }
                adjust_points(edge, threshold, left, right + 1);
            }
            else {
                for (let k = left;k < right;k++) {
                    edge.points[k].y = edge.points[left - 1].y;
                }
                adjust_points(edge, threshold, left - 1, right);
            }
        }
        else if (edge.points[left - 1].y > edge.points[left].y && edge.points[right].y > edge.points[right - 1].y) {
            if (edge.points[left - 1].y > edge.points[right].y) {
                for (let k = left;k < right;k++) {
                    edge.points[k].y = edge.points[right].y;
                }
                adjust_points(edge, threshold, left, right + 1);
            }
            else {
                for (let k = left;k < right;k++) {
                    edge.points[k].y = edge.points[left - 1].y;
                }
                adjust_points(edge, threshold, left - 1, right);
            }
        }
        else {
            adjust_points(edge, threshold, right, right + 1);
        }
    }
    else if (edge.points[right].y === edge.points[right - 1].y) {
        adjust_points(edge, threshold, left, right + 1);
    }
    else if ((edge.points[left - 1].y - edge.points[left].y) * (edge.points[right].y - edge.points[right - 1].y) < 0) {
        if (Math.abs(edge.points[left - 1].y - edge.points[left].y) < Math.abs(edge.points[right].y - edge.points[right - 1].y)) {
            for (let k = left;k < right;k++) {
                edge.points[k].y = edge.points[left - 1].y;
            }
            adjust_points(edge, threshold, left - 1, right);
        }
        else {
            for (let k = left;k < right;k++) {
                edge.points[k].y = edge.points[right].y;
            }
            adjust_points(edge, threshold, left, right + 1);
        }
    }
    else if (edge.points[left - 1].y < edge.points[left].y && edge.points[right].y < edge.points[right - 1].y) {
        if (Math.abs(edge.points[left].y - Math.max(edge.points[left - 1].y, edge.points[right].y)) <= 5) {
            if (edge.points[left - 1].y < edge.points[right].y) {
                for (let k = left;k < right;k++) {
                    edge.points[k].y = edge.points[right].y;
                }
                adjust_points(edge, threshold, left, right + 1);
            }
            else {
                for (let k = left;k < right;k++) {
                    edge.points[k].y = edge.points[left - 1].y;
                }
                adjust_points(edge, threshold, left - 1, right);
            }
        }
        else {
            adjust_points(edge, threshold, right, right + 1);
        }
    }
    else if (edge.points[left - 1].y > edge.points[left].y && edge.points[right].y > edge.points[right - 1].y) {
        if (Math.abs(edge.points[left].y - Math.min(edge.points[left - 1].y, edge.points[right].y)) <= 5) {
            if (edge.points[left - 1].y > edge.points[right].y) {
                for (let k = left;k < right;k++) {
                    edge.points[k].y = edge.points[right].y;
                }
                adjust_points(edge, threshold, left, right + 1);
            }
            else {
                for (let k = left;k < right;k++) {
                    edge.points[k].y = edge.points[left - 1].y;
                }
                adjust_points(edge, threshold, left - 1, right);
            }
        }
        else {
            adjust_points(edge, threshold, right, right + 1);
        }
    }
    else {
        adjust_points(edge, threshold, right, right + 1);
    }
}

function get_exploring_height_of_node(d, all_nodes) {
    if (d.parent === -1) {
        d.exploring_height = -1;
    }
    else {
        let exploring_height = all_nodes[d.parent].exploring_height;
        d.exploring_height = -1;
        if (exploring_height < 3) {
            d.exploring_height = exploring_height;
        }
    }
}

function get_exploring_height_of_edge(edge) {
    let start_exploring_height = edge.startNode.exploring_height,
        end_exploring_height = edge.endNode.exploring_height;

    if (start_exploring_height === end_exploring_height) {
        edge.exploring_height = start_exploring_height;
    }
    else {
        edge.exploring_height = -2;
    }
}

function op_schedule_bar_icon_path_d(left, right, top, bottom, direction) {
    let res = '';
    let center_x = (left + right) / 2, height = bottom - top;
    let x_y_scale = 1;
    let triangle_width = height * 0.75 * x_y_scale;
    if (right - left > 3 * triangle_width) {
        res += triangle(center_x - triangle_width * 1.5, center_x - triangle_width * 0.5, top + height / 8, bottom - height / 8, direction);
        res += triangle(center_x - triangle_width * 0.5, center_x + triangle_width * 0.5, top + height / 8, bottom - height / 8, direction);
        res += triangle(center_x + triangle_width * 0.5, center_x + triangle_width * 1.5, top + height / 8, bottom - height / 8, direction);
    }
    else if (right - left > 2 * triangle_width) {
        res += triangle(center_x - triangle_width, center_x, top + height / 8, bottom - height / 8, direction);
        res += triangle(center_x, center_x + triangle_width, top + height / 8, bottom - height / 8, direction);
    }
    else if (right - left > 1.5 * triangle_width) {
        let delta_x_scale = d3.scaleLinear()
                        .domain([1.5 * triangle_width, 2 * triangle_width])
                        .range([triangle_width / 2, triangle_width]);
        let delta_x = delta_x_scale(right - left);
        res += triangle(center_x - delta_x / 2 - triangle_width / 2, center_x - delta_x / 2 + triangle_width / 2, top + height / 8, bottom - height / 8, direction);
        res += triangle(center_x + delta_x / 2 - triangle_width / 2, center_x + delta_x / 2 + triangle_width / 2, top + height / 8, bottom - height / 8, direction);
    }
    else {
        if (right - left < triangle_width) {
            triangle_width = right - left;
        }
        res += triangle(center_x - triangle_width / 2, center_x + triangle_width / 2, top + height / 8, bottom - height / 8, direction);
    }
    return res;
}

function triangle(left, right, top, bottom, direction) {
    let res = '';
    if (direction === 'left') {
        res += `M${right}, ${top}`;
        res += `L${left}, ${(top + bottom) / 2}`;
        res += `L${right}, ${bottom}`;
        res += `L${right}, ${top}`;
    }
    else if (direction === 'right') {
        res += `M${left}, ${top}`;
        res += `L${right}, ${(top + bottom) / 2}`;
        res += `L${left}, ${bottom}`;
        res += `L${left}, ${top}`;
    }
    return res;
}

function get_exploring_node_grid_hull(all_nodes, all_edges, exploring_nodes, nodes, time_cost_height, update_layout) {
    let formData = new FormData();
    let node_keys = ['index', 'x', 'y', 'w', 'h', 'parent', 'children', 'id', 'pre', 'next', 'exploring', 'is_var', 'expand'];
    let new_all_nodes = all_nodes.map(node=>{
        let new_node = {};
        node_keys.forEach(key=>{
            new_node[key] = node[key];
        });
        new_node['node_top'] = new_node['y'] - new_node['h'] / 2;
        new_node['node_bottom'] = new_node['y'] + new_node['h'] / 2;
        new_node['node_left'] = new_node['x'] - new_node['w'] / 2;
        new_node['node_right'] = new_node['x'] + new_node['w'] / 2;
        if (!new_node['is_var'] && !new_node['expand']) {
            new_node['node_bottom'] -= time_cost_height;
        }
        return new_node;
    });
    let new_nodes = nodes.map(node=>{
        let new_node = {};
        node_keys.forEach(key=>{
            new_node[key] = node[key];
        });
        new_node['node_top'] = new_node['y'] - new_node['h'] / 2;
        new_node['node_bottom'] = new_node['y'] + new_node['h'] / 2;
        new_node['node_left'] = new_node['x'] - new_node['w'] / 2;
        new_node['node_right'] = new_node['x'] + new_node['w'] / 2;
        if (!new_node['is_var'] && !new_node['expand']) {
            new_node['node_bottom'] -= time_cost_height;
        }
        return new_node;
    });
    let edge_keys = ['index', 'start', 'start_index', 'start_stack', 'end', 'end_index', 'end_stack'];
    let new_all_edges = all_edges.map(edge=>{
        let new_edge = {};
        edge_keys.forEach(key=>{
            new_edge[key] = edge[key];
        });
        return new_edge;
    });
    let exploring_node_keys = ['index', 'x', 'y', 'w', 'h', 'parent', 'children', 'id', 'pre', 'next', 'exploring', 'is_var'];
    let new_exploring_nodes = exploring_nodes.map(node=>{
        let new_node = {};
        exploring_node_keys.forEach(key=>{
            new_node[key] = node[key];
        });
        return new_node;
    });
    formData.append('all_nodes', JSON.stringify(new_all_nodes));
    formData.append('nodes', JSON.stringify(new_nodes));
    formData.append('all_edges', JSON.stringify(new_all_edges));
    formData.append('exploring_nodes', JSON.stringify(new_exploring_nodes));
    formData.append('update_layout', JSON.stringify(update_layout));

    let xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/get_exploring_node_hull', true);

    xhr.onload = function (e) {
        if (xhr.status === 200) {
            let response = JSON.parse(xhr.response);
            let hulls = response['hulls'];
            if (hulls.length > 0) {
                hulls.forEach(hull=>{
                    all_nodes[hull.node_index].hull_points = hull.hull_points;
                    all_nodes[hull.node_index].exploring_height = hull.exploring_height;
                    all_nodes[hull.node_index].exploring_level = hull.exploring_level;
                    all_nodes[hull.node_index].del_btn_pos = hull.del_btn_pos;
                });
                if (response['update_layout']) {
                    let updated_nodes = response['nodes'];
                    updated_nodes.forEach(node=>{
                        all_nodes[node.node_index].y = node.y;
                    });
                }

            }
            network_component.repaint();

        } else {
            alert('An error occurred!');
        }
    };
    xhr.send(formData);
}

function get_exploring_node_level_and_height(all_nodes, exploring_nodes, time_cost_height) {
    let formData = new FormData();
    let node_keys = ['index', 'x', 'y', 'w', 'h', 'parent', 'children', 'id', 'pre', 'next', 'exploring', 'is_var', 'expand'];
    let new_all_nodes = all_nodes.map(node=>{
        let new_node = {};
        node_keys.forEach(key=>{
            new_node[key] = node[key];
        });
        new_node['node_top'] = new_node['y'] - new_node['h'] / 2;
        new_node['node_bottom'] = new_node['y'] + new_node['h'] / 2;
        new_node['node_left'] = new_node['x'] - new_node['w'] / 2;
        new_node['node_right'] = new_node['x'] + new_node['w'] / 2;
        if (!new_node['is_var'] && !new_node['expand']) {
            new_node['node_bottom'] -= time_cost_height;
        }
        return new_node;
    });
    let exploring_node_keys = ['index', 'x', 'y', 'w', 'h', 'parent', 'children', 'id', 'pre', 'next', 'exploring', 'is_var'];
    let new_exploring_nodes = exploring_nodes.map(node=>{
        let new_node = {};
        exploring_node_keys.forEach(key=>{
            new_node[key] = node[key];
        });
        return new_node;
    });
    formData.append('all_nodes', JSON.stringify(new_all_nodes));
    formData.append('exploring_nodes', JSON.stringify(new_exploring_nodes));

    let xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/get_exploring_node_level_and_height', true);

    xhr.onload = function (e) {
        if (xhr.status === 200) {
            let response = JSON.parse(xhr.response);
            let exploring_nodes = response['exploring_nodes'];
            if (exploring_nodes.length > 0) {
                exploring_nodes.forEach(exploring_node=>{
                    all_nodes[exploring_node.index].exploring_height = exploring_node.exploring_height;
                    all_nodes[exploring_node.index].exploring_level = exploring_node.exploring_level;
                });
            }
            network_component.repaint();

        } else {
            alert('An error occurred!');
        }
    };
    xhr.send(formData);
}

function id_manager() {
    return {
        used_id: [],
        get_id: function () {
            for (let i = 0;i < this.used_id.length;i++) {
                if (this.used_id[i] === 0) {
                    this.used_id[i] = 1;
                    return i;
                }
            }
            this.used_id.push(1);
            return this.used_id.length - 1;
        },
        del_id: function (id) {
            this.used_id[id] = 0;
            return;
        }
    }
}

function highlight_network_dots(op_group, label, state) {
    network_component.dot_highlight(op_group.nodes, label, state);
}

function click_manager(duration) {
    return {
        _clickable: true,
        duration: duration,
        tasks: [],
        try_click: function (func) {
            let self = this;
            if (self._clickable) {
                func();
                self._clickable = false;
                setTimeout(function () {
                    self._clickable = true;
                    if (self.tasks.length > 0) {
                        let task = self.tasks.pop();
                        self.try_click(task);
                    }
                }, self.duration);
            }
            else {
                self.tasks.unshift(func);
            }
        }
    }
}

function my_throttle(duration) {
    return {
        _available: true,
        duration: duration,
        func: undefined,
        run: function (func) {
            let self = this;
            if (self._available) {
                func();
                self.func = undefined;
                self._available = false;
                setTimeout(function () {
                    self._available = true;
                    if (self.func !== undefined) {
                        self.run(self.func);
                    }
                }, self.duration);
            }
            else {
                self.func = func;
            }
        }
    }
}

function process_duplicated_brother_var_nodes(nodes, all_edges) {
    let res = [];
    let var_groups = {};
    let node_index = {};
    nodes.forEach(node=>{
        node_index[node.index] = true;
    });
    nodes.forEach(node=>{
        if (node.is_var) {
            let pre_edge_ids = node.pre, next_edge_ids = node.next;
            let pre_edges = pre_edge_ids.map(x=>all_edges[x]),
                next_edges = next_edge_ids.map(x=>all_edges[x]);

            let pre_node_ids = [],
                next_node_ids = [];
            pre_edges.forEach(edge=>{
                if (node_index[edge.start] !== undefined) {
                    pre_node_ids.push(edge.start);
                }
            });
            next_edges.forEach(edge=>{
                if (node_index[edge.end] !== undefined) {
                    next_node_ids.push(edge.end);
                }
            });
            pre_node_ids.sort();
            next_node_ids.sort();
            let var_key = pre_node_ids.join('$') + '->' + next_node_ids.join('$');
            if (var_groups[var_key] === undefined) {
                var_groups[var_key] = [];
            }
            node.children = [];
            var_groups[var_key].push(node);
        }
        else {
            res.push(node);
        }
    });

    for (let var_key in var_groups) {
        let var_nodes = var_groups[var_key];
        if (var_nodes.length > 1) {
            var_nodes[0].children = var_nodes.slice(1).map(x=>x.index);
        }
        res.push(var_nodes[0]);
    }
    return res;
}

function process_connected_op_groups(op_groups, schedule, time_to_x_scale, threshold_x=10) {
    let res = [];
    schedule.forEach(row=>{
        let new_row = {};
        new_row.gpu_id = row.gpu_id;
        new_row.op_groups = [];
        if (row.op_groups.length < 2) {
            new_row.op_groups = row.op_groups;
            new_row.op_groups.forEach(op_index=>{
                op_groups[op_index]._end_time = op_groups[op_index].end_time;
                op_groups[op_index].connected_number = 1;
            });
        }
        else {
            let curr_index = 0;
            op_groups[row.op_groups[curr_index]]._end_time = op_groups[row.op_groups[curr_index]].end_time;
            op_groups[row.op_groups[curr_index]].connected_number = 1;
            for (let i = 1;i < row.op_groups.length;i++) {
                if (time_to_x_scale(op_groups[row.op_groups[i]].start_time) - time_to_x_scale(op_groups[row.op_groups[curr_index]]._end_time) < threshold_x) {
                    op_groups[row.op_groups[curr_index]]._end_time = op_groups[row.op_groups[i]].end_time;
                    op_groups[row.op_groups[curr_index]].connected_number += 1;
                }
                else {
                    new_row.op_groups.push(row.op_groups[curr_index]);
                    curr_index = i;
                    op_groups[row.op_groups[curr_index]]._end_time = op_groups[row.op_groups[curr_index]].end_time;
                    op_groups[row.op_groups[curr_index]].connected_number = 1;
                }
            }
            new_row.op_groups.push(row.op_groups[curr_index]);
        }
        res.push(new_row)
    });
    return res;
}

function process_op_groups(op_groups, schedule) {
    let res = [];
    schedule.forEach(row=>{
        let new_row = {};
        new_row.gpu_id = row.gpu_id;
        new_row.op_groups = [];

        row.op_groups.forEach(op_group=>{
            new_row.op_groups.push(op_group);
            // if (op_groups[op_group].end_time > 12.5 && op_groups[op_group].start_time < 46.5) {
            //     new_row.op_groups.push(op_group);
            // }
        });

        res.push(new_row)
    });
    return res;
}