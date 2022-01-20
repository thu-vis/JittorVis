/* eslint-disable */

export {
    evaluatePalette,
    simulatedAnnealing2FindBestPalette,
}

var color_names_checked = [];
var best_color_names = {};

c3.load('./src/js/c3_data.json');
// color name lookup table
var color_name_map = {};
for (var c = 0; c < c3.color.length; ++c) {
    var x = c3.color[c];
    color_name_map[[x.L, x.a, x.b].join(",")] = c;
}
window.color_name_map = color_name_map;

var bgcolor = "#fff";
var score_importance_weight = [0, 0, 1];

var global_count = 1; // if there are more than <global_count> colors that are not in selected color names, then discard this palette  default: 2
var global_dec = 0.99; // 0.99: high efficiency   0.999: high quality
var global_color_dis = 10; // min dis between two color, default: 10
var max_count = 100;
var seed = 35;
function random() {
    var x = Math.sin(seed++) * 10000;
    return x - Math.floor(x);
}

/**
 * calculating the Color Saliency 
 * reference to "Color Naming Models for Color Selection, Image Editing and Palette Design"
 */

 function getColorNameIndex(c) {
    var x = d3mlab(c),
        L = 5 * Math.round(x.L / 5),
        a = 5 * Math.round(x.a / 5),
        b = 5 * Math.round(x.b / 5),
        s = [L, a, b].join(",");
    return color_name_map[s];
}
function getColorSaliency(x) {
    let c = getColorNameIndex(x);
    return (c3.color.entropy(c) - minE) / (maxE - minE);
}
function getNameDifference(x1, x2) {
    let c1 = getColorNameIndex(x1),
        c2 = getColorNameIndex(x2);
    return 1 - c3.color.cosine(c1, c2);
}

/**
 * alpha-Shape graph Implementation
 * using Philippe Rivière’s bl.ocks.org/1b7ddbcd71454d685d1259781968aefc 
 * voronoi.find(x,y) finds the nearest cell to the point (x,y).
 * extent is like: [[30, 30], [width - 30, height - 30]]
 */
function calculateAlphaShape(data, extent) {
    let voronoi = d3.voronoi().x(function (d) { return xMap(d); }).y(function (d) { return yMap(d); })
        .extent(extent);
    let diagram = voronoi(data);
    let cells = diagram.cells;
    let alpha = 25 * 25 * 2;
    let distanceDict = {};
    for (let cell of cells) {
        if(cell === undefined) continue;
        let label = labelToClass[cell.site.data.label];
        cell.halfedges.forEach(function (e) {
            let edge = diagram.edges[e];
            let ea = edge.left;
            if (ea === cell.site || !ea) {
                ea = edge.right;
            }
            if (ea) {
                let ea_label = labelToClass[ea.data.label];
                if (label != ea_label) {
                    let dx = cell.site[0] - ea[0],
                        dy = cell.site[1] - ea[1],
                        dist = dx * dx + dy * dy;
                    if (alpha > dist) {
                        if (distanceDict[label] === undefined)
                            distanceDict[label] = {};
                        if (distanceDict[label][ea_label] === undefined)
                            distanceDict[label][ea_label] = [];
                        distanceDict[label][ea_label].push(inverseFunc(Math.sqrt(dist)));
                    }
                }
            }
        });
    }


    var distanceOf2Clusters = new TupleDictionary();
    for (var i in distanceDict) {
        for (var j in distanceDict[i]) {
            i = +i, j = +j;
            var dist;
            if (distanceDict[j] === undefined || distanceDict[j][i] === undefined)
                dist = 2 * d3.sum(distanceDict[i][j]);
            else
                dist = d3.sum(distanceDict[i][j]) + d3.sum(distanceDict[j][i]);
            if (i < j)
                distanceOf2Clusters.put([i, j], dist);
            else
                distanceOf2Clusters.put([j, i], dist);
        }
    }


    return distanceOf2Clusters;
}

/**
 * score the given palette
 */
function evaluatePalette(palette) {
    if (color_names_checked != undefined && color_names_checked.length > 0) {
        let count = 0;
        for (let i = 0; i < palette.length; i++) {
            let c = getColorNameIndex(d3.rgb(palette[i])),
                t = c3.color.relatedTerms(c, 1);
            if (t[0] === undefined || color_names_checked.indexOf(c3.terms[t[0].index]) === -1) {
                count++;
            }
        }
        if (count > global_count) {// if there are more than <global_count> colors that are not in selected color names, then discard this palette
            return -1;
        }
    }
    // let class_distance = cd_weight;
    // calcualte color distance of given palette
    let class_discriminability = 0,
        name_difference = 0,
        color_discrimination_constraint = 100000;
    let dis;
    for (let i = 0; i < palette.length; i++) {
        for (let j = i + 1; j < palette.length; j++) {
            dis = d3_ciede2000(d3mlab(palette[i]), d3mlab(palette[j]));
            // if (class_distance.get([i, j]) != undefined)
            //     class_discriminability += class_distance.get([i, j]) * dis;
            let nd = getNameDifference(palette[i], palette[j]);
            name_difference += nd;
            color_discrimination_constraint = (color_discrimination_constraint > dis) ? dis : color_discrimination_constraint;
        }
        dis = d3_ciede2000(d3mlab(palette[i]), d3mlab(d3.rgb(bgcolor)));
        color_discrimination_constraint = (color_discrimination_constraint > dis) ? dis : color_discrimination_constraint;
    }
    // if (criterion_cd < 0)
    //     criterion_cd = class_discriminability;
    // class_discriminability /= criterion_cd;
    name_difference /= palette.length * (palette.length - 1) * 0.25;

    return (score_importance_weight[0] * class_discriminability + score_importance_weight[1] * name_difference + score_importance_weight[2] * (color_discrimination_constraint * 0.1));
}

/**
 * using simulated annealing to find the best palette of given data
 * @param {*} palette_size 
 * @param {*} evaluateFunc 
 * @param {*} colors_scope: hue range, lightness range, saturation range
 * @param {*} flag 
 */
function simulatedAnnealing2FindBestPalette(color_hex, palette_size, evaluateFunc, colors_scope = { "hue_scope": [0, 360], "lumi_scope": [25, 85] }, flag = true) {
    let c = getColorNameIndex(d3.rgb(color_hex)), t = c3.color.relatedTerms(c, 1);
    color_names_checked = [c3.terms[t[0].index]];
    best_color_names = {};
    best_color_names[c3.terms[t[0].index]] = d3.color(c3.terms[t[0].index]);
    console.log(best_color_names);

    let iterate_times = 0;
    //default parameters
    let max_temper = 100000,
        dec = global_dec, // 0.999
        max_iteration_times = 10000000,
        end_temper = 0.001;
    let cur_temper = max_temper;
    //generate a totally random palette
    let color_palette = getColorPaletteRandom(palette_size);
    let criterion_cd = -1.0;
    //evaluate the default palette
    let o = {
        id: color_palette,
        score: evaluateFunc(color_palette)
    },
    preferredObj = o;


    while (cur_temper > end_temper) {
        for (let i = 0; i < 1; i++) {//disturb at each temperature
            iterate_times++;
            color_palette = o.id.slice();
            disturbColors(color_palette, colors_scope);
            let color_palette_2 = color_palette.slice();
            let o2 = {
                id: color_palette_2,
                score: evaluateFunc(color_palette_2)
            };

            let delta_score = o.score - o2.score;
            if (delta_score <= 0 || delta_score > 0 && random() <= Math.exp((-delta_score) / cur_temper)) {
                o = o2;
                if (preferredObj.score - o.score < 0) {
                    preferredObj = o;
                }
            }
            if (iterate_times > max_iteration_times) { 
                break;
            }
        }

        cur_temper *= dec;
    }
    for(let color of preferredObj.id) {
        color.r = norm255(color.r);
        color.g = norm255(color.g);
        color.b = norm255(color.b);
    }
    return preferredObj;
}

function getColorPaletteRandom(palette_size) {
    let palette = [];
    for (let i = 0; i < palette_size; i++) {
        let rgb = d3.rgb(getRandomIntInclusive(0, 255), getRandomIntInclusive(0, 255), getRandomIntInclusive(0, 255));
        palette.push(rgb);
    }
    return palette;
}

function randomDisturbColors(palette, colors_scope) {
    let disturb_step = 50;
    // random disturb one color
    let idx = getRandomIntInclusive(0, palette.length - 1),
        rgb = d3.rgb(palette[idx]),
        color = d3.rgb(norm255(rgb.r + getRandomIntInclusive(-disturb_step, disturb_step)), norm255(rgb.g + getRandomIntInclusive(-disturb_step, disturb_step)), norm255(rgb.b + getRandomIntInclusive(-disturb_step, disturb_step))),
        hcl = d3.hcl(color);
    color = d3.rgb(d3.hcl(normScope(hcl.h, colors_scope.hue_scope), normScope(hcl.c, [0, 100]), normScope(hcl.l, colors_scope.lumi_scope)));
    palette[idx] = d3.rgb(norm255(color.r), norm255(color.g), norm255(color.b));
    let count = 0, sign;
    while (true) {
        while ((sign = isDiscriminative(palette)) > 0) {
            count += 1;
            if (count === max_count) {
                break;
            }
            rgb = d3.rgb(palette[sign])
            color = d3.rgb(norm255(rgb.r + getRandomIntInclusive(-disturb_step, disturb_step)), norm255(rgb.g + getRandomIntInclusive(-disturb_step, disturb_step)), norm255(rgb.b + getRandomIntInclusive(-disturb_step, disturb_step)))
            hcl = d3.hcl(color);
            // if (hcl.h >= 85 && hcl.h <= 114 && hcl.l >= 35 && hcl.l <= 75) {
            //     if (Math.abs(hcl.h - 85) > Math.abs(hcl.h - 114)) {
            //         hcl.h = 115;
            //     } else {
            //         hcl.h = 84;
            //     }
            // }
            palette[sign] = d3.rgb(d3.hcl(normScope(hcl.h, colors_scope.hue_scope), normScope(hcl.c, [0, 100]), normScope(hcl.l, colors_scope.lumi_scope)));
        }
        let satisfy_color_name = true;
        if (color_names_checked.length > 0) {
            for (let i = 0; i < palette.length; i++) {
                let c = getColorNameIndex(d3.rgb(palette[i])),
                    t = c3.color.relatedTerms(c, 1);
                if (t[0] === undefined || color_names_checked.indexOf(c3.terms[t[0].index]) === -1) {
                    rgb = best_color_names[color_names_checked[getRandomIntInclusive(0, color_names_checked.length - 1)]]
                    palette[i] = d3.rgb(norm255(rgb.r + getRandomIntInclusive(-10, 10)), norm255(rgb.g + getRandomIntInclusive(-10, 10)), norm255(rgb.b + getRandomIntInclusive(-10, 10)))
                    satisfy_color_name = false;
                }
            }
        }

        if (satisfy_color_name || count >= max_count) {
            break;
        }
    }
}


function isDiscriminative(palette) {
    let idx = -1;
    for (let i = 0; i < palette.length; i++) {
        for (let j = i + 1; j < palette.length; j++) {
            let color_dis = d3_ciede2000(d3mlab(palette[i]), d3mlab(palette[j]));
            if (color_dis < global_color_dis) { 
                return j;
            }
        }
    }
    return idx;
}

/**
 * only use color discrimination
 * @param {} palette 
 * @param {*} colors_scope 
 */
function disturbColors(palette, colors_scope) {
    if (random() < 0.5) {
        randomDisturbColors(palette, colors_scope);
    } else {
        // randomly shuffle two colors of the palette 
        let idx_0 = getRandomIntInclusive(0, palette.length - 1),
            idx_1 = getRandomIntInclusive(0, palette.length - 1);
        while (idx_0 === idx_1) {
            idx_1 = getRandomIntInclusive(0, palette.length - 1);
        }
        let tmp = palette[idx_0];
        palette[idx_0] = palette[idx_1];
        palette[idx_1] = tmp;
    }
}
