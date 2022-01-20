/* eslint-disable */
/*
 * value accessor - returns the value to encode for a given data object.
 * scale - maps value to a visual display encoding, such as a pixel position.
 * map function - maps from data value to display value
 * axis - sets up axis
 */

// setup x
let xValue = function (d) {
    return d.x;
}; // data -> value
// setup y
let yValue = function (d) {
    return d.y;
}; // data -> value
// setup fill color
let cValue = function (d) {
    return d.label;
};

function norm255(v) {
    let normV = Math.max(0, v);
    normV = Math.min(normV, 255);
    return normV;
}
function normScope(v, vscope) {
    let normV = Math.max(vscope[0], v);
    normV = Math.min(normV, vscope[1]);
    return normV;
}

function inverseFunc(x) {
    //return Math.exp(-x);
    x = x == 0 ? 1 : x;
    return 1 / x;
}

//convert rgb to hex
var rgbToHex = function (rgb) {
    var hex = Number(rgb).toString(16);
    if (hex.length < 2) {
        hex = "0" + hex;
    }
    return hex;
};
var fullColorHex = function (r, g, b) {
    var red = rgbToHex(r);
    var green = rgbToHex(g);
    var blue = rgbToHex(b);
    return "#" + red + green + blue;
};

function euclidean_of_rgb(c1, c2) {
    var r_avg = (c1.r + c2.r) / 2,
        dR = c1.r - c2.r,
        dG = c1.g - c2.g,
        dB = c1.b - c2.b;
    var de = Math.sqrt(2 * dR * dR + 4 * dG * dG + 3 * dB * dB + r_avg * (dR * dR - dB * dB) / 256);

    return de;
}

class TupleDictionary {
    constructor() {
        this.dict = new Map();
    }

    tupleToString(tuple) {
        return tuple.join(",");
    }

    put(tuple, val) {
        this.dict.set(this.tupleToString(tuple), val);
    }

    get(tuple) {
        return this.dict.get(this.tupleToString(tuple));
    }

    keys() {
        return this.dict.keys();
    }

    length() {
        return this.dict.size;
    }
}

function getClassId(obj) {
    let legend_id = obj.attr("id");
    legend_id = (legend_id == undefined) ? 0 : legend_id;
    let removed_color_index = parseInt(legend_id.split("_")[1]);
    return removed_color_index;
}

//calculate distance of 2 colors
function calculateDistOf2Colors(palette) {
    let distanceOf2Colors = new TupleDictionary();
    let color_difference = function (lab1, lab2) {
        // let maxDistance = 122.48163103;
        // let minDistance = 1.02043527056;
        // let dis = (ciede2000(lab1, lab2) - minDistance) / (maxDistance - minDistance);
        let dis = d3_ciede2000(lab1, lab2)
        return dis;
    };
    let contrastToBg = function (lab1, lab2) {
        let c1 = d3.hcl(lab1),
            c2 = d3.hcl(lab2);
        if (!isNaN(c1.l) && !isNaN(c2.l)) {
            let dl = c1.l - c2.l;
            return Math.sqrt(dl * dl) / 100.0;
        } else {
            return 0;
        }
    }
    for (let i = 0; i < palette.length; i++) {
        for (let j = i + 1; j < palette.length; j++) {
            let dis = color_difference(d3mlab(palette[i]), d3mlab(palette[j]));
            distanceOf2Colors.put([i, j], dis);
        }
        distanceOf2Colors.put([i, palette.length], contrastToBg(palette[i], bgcolor));
    }
    return distanceOf2Colors;
}

function getRandomIntInclusive(min, max) {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(random() * (max - min + 1)) + min; //The maximum is inclusive and the minimum is inclusive
}

var seed = 1;
function random() {
    var x = Math.sin(seed++) * 10000;
    return x - Math.floor(x);
}

