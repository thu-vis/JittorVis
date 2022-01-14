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
    return Math.floor(Math.random() * (max - min + 1)) + min; //The maximum is inclusive and the minimum is inclusive
}

function getLabelToClassMapping(labelSet) {
    var i = 0;
    var label2class = {};
    for (let e of labelSet.values()) {
        label2class[e] = i++;
    }
    return label2class;
}

var resultsColorSpace = 'Hex',
    resultsQuote = '';

var colorConversionFns = {
    Hex: function (c) { c = d3.rgb(c); return fullColorHex(parseInt(c.r), parseInt(c.g), parseInt(c.b)); },
    RGB: function (c) {
        c = d3.rgb(c);
        c = [norm255(parseInt(c.r)), norm255(parseInt(c.g)), norm255(parseInt(c.b))].join(',');
        return 'rgb(' + c + ')';
    },
    Lab: function (c) { return 'Lab(' + parseInt(c.L) + ',' + parseInt(c.a) + ',' + parseInt(c.b) + ')'; },
    LCH: function (c) {
        c = d3.hcl(c);
        c = [parseInt(c.l), Math.round(c.c), Math.round(c.h)].join(',');
        return 'LCH(' + c + ')';
    },
    HSL: function (c) {
        c = d3.hsl(c);
        c = [parseInt(c.h), c.s.toFixed(2), c.l.toFixed(2)].join(',');
        return 'HSL(' + c + ')';
    }
};

function downloadSvg() {
    let svgEl = d3.select("#renderDiv").select("svg")._groups[0][0];
    // let svgEl = document.getElementById(svgId);
    svgEl.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    var svgData = svgEl.outerHTML;
    var preface = '<?xml version="1.0" standalone="no"?>\r\n';
    var svgBlob = new Blob([preface, svgData], {
        type: "image/svg+xml;charset=utf-8"
    });
    // console.log(svgBlob);
    var svgUrl = URL.createObjectURL(svgBlob);
    var downloadLink = document.createElement("a");
    downloadLink.href = svgUrl;
    downloadLink.download = "result.svg";
    downloadLink.click();
    URL.revokeObjectURL(svgUrl);
}

function savePNG() {
    let scatterplot_svg = d3.select("#renderDiv").select("svg");
    let fileName = "result.png";
    let image = new Image;
    // get svg data
    var xml = new XMLSerializer().serializeToString(scatterplot_svg._groups[0][0]);
    // make it base64
    var svg64 = btoa(xml);
    var b64Start = 'data:image/svg+xml;base64,';
    // prepend a "header"
    var image64 = b64Start + svg64;
    //create a temporary canvas
    d3.select("#renderDiv").append("canvas").attr("id", "virtual_canvas")
        .attr("width", SVGWIDTH).attr("height", SVGHEIGHT)//.attr("style", "display:none");

    image.onload = function () {
        document.getElementById("virtual_canvas").getContext('2d').drawImage(image, 0, 0);
        var canvasElement = document.getElementById("virtual_canvas");
        var MIME_TYPE = "image/png";
        var imgURL = canvasElement.toDataURL(MIME_TYPE);

        var dlLink = document.createElement('a');
        dlLink.download = fileName;
        dlLink.href = imgURL;
        dlLink.dataset.downloadurl = [MIME_TYPE, dlLink.download, dlLink.href].join(':');

        document.body.appendChild(dlLink);
        dlLink.click();
        document.body.removeChild(dlLink);
        d3.select("#virtual_canvas").remove();
    };
    image.onerror = function () { console.log("Image failed!"); };
    // set it as the source of the img element
    image.src = image64;

}

/**
* set cookie
* @param cookName cookie name
* @param cookName cookie value
* @param expiredays expire time
*/
function setCookie(cookName, cookValue, expiredays) {
    var exdate = new Date();
    exdate.setTime(exdate.getTime() + expiredays * 24 * 3600 * 1000);
    var cookieVal = cookName + "=" + escape(cookValue) + ((expiredays == null) ? "" : ";expires=" + exdate.toGMTString()) + ";path=/";
    document.cookie = cookieVal;
}


/**
 * get cookie
 * @param cookName cookie name
 * @return
 */
function getCookie(cookName) {
    if (document.cookie.length > 0) {
        var c_start = document.cookie.indexOf(cookName + "=");
        if (c_start != -1) {
            return true;
        }
    }
    return false;
}

/**
 * delete cookie
 * @param cookName cookie name
 * @return
 */
function delCookie(cookName) {
    var exp = new Date();
    exp.setTime(exp.getTime());
    var cval = 0;
    document.cookie = cookName + "=" + cval + ";expires=" + exp.toGMTString();
}