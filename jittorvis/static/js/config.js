/*
 * Created by shouxing on 2018/1/16.
 */

var statistic_component;
var network_component;
var processed_data = [];
var WINDOW_HEIGHT = window.innerHeight - 40;
var WINDOW_WIDTH = window.innerWidth;
var REFRESH_PERIOD = 10;

Array.prototype.contains = function (obj) {
    let i = this.length;
    while (i--) {
        if (this[i] === obj) {
            return true;
        }
    }
    return false;
};

Array.prototype.indexOf = function(val) {
    for (let i = 0; i < this.length; i++) {
        if (this[i] === val) return i;
    }
    return -1;
};

Array.prototype.remove = function(val) {
    let index = this.indexOf(val);
    if (index > -1) {
        this.splice(index, 1);
    }
};

Array.prototype.sum = function () {
    return eval(this.join("+"));
};
