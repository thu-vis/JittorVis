/**
 * Modified by shouxing on 2018/3/4.
 */


function ColorManager() {
    let self = this;
    self.selection_max = 15;
    self.available_colors = [
    "#ffa953",
    "#55ff99",
    "#ba9b96",
    "#c982ce",
    "#bcbd22",
    "#e377c2",
    "#990099",
    "#8c564b",
    "#dc143c",
    "#008000",
    "#ff4500",
    "#ffff00",
    "#0000cd",
    "#4fa7ff",
    "#17becf"
    ];

    self.lighter_default_color = '#e7f4ff';
    self.default_color = '#bbdefb';
    self.darker_default_color = "#2097f6" //"#8bc7f8"
    self.tree_highlight_color = "#2097f6";
    self.disable_color = "gray" //"#8bc7f8"
    self.tree_default_color = "gray"
    self.darker_text_color = 'darkgray';
    self.text_color = '#666';
    self.node_border_color = 'rgb(186,187,184)';
    self.seg_node_color = '#4F4F4F';
    self.node_time_cost_unit_color = '#4F4F4F';
    self.node_btn_highlight_background_color = '#ccc';
    self.expand_btn_highlight_color = 'white';
    self.edge_color = 'rgb(186,187,184)';
    self.reverse_stroke_color = 'rgb(0,150,118)';//"#17ffcf";
    self.brother_node_highlight_color = "#ff7f00";
}


ColorManager.prototype.get_color_by_label = function (class_label) {
    let self = this;

    if (class_label < self.selection_max) {
        return self.available_colors[class_label];
    }
    else {
        return self.available_colors[class_label % self.selection_max];
        return self.default_color;
    }
};


ColorManager.prototype.get_color_by_exploring_height = function (exploring_height) {
    if (exploring_height === -1) {
        return 'white';
    }
    else if (exploring_height === -2) {
        return 'transparent';
    }
    else {
        let self = this;
        return self.old_get_color_by_exploring_height(exploring_height);
    }
    // else if (exploring_height === 0) {
    //     return `rgb(${240},${252},${255})`;
    // }
    // else if (exploring_height === 1) {
    //     return `rgb(${219},${245},${251})`;
    // }
    // else if (exploring_height === 2) {
    //     return `rgb(${198},${237},${248})`;
    // }
    // else if (exploring_height === 3) {
    //     return `rgb(${156},${222},${240})`;
    // }
};


ColorManager.prototype.old_get_color_by_exploring_height = function (exploring_height) {
    if (exploring_height === -1) {
        return 'white';
    }
    else if (exploring_height === -2) {
        return 'transparent';
    }
    let self = this;
    let start_color = [136 / 255 * 360, 255 / 255 * 100, 248 / 255 * 100];
    // let start_color = [64 / 255 * 360, 85 / 255 * 100, 228 / 255 * 100];
    let [R, G, B] = HSL2RGB(start_color[0], start_color[1], start_color[2] - 1500 / 255 * exploring_height);
    // let [R, G, B] = HSL2RGB(start_color[0], start_color[1], start_color[2] - 4500 / 255 * exploring_height);
    return `rgb(${R},${G},${B})`;
};

ColorManager.prototype.get_op_group_highlight_color = function (label) {
    let self = this;
    if (label === -1) {
        return self.default_color;
    }
    return self.get_color_by_label(label);
};

function HSL2RGB(H = 0, S = 0, L = 0) {
    let h= H/360;
    let s= S/100;
    let l= L/100;
    let rgb=[];

    if(s==0){
        rgb=[Math.round(l*255),Math.round(l*255),Math.round(l*255)];
    }else{
        let q=l>=0.5?(l+s-l*s):(l*(1+s));
        let p=2*l-q;
        let tr=rgb[0]=h+1/3;
        let tg=rgb[1]=h;
        let tb=rgb[2]=h-1/3;
        for(let i=0; i<rgb.length;i++){
            let tc=rgb[i];
            console.log(tc);
            if(tc<0){
                tc=tc+1;
            }else if(tc>1){
                tc=tc-1;
            }
            switch(true){
                case (tc<(1/6)):
                    tc=p+(q-p)*6*tc;
                    break;
                case ((1/6)<=tc && tc<0.5):
                    tc=q;
                    break;
                case (0.5<=tc && tc<(2/3)):
                    tc=p+(q-p)*(4-6*tc);
                    break;
                default:
                    tc=p;
                    break;
            }
            rgb[i]=Math.round(tc*255);
        }
    }

    return rgb;
};