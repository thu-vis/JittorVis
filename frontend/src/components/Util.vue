<script>
export default {
    name: 'Util',
    methods: {
        /**
         * Uses canvas.measureText to compute and return the width of the given text of given font in pixels.
         *
         * @param {String} text The text to be rendered.
         * @param {String} font The css font descriptor that text is to be rendered with (e.g. "bold 14px verdana").
         *
         * @return {int} text width(px)
         *
         * @see https://stackoverflow.com/questions/118241/calculate-text-width-with-javascript/21015393#21015393
         */
        getTextWidth: function(text, font) {
            // re-use canvas object for better performance
            const canvas = this.getTextWidth.canvas || (this.getTextWidth.canvas = document.createElement('canvas'));
            const context = canvas.getContext('2d');
            context.font = font;
            const metrics = context.measureText(text);
            return metrics.width;
        },
        toImage: function(feature) {
            // return array;
            window.feature = feature;
            const canvas = this.toImage.canvas || (this.toImage.canvas = document.createElement('canvas'));
            const context = canvas.getContext('2d');
            const height = feature.length;
            const width = feature[0].length;
            const depth = feature[0][0].length;
            canvas.width = width;
            canvas.height = height;
            const image = context.createImageData(width, height);
            const data = image.data;
            if (depth==1) {
                for (let i=0; i<data.length; i+=4) {
                    const v = feature[Math.floor(i/4/width)][(i/4)%width];
                    data[i] = v[0];
                    data[i+1] = v[0];
                    data[i+2] = v[0];
                    data[i+3] = 255;
                }
            } else {
                for (let i=0; i<data.length; i+=4) {
                    const v = feature[Math.floor(i/4/width)][(i/4)%width];
                    data[i] = v[0];
                    data[i+1] = v[1];
                    data[i+2] = v[2];
                    data[i+3] = 255;
                }
            }
            context.putImageData(image, 0, 0);
            return canvas.toDataURL();
        },
    },
};
</script>
